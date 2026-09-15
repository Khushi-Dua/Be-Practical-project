"""
RetailIQ Agent Layer
=====================
Planner-executor agent exposing three tools:
  1. SQLTool       -> translates business questions into queries against a star schema
  2. ForecastTool   -> invokes a trained model for a given store/dept and horizon
  3. RetrievalTool  -> answers policy questions from internal documentation, with citation

Design notes (for defence):
- Tool selection is rule-based (keyword/intent matching), NOT an LLM call. This mirrors
  the no-external-LLM-API approach used in the ParcelPilot assessment: it is deterministic,
  free to run, and every routing decision can be explained line-by-line under questioning.
  If you want LLM-based routing instead, swap `Planner.route()` for a single classification
  call to an LLM and keep everything else (tools, executor, trace) unchanged.
- The star schema is built from output/retailiq_clean.csv (the file your EDA notebook
  already exports). If that file isn't present, a small synthetic dataset with the same
  schema is generated so this script runs standalone for testing.
- ForecastTool trains a lightweight RandomForest on the fact table using the same
  lag/rolling features as your notebook, so this file doesn't depend on a saved model
  artifact. Swap `ForecastTool._train_or_load()` to load your saved .keras/.pkl model
  if you'd rather reuse the notebook's trained model directly.
- inventory cover uses a SYNTHETIC starting-stock table (documented inline) because the
  Walmart dataset has no real inventory feed. Say this explicitly in your defence -
  it's a placeholder wired to real inventory data if/when you have it.
"""

import os
import re
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

DB_PATH = "output/retailiq_star.db"
CLEAN_CSV = "output/retailiq_clean.csv"


# ---------------------------------------------------------------------------
# 0. Synthetic fallback data (only used if retailiq_clean.csv is missing)
# ---------------------------------------------------------------------------
def _make_synthetic_clean_csv(path, n_stores=5, n_depts=6, n_weeks=60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-05", periods=n_weeks, freq="W-FRI")
    rows = []
    for store in range(1, n_stores + 1):
        store_type = ["A", "B", "C"][store % 3]
        for dept in range(1, n_depts + 1):
            base = rng.uniform(8000, 25000)
            for i, d in enumerate(dates):
                seasonal = 1 + 0.25 * np.sin(2 * np.pi * i / 52)
                promo = int(rng.random() < 0.2)
                holiday = int(d.month in (11, 12) and rng.random() < 0.3)
                sales = max(
                    0,
                    base * seasonal
                    * (1.35 if promo else 1.0)
                    * (1.15 if holiday else 1.0)
                    + rng.normal(0, base * 0.05),
                )
                rows.append({
                    "Store": store, "Dept": dept, "Date": d,
                    "Weekly_Sales": round(sales, 2),
                    "IsHoliday": holiday, "Promotion": promo,
                    "Temperature": round(rng.uniform(30, 90), 1),
                    "Fuel_Price": round(rng.uniform(2.5, 4.2), 2),
                    "CPI": round(rng.uniform(180, 230), 2),
                    "Unemployment": round(rng.uniform(5, 9), 2),
                    "Type": store_type,
                    "Year": d.year, "Month": d.month,
                    "Week": int(d.isocalendar().week), "Quarter": d.quarter,
                })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# 1. Star schema build
# ---------------------------------------------------------------------------
def build_star_schema(csv_path=CLEAN_CSV, db_path=DB_PATH):
    if not os.path.exists(csv_path):
        print(f"[build_star_schema] {csv_path} not found - generating synthetic data for demo.")
        df = _make_synthetic_clean_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, parse_dates=["Date"])

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS fact_sales;
    DROP TABLE IF EXISTS dim_store;
    DROP TABLE IF EXISTS dim_dept;
    DROP TABLE IF EXISTS dim_date;
    DROP TABLE IF EXISTS dim_inventory;

    CREATE TABLE dim_store (
        store_id INTEGER PRIMARY KEY,
        store_type TEXT
    );
    CREATE TABLE dim_dept (
        dept_id INTEGER PRIMARY KEY
    );
    CREATE TABLE dim_date (
        date_key TEXT PRIMARY KEY,
        year INTEGER, month INTEGER, week INTEGER, quarter INTEGER,
        is_holiday INTEGER
    );
    CREATE TABLE fact_sales (
        store_id INTEGER, dept_id INTEGER, date_key TEXT,
        weekly_sales REAL, promotion INTEGER,
        temperature REAL, fuel_price REAL, cpi REAL, unemployment REAL,
        FOREIGN KEY (store_id) REFERENCES dim_store(store_id),
        FOREIGN KEY (dept_id) REFERENCES dim_dept(dept_id),
        FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
    );
    -- SYNTHETIC placeholder: starting stock per store/dept, used only to illustrate
    -- an inventory-cover query. Replace with a real inventory feed when available.
    CREATE TABLE dim_inventory (
        store_id INTEGER, dept_id INTEGER, stock_units REAL
    );
    """)

    dim_store = df[["Store", "Type"]].drop_duplicates().rename(
        columns={"Store": "store_id", "Type": "store_type"})
    dim_store.to_sql("dim_store", conn, if_exists="append", index=False)

    dim_dept = df[["Dept"]].drop_duplicates().rename(columns={"Dept": "dept_id"})
    dim_dept.to_sql("dim_dept", conn, if_exists="append", index=False)

    dim_date = df[["Date", "Year", "Month", "Week", "Quarter", "IsHoliday"]].drop_duplicates(subset=["Date"])
    dim_date = dim_date.rename(columns={
        "Date": "date_key", "Year": "year", "Month": "month",
        "Week": "week", "Quarter": "quarter", "IsHoliday": "is_holiday"})
    dim_date["date_key"] = dim_date["date_key"].astype(str)
    dim_date.to_sql("dim_date", conn, if_exists="append", index=False)

    fact = df[["Store", "Dept", "Date", "Weekly_Sales", "Promotion",
               "Temperature", "Fuel_Price", "CPI", "Unemployment"]].copy()
    fact["Date"] = fact["Date"].astype(str)
    fact = fact.rename(columns={
        "Store": "store_id", "Dept": "dept_id", "Date": "date_key",
        "Weekly_Sales": "weekly_sales", "Promotion": "promotion",
        "Temperature": "temperature", "Fuel_Price": "fuel_price",
        "CPI": "cpi", "Unemployment": "unemployment"})
    fact.to_sql("fact_sales", conn, if_exists="append", index=False)

    # synthetic starting stock ~= 6 weeks of average recent sales in units (proxy: sales/50)
    recent = fact.sort_values("date_key").groupby(["store_id", "dept_id"]).tail(8)
    avg_recent = recent.groupby(["store_id", "dept_id"])["weekly_sales"].mean().reset_index()
    avg_recent["stock_units"] = (avg_recent["weekly_sales"] / 50 * 6).round(1)
    avg_recent[["store_id", "dept_id", "stock_units"]].to_sql(
        "dim_inventory", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"[build_star_schema] Star schema built at {db_path} ({len(fact)} fact rows).")


# ---------------------------------------------------------------------------
# 2. SQL Tool
# ---------------------------------------------------------------------------
class SQLTool:
    """Maps a business question to a parametrised SQL template against the star schema."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def _run(self, sql, params=()):
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def sales_by_dimension(self, dimension="store_id", top_n=None):
        col = {"store": "store_id", "dept": "dept_id"}.get(dimension, dimension)
        sql = f"""
            SELECT {col}, ROUND(SUM(weekly_sales), 2) AS total_sales
            FROM fact_sales GROUP BY {col} ORDER BY total_sales DESC
        """
        if top_n:
            sql += f" LIMIT {int(top_n)}"
        return self._run(sql)

    def top_n_per_group(self, group_col="store_id", metric_col="dept_id", n=3):
        sql = f"""
            SELECT * FROM (
                SELECT {group_col}, {metric_col}, SUM(weekly_sales) AS total_sales,
                       RANK() OVER (PARTITION BY {group_col} ORDER BY SUM(weekly_sales) DESC) AS rnk
                FROM fact_sales GROUP BY {group_col}, {metric_col}
            ) WHERE rnk <= {int(n)}
        """
        return self._run(sql)

    def period_over_period_growth(self, group_col="store_id"):
        sql = f"""
            WITH monthly AS (
                SELECT {group_col}, d.year, d.month, SUM(f.weekly_sales) AS sales
                FROM fact_sales f JOIN dim_date d ON f.date_key = d.date_key
                GROUP BY {group_col}, d.year, d.month
            )
            SELECT {group_col}, year, month, sales,
                   LAG(sales) OVER (PARTITION BY {group_col} ORDER BY year, month) AS prev_sales,
                   ROUND(100.0 * (sales - LAG(sales) OVER (PARTITION BY {group_col} ORDER BY year, month))
                         / NULLIF(LAG(sales) OVER (PARTITION BY {group_col} ORDER BY year, month), 0), 2)
                         AS pct_growth
            FROM monthly ORDER BY {group_col}, year, month
        """
        return self._run(sql)

    def running_total(self, store_id):
        sql = """
            SELECT date_key, weekly_sales,
                   SUM(weekly_sales) OVER (ORDER BY date_key) AS running_total
            FROM fact_sales WHERE store_id = ? ORDER BY date_key
        """
        return self._run(sql, (store_id,))

    def inventory_cover(self, store_id, dept_id):
        """weeks of cover = current stock / avg weekly sales (last 8 weeks). Synthetic stock - see dim_inventory."""
        sql = """
            SELECT i.stock_units,
                   (SELECT AVG(weekly_sales) FROM fact_sales
                    WHERE store_id = ? AND dept_id = ?
                    ORDER BY date_key DESC LIMIT 8) AS avg_weekly_sales
            FROM dim_inventory i WHERE i.store_id = ? AND i.dept_id = ?
        """
        res = self._run(sql, (store_id, dept_id, store_id, dept_id))
        if res.empty or res["avg_weekly_sales"].iloc[0] in (None, 0):
            return res
        res["weeks_of_cover"] = (
            res["stock_units"] / (res["avg_weekly_sales"] / 50)
        ).round(1)  # sales -> units proxy, same /50 factor as dim_inventory build
        return res


# ---------------------------------------------------------------------------
# 3. Forecast Tool
# ---------------------------------------------------------------------------
class ForecastTool:
    """Predicts weekly sales for a store/dept N weeks ahead using lag/rolling features."""

    FEATURES = ["store_id", "dept_id", "year", "month", "week", "quarter",
                "is_holiday", "promotion", "temperature", "fuel_price",
                "cpi", "unemployment", "lag_1", "lag_2", "lag_4", "lag_8",
                "rolling_4", "rolling_8"]

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.model = None
        self.rmse = None
        self._train()

    def _load_fact_frame(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT f.*, d.year, d.month, d.week, d.quarter, d.is_holiday
            FROM fact_sales f JOIN dim_date d ON f.date_key = d.date_key
            ORDER BY store_id, dept_id, date_key
        """, conn)
        conn.close()
        df["date_key"] = pd.to_datetime(df["date_key"])
        g = df.groupby(["store_id", "dept_id"])["weekly_sales"]
        df["lag_1"] = g.shift(1)
        df["lag_2"] = g.shift(2)
        df["lag_4"] = g.shift(4)
        df["lag_8"] = g.shift(8)
        df["rolling_4"] = g.transform(lambda x: x.shift(1).rolling(4).mean())
        df["rolling_8"] = g.transform(lambda x: x.shift(1).rolling(8).mean())
        return df

    def _train(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        df = self._load_fact_frame().dropna(subset=self.FEATURES + ["weekly_sales"])
        if df.empty or len(df) < 20:
            print("[ForecastTool] Not enough rows to train - forecast tool disabled.")
            return
        split = df["date_key"].quantile(0.85)
        train, test = df[df["date_key"] < split], df[df["date_key"] >= split]
        self.model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        self.model.fit(train[self.FEATURES], train["weekly_sales"])
        if not test.empty:
            preds = self.model.predict(test[self.FEATURES])
            self.rmse = float(np.sqrt(mean_squared_error(test["weekly_sales"], preds)))
        self._latest = df.sort_values("date_key").groupby(["store_id", "dept_id"]).tail(1)

    def forecast(self, store_id, dept_id, horizon_weeks=1):
        if self.model is None:
            return {"error": "model not trained"}
        row = self._latest[
            (self._latest.store_id == store_id) & (self._latest.dept_id == dept_id)
        ]
        if row.empty:
            return {"error": f"no history for store {store_id}, dept {dept_id}"}
        row = row.iloc[0].copy()
        preds = []
        for h in range(1, horizon_weeks + 1):
            x = row[self.FEATURES].to_frame().T
            pred = float(self.model.predict(x)[0])
            preds.append(round(pred, 2))
            # roll features forward for a naive multi-step forecast
            row["lag_8"], row["lag_4"], row["lag_2"], row["lag_1"] = (
                row["lag_4"], row["lag_2"], row["lag_1"], pred)
            row["rolling_4"] = np.mean([row["lag_1"], row["lag_2"], row["lag_4"]])
            row["week"] = int(row["week"]) + 1 if row["week"] < 52 else 1
        return {
            "store_id": store_id, "dept_id": dept_id,
            "horizon_weeks": horizon_weeks, "forecast": preds,
            "model_rmse": round(self.rmse, 2) if self.rmse else None,
        }


# ---------------------------------------------------------------------------
# 4. Retrieval Tool (internal documentation)
# ---------------------------------------------------------------------------
POLICY_DOCS = {
    "inventory_policy.txt": (
        "Inventory Policy. Reorder point is set at two weeks of average demand plus "
        "safety stock of one week. Departments below 1.5 weeks of cover are flagged "
        "for expedited replenishment. Stock counts are reconciled weekly."
    ),
    "markdown_rules.txt": (
        "Markdown Rules. Items with more than eight weeks of inventory cover and "
        "declining weekly sell-through are eligible for a first markdown of 20 percent. "
        "A second markdown of 40 percent applies after a further four weeks without "
        "sufficient sell-through improvement. Markdowns require category manager sign-off."
    ),
    "supplier_terms.txt": (
        "Supplier Terms. Standard payment terms are net 30 days from delivery. "
        "Suppliers must confirm shipment within 48 hours of a purchase order and "
        "provide advance shipping notices for orders above 500 units."
    ),
    "returns_policy.txt": (
        "Returns Policy. Customer returns are accepted within 30 days with a receipt. "
        "Returned stock is inspected before being returned to sellable inventory; "
        "damaged units are written off and recorded against the receiving store."
    ),
    "sop_replenishment.txt": (
        "Standard Operating Procedure - Replenishment. Store managers review "
        "inventory cover every Monday. Departments under reorder point trigger an "
        "automatic purchase order unless overridden by regional planning."
    ),
}


class RetrievalTool:
    def __init__(self, docs=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = docs or POLICY_DOCS
        self.names = list(self.docs.keys())
        self.texts = list(self.docs.values())
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=2):
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sims.argsort()[::-1][:top_k]
        return [
            {"source": self.names[i], "text": self.texts[i], "score": round(float(sims[i]), 3)}
            for i in ranked if sims[i] > 0
        ]


# ---------------------------------------------------------------------------
# 5. Planner - rule-based intent routing
# ---------------------------------------------------------------------------
class Planner:
    FORECAST_KEYWORDS = ["forecast", "predict", "next week", "next n weeks", "expected sales", "projection"]
    POLICY_KEYWORDS = ["policy", "markdown", "return", "supplier", "sop", "reorder", "procedure", "terms"]
    SQL_KEYWORDS = ["top", "growth", "running total", "total sales", "compare", "rank", "cover", "trend"]

    def route(self, question):
        q = question.lower()
        if any(k in q for k in self.POLICY_KEYWORDS):
            return "retrieval", f"matched policy keyword(s) in: '{question}'"
        if any(k in q for k in self.FORECAST_KEYWORDS):
            return "forecast", f"matched forecast keyword(s) in: '{question}'"
        return "sql", f"defaulted to SQL analytics - matched: {[k for k in self.SQL_KEYWORDS if k in q] or 'no explicit keyword, general analytics question'}"


# ---------------------------------------------------------------------------
# 6. Executor - ties planner + tools together, returns a grounded answer + trace
# ---------------------------------------------------------------------------
class RetailIQAgent:
    def __init__(self, db_path=DB_PATH):
        if not os.path.exists(db_path):
            build_star_schema(db_path=db_path)
        self.sql_tool = SQLTool(db_path)
        self.forecast_tool = ForecastTool(db_path)
        self.retrieval_tool = RetrievalTool()
        self.planner = Planner()

    def answer(self, question, store_id=1, dept_id=1, horizon_weeks=4):
        trace = []
        tool, reason = self.planner.route(question)
        trace.append(f"Planner selected tool: '{tool}' ({reason})")

        if tool == "retrieval":
            hits = self.retrieval_tool.retrieve(question)
            trace.append(f"Retrieved {len(hits)} passage(s) from internal documentation")
            if not hits:
                return {"answer": "No relevant policy document found.", "trace": trace}
            answer = hits[0]["text"]
            citation = hits[0]["source"]
            trace.append(f"Top match: {citation} (similarity {hits[0]['score']})")
            return {"answer": answer, "citation": citation, "trace": trace}

        if tool == "forecast":
            trace.append(f"Calling ForecastTool(store={store_id}, dept={dept_id}, horizon={horizon_weeks})")
            result = self.forecast_tool.forecast(store_id, dept_id, horizon_weeks)
            trace.append(f"Forecast tool returned: {result}")
            return {"answer": result, "trace": trace}

        trace.append("Calling SQLTool.sales_by_dimension(dimension='store_id', top_n=5)")
        result = self.sql_tool.sales_by_dimension("store_id", top_n=5)
        trace.append(f"SQL returned {len(result)} row(s)")
        return {"answer": result.to_dict(orient="records"), "trace": trace}


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_star_schema()
    agent = RetailIQAgent()

    demo_questions = [
        "What are the top 5 stores by total sales?",
        "What's the forecast for store 1 dept 1 over the next 4 weeks?",
        "What is our markdown policy for slow-moving stock?",
    ]
    for q in demo_questions:
        print("\n" + "=" * 70)
        print("Q:", q)
        result = agent.answer(q, store_id=1, dept_id=1, horizon_weeks=4)
        print("\nReasoning trace:")
        for step in result["trace"]:
            print(" -", step)
        print("\nAnswer:", result["answer"])
        if "citation" in result:
            print("Cited source:", result["citation"])
