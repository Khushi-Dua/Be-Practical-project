"""
RetailIQ Application
=====================
Single Streamlit app covering the three required application surfaces from the brief:
  1. Executive Dashboard  - revenue/volume, category performance, store ranking,
                             promotional effectiveness, stock position, drill-through
  2. Forecast Explorer     - horizon selection, store/dept picker, forecast chart
  3. Conversational Agent  - natural-language Q&A with visible tool-selection reasoning

Run locally:   streamlit run app.py
Deploy:        push this file + agent.py + output/retailiq_clean.csv to a public repo,
                then deploy on Streamlit Community Cloud (free, public URL) pointing
                at app.py as the entrypoint.

Data dependency: expects output/retailiq_clean.csv (produced by the EDA/notebook).
If it isn't present, agent.py's build_star_schema() falls back to synthetic data
automatically, so this app runs standalone for testing/demo purposes.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import RetailIQAgent, build_star_schema, DB_PATH, CLEAN_CSV

st.set_page_config(page_title="RetailIQ", layout="wide", page_icon="\U0001F4CA")


# ---------------------------------------------------------------------------
# Cached resources - build schema + agent once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building star schema and loading agent...")
def get_agent():
    if not os.path.exists(DB_PATH):
        build_star_schema()
    return RetailIQAgent()


@st.cache_data(show_spinner=False)
def load_fact_frame():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT f.*, d.year, d.month, d.week, d.quarter, d.is_holiday, s.store_type
        FROM fact_sales f
        JOIN dim_date d ON f.date_key = d.date_key
        JOIN dim_store s ON f.store_id = s.store_id
        ORDER BY store_id, dept_id, date_key
    """, conn)
    conn.close()
    df["date_key"] = pd.to_datetime(df["date_key"])
    return df


agent = get_agent()
df = load_fact_frame()

if not os.path.exists(CLEAN_CSV):
    st.warning(
        f"'{CLEAN_CSV}' not found - the app is running on synthetic demo data generated "
        "by agent.py. Re-run the EDA notebook to export the real cleaned dataset, then "
        "restart this app to use real numbers.",
        icon="\u26A0\uFE0F",
    )

st.title("\U0001F4CA RetailIQ")
st.caption("Demand Forecasting and Multi-Tool Business Assistant \u2014 CP-05")

tab_dash, tab_forecast, tab_chat = st.tabs(
    ["\U0001F3E2 Executive Dashboard", "\U0001F4C8 Forecast Explorer", "\U0001F4AC Assistant"]
)


# ---------------------------------------------------------------------------
# TAB 1: Executive Dashboard
# ---------------------------------------------------------------------------
with tab_dash:
    stores = sorted(df["store_id"].unique().tolist())
    depts = sorted(df["dept_id"].unique().tolist())

    c1, c2 = st.columns(2)
    store_filter = c1.multiselect("Filter by store", stores, default=stores)
    dept_filter = c2.multiselect("Filter by department", depts, default=depts)

    fdf = df[df["store_id"].isin(store_filter) & df["dept_id"].isin(dept_filter)]

    total_revenue = fdf["weekly_sales"].sum()
    total_volume_weeks = fdf["date_key"].nunique()
    promo_share = fdf["promotion"].mean() * 100 if len(fdf) else 0
    avg_weekly = fdf.groupby("date_key")["weekly_sales"].sum().mean() if len(fdf) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${total_revenue:,.0f}")
    k2.metric("Avg Weekly Sales (all stores)", f"${avg_weekly:,.0f}")
    k3.metric("Weeks Covered", f"{total_volume_weeks}")
    k4.metric("Promo Week Share", f"{promo_share:.1f}%")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue Trend")
        trend = fdf.groupby("date_key")["weekly_sales"].sum().reset_index()
        fig = px.line(trend, x="date_key", y="weekly_sales", labels={"weekly_sales": "Sales ($)", "date_key": "Week"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Store Ranking")
        rank = fdf.groupby("store_id")["weekly_sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(rank, x="store_id", y="weekly_sales", labels={"weekly_sales": "Total Sales ($)", "store_id": "Store"})
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Category (Dept) Performance")
        cat = fdf.groupby("dept_id")["weekly_sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat, x="dept_id", y="weekly_sales", labels={"weekly_sales": "Total Sales ($)", "dept_id": "Dept"})
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.subheader("Promotional Effectiveness")
        promo_cmp = fdf.groupby("promotion")["weekly_sales"].mean().reset_index()
        promo_cmp["promotion"] = promo_cmp["promotion"].map({0: "No Promo", 1: "Promo"})
        fig = px.bar(promo_cmp, x="promotion", y="weekly_sales", labels={"weekly_sales": "Avg Weekly Sales ($)", "promotion": ""})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Stock Position (Inventory Cover) \u2014 drill-through by store/dept")
    st.caption("Inventory cover is a synthetic placeholder (see agent.py docstring) \u2014 swap in a real inventory feed when available.")
    dc1, dc2 = st.columns(2)
    drill_store = dc1.selectbox("Store", stores, key="drill_store")
    drill_dept = dc2.selectbox("Department", depts, key="drill_dept")
    cover = agent.sql_tool.inventory_cover(drill_store, drill_dept)
    st.dataframe(cover, use_container_width=True)

    st.subheader("Drill-through: weekly sales detail")
    detail = fdf[(fdf.store_id == drill_store) & (fdf.dept_id == drill_dept)][
        ["date_key", "weekly_sales", "promotion", "is_holiday", "temperature", "fuel_price"]
    ].sort_values("date_key")
    st.dataframe(detail, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2: Forecast Explorer
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast Explorer")
    f1, f2, f3 = st.columns(3)
    fc_store = f1.selectbox("Store", stores, key="fc_store")
    fc_dept = f2.selectbox("Department", depts, key="fc_dept")
    fc_horizon = f3.slider("Horizon (weeks)", 1, 12, 4)

    if st.button("Run Forecast", type="primary"):
        result = agent.forecast_tool.forecast(fc_store, fc_dept, fc_horizon)
        if "error" in result:
            st.error(result["error"])
        else:
            hist = df[(df.store_id == fc_store) & (df.dept_id == fc_dept)].sort_values("date_key").tail(20)
            future_dates = pd.date_range(
                hist["date_key"].max() + pd.Timedelta(weeks=1), periods=fc_horizon, freq="W"
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["date_key"], y=hist["weekly_sales"], name="History", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=future_dates, y=result["forecast"], name="Forecast", mode="lines+markers", line=dict(dash="dash")))
            fig.update_layout(xaxis_title="Week", yaxis_title="Sales ($)")
            st.plotly_chart(fig, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Model RMSE (validation)", f"${result['model_rmse']:,.0f}" if result["model_rmse"] else "n/a")
            m2.metric(f"Forecast \u2014 week 1 of {fc_horizon}", f"${result['forecast'][0]:,.0f}")
            st.dataframe(
                pd.DataFrame({"week": range(1, fc_horizon + 1), "date": future_dates, "forecast_sales": result["forecast"]}),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# TAB 3: Conversational Assistant
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask RetailIQ")
    st.caption(
        "Ask about sales analytics (SQL tool), demand forecasts (Forecast tool), "
        "or internal policy (Retrieval tool). The agent shows its tool-selection reasoning below each answer."
    )

    ac1, ac2, ac3 = st.columns(3)
    chat_store = ac1.number_input("Store (used if forecast is invoked)", min_value=int(min(stores)), max_value=int(max(stores)), value=int(stores[0]))
    chat_dept = ac2.number_input("Dept (used if forecast is invoked)", min_value=int(min(depts)), max_value=int(max(depts)), value=int(depts[0]))
    chat_horizon = ac3.number_input("Horizon weeks (used if forecast is invoked)", min_value=1, max_value=12, value=4)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    example_qs = [
        "What are the top 5 stores by total sales?",
        "What's the forecast for the next 4 weeks?",
        "What is our markdown policy?",
        "Show period-over-period growth by store",
    ]
    st.write("Try:", " \u00b7 ".join(f"`{q}`" for q in example_qs))

    question = st.chat_input("Ask a business question...")

    if question:
        st.session_state.chat_history.append(("user", question))
        result = agent.answer(question, store_id=chat_store, dept_id=chat_dept, horizon_weeks=chat_horizon)
        st.session_state.chat_history.append(("assistant", result))

    for role, content in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                ans = content["answer"]
                if isinstance(ans, list):
                    st.dataframe(pd.DataFrame(ans), use_container_width=True)
                elif isinstance(ans, dict):
                    st.json(ans)
                else:
                    st.write(ans)
                if "citation" in content:
                    st.caption(f"\U0001F4CE Source: {content['citation']}")
                with st.expander("Reasoning trace"):
                    for step in content["trace"]:
                        st.write("- " + step)
