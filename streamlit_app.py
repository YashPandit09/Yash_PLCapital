"""
streamlit_app.py – Interactive Web UI for the Portfolio Analytics Agent.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.agent import PortfolioAgent, AgentResult
from src.exposure_tool import ExposureResult
from src.sql_tool import SQLResult

st.set_page_config(
    page_title="Portfolio Analytics Agent",
    page_icon="📊",
    layout="wide",
)

# Custom header styling
st.title("📊 Portfolio Analytics Agent")
st.caption("AI-powered portfolio analytics: text-to-SQL querying and sector exposure analysis.")

# Initialize agent in session state
if "agent" not in st.session_state:
    with st.spinner("Initializing database and agent..."):
        st.session_state.agent = PortfolioAgent()

agent: PortfolioAgent = st.session_state.agent

# Sidebar with sample questions and details
with st.sidebar:
    st.header("💡 Sample Questions")
    sample_questions = [
        "How many portfolios do we have in total?",
        "What are the names of all active portfolios?",
        "Which securities are in the Technology sector?",
        "What is the total Assets Under Management (AUM) for portfolios with high target risk level?",
        "Show me the top 5 holdings by cost basis in the Growth Equity Fund",
        "What is the average current price of securities in each sector?",
        "What are the sector exposures for the Tech Innovation Fund?",
        "Calculate the sector exposure breakdown for international equity",
    ]
    for sq in sample_questions:
        if st.button(sq, use_container_width=True):
            st.session_state.selected_question = sq

    st.divider()
    st.subheader("ℹ️ System Architecture")
    st.markdown(
        """
        - **In-Memory SQLite**: 9 relational tables with foreign keys
        - **Intelligent Routing**: Natural language query classifier
        - **Text-to-SQL Tool**: Schema-aware SQL generation with EXPLAIN safety checks
        - **Exposure Calculator**: Deterministic equity-renormalized exposure
        """
    )

# Input box
default_q = st.session_state.get("selected_question", "")
question = st.text_input(
    "Ask a question about your portfolio data:",
    value=default_q,
    placeholder="e.g. What are the sector exposures for the Tech Innovation Fund?",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("Submit Query", type="primary", use_container_width=True)

if ask_clicked and question:
    with st.spinner("Analyzing..."):
        result: AgentResult = agent.answer_question(question)

    st.subheader("Results")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Tool Selected", result.tool_used)
    m2.metric("Latency", f"{result.latency_ms:.1f} ms")
    status_label = "Error" if result.error else "Success"
    m3.metric("Status", status_label)

    if result.error and result.tool_used == "none":
        st.warning(result.answer or result.error)
    elif result.error:
        st.error(f"Error: {result.error}")
    else:
        # Tool-specific rendering
        if result.tool_used == "sql_query":
            raw_sql: SQLResult = result.raw_result
            with st.expander("🔍 Generated SQL Query", expanded=True):
                st.code(raw_sql.sql, language="sql")

            if raw_sql.columns and raw_sql.rows:
                df = pd.DataFrame(raw_sql.rows, columns=raw_sql.columns)
                st.dataframe(df, use_container_width=True)
                st.caption(f"Returned {len(df)} row(s).")
            else:
                st.info("Query executed successfully, returning 0 rows.")

        elif result.tool_used == "exposure_calculator":
            raw_exp: ExposureResult = result.raw_result
            st.write(f"**Portfolio:** {raw_exp.portfolio_name} (ID: {raw_exp.portfolio_id})")
            st.write(
                f"*Equity Weight:* {raw_exp.equity_weight_total}% of portfolio | "
                f"*Bonds Excluded:* {raw_exp.bonds_excluded}"
            )

            if raw_exp.exposure_pct_of_equity:
                df_exp = pd.DataFrame([
                    {
                        "Sector": sector,
                        "% of Equity (Renormalized)": eq_pct,
                        "% of Portfolio": raw_exp.exposure_pct_of_portfolio.get(sector, 0.0),
                    }
                    for sector, eq_pct in raw_exp.exposure_pct_of_equity.items()
                ])
                st.dataframe(df_exp, use_container_width=True)

                # Bar chart visualization
                st.bar_chart(
                    df_exp.set_index("Sector")[["% of Equity (Renormalized)", "% of Portfolio"]],
                    use_container_width=True,
                )
            else:
                st.info(raw_exp.error or "No equity holdings found.")
        else:
            st.write(result.answer)

