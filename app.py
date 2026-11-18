"""FlyCheck — Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="FlyCheck", page_icon="✈️", layout="wide")

pages = [
    st.Page("app_pages/analyze.py", title="Analyze a flight", icon=":material/flight_takeoff:", default=True),
    st.Page("app_pages/compare.py", title="Compare flights", icon=":material/compare_arrows:"),
    st.Page("app_pages/explore.py", title="Explore data", icon=":material/bar_chart:"),
    st.Page("app_pages/guide.py", title="Airport guide", icon=":material/info:"),
    st.Page("app_pages/about.py", title="About & data", icon=":material/dataset:"),
]

with st.sidebar:
    st.markdown("### ✈️ FlyCheck")
    st.caption("Know before you book · Ben Gurion (TLV) departures")

st.navigation(pages).run()

st.sidebar.caption(
    "Historical decision support, not a live tracker. "
    "Academic MVP, Ariel University, 2026."
)
