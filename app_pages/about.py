"""About & data: cleaning funnel, coverage, sources, limitations, status."""
import pandas as pd
import plotly.express as px
import streamlit as st

from app_pages import common as C

C.require_data()
r = C.load_report()
s = r["model_scope_summary"]

st.title("About & data")
st.write("What FlyCheck knows, where it comes from, and where it stops.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("From raw log to labelled flights")
    funnel = pd.DataFrame([
        ("Raw board snapshots", r["raw_rows"], f"{r['malformed_lines_skipped']} truncated lines skipped"),
        ("Unique flight listings", r["unique_flight_listings"], "latest snapshot of each flight kept"),
        ("Physical flights", r["physical_flights"], "codeshares merged into one flight"),
        ("Departures", r["departures"], f"{r['departures_cancelled']} cancelled, "
                                        f"{r['departures_unresolved']} without a final status (set aside)"),
        ("Completed departures", r["departures_completed"],
         f"{r['completed_domestic_excluded']} domestic and {r['completed_suspect_early_excluded']} "
         "likely schedule change flagged out of scope"),
        ("Labelled for analysis and the model", r["model_scope_departures"],
         "completed international departures with a trusted delay"),
    ], columns=["Step", "Rows", "What happened"])
    st.dataframe(funnel, hide_index=True, width="stretch",
                 column_config={"Rows": st.column_config.NumberColumn(format="localized")})
    st.caption("Rows are flagged, never silently deleted. Full counts: data/processed/cleaning_report.json.")
with c2:
    st.subheader("Coverage of labelled departures")
    per_day = pd.DataFrame(list(s["per_day"].items()), columns=["date", "departures"])
    fig = px.bar(per_day, x="date", y="departures", text="departures", color_discrete_sequence=[C.SKY])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="category",
                      xaxis_title="Scheduled date", yaxis_title="Completed departures")
    st.plotly_chart(fig, width="stretch")
    st.caption("The archive logged the board on Jan 17, Mar 17, Jun 26 and Jul 3, 2026. "
               "TLV was closed from Feb 11 to May 15, 2026.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sources")
    st.markdown(
        "- **Flight archive**: IAA flight board snapshots archived by The Octopus "
        "([odata.org.il](https://www.odata.org.il/he/dataset/flydata)), CC BY.\n"
        "- **Airline reliability**: [TLVFlights](https://www.tlvflights.com/en/flight-statistics/) "
        "airline-level statistics, used with attribution.\n"
        "- **Price index**: Israel Central Bureau of Statistics, series 121390.\n"
        "- **Airport information**: Israel Airports Authority website.")
    st.subheader("Definitions")
    st.markdown(
        "- **Delay** = IAA board time − scheduled time, for departures whose final status is DEPARTED. "
        "Times are Israel local time.\n"
        "- **Delayed 30+** = delay of at least 30 minutes.\n"
        "- **Hall activity** = international departures scheduled 1–4 hours after you arrive, ranked "
        "against all hours of the day (0–100; low < 33 ≤ medium < 66 ≤ high).")
with c2:
    st.subheader("Limitations")
    st.markdown(
        f"- The labelled sample is small ({s['n']} departures) and comes from days just before and "
        "after the airport closure. It cannot show seasonality.\n"
        "- The board's actual time is close to takeoff, not gate departure.\n"
        "- Hall activity is a proxy from scheduled departures, not a measurement of queues or waiting time.\n"
        "- FlyCheck does not predict ticket prices or security wait times; no public data exists for them.\n"
        "- Larger sources have been requested (a year-long archive of the same board; EUROCONTROL "
        "research data). Analyses and the model will be updated if they arrive.\n"
        "- Historical patterns are not a guarantee about any future flight.")

st.subheader("Project status")
st.dataframe(pd.DataFrame([
    ("Data audit and cleaning pipeline", "Nov 18", "Done"),
    ("M2 · Analytics dashboard", "Nov 25", "This app"),
    ("M3 · Delay model trained, evaluated, integrated", "Dec 16", "Planned"),
    ("Deployment · Streamlit Community Cloud", "Dec 23", "Planned"),
    ("Final demo", "Jan 6, 2027", "Planned"),
], columns=["Step", "Due", "Status"]), hide_index=True, width="stretch")
