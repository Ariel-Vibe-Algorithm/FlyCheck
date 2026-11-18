"""Analyze a flight: model output (when validated), historical reference, cancellation
risk and departure-hall activity at arrival."""
from datetime import date, datetime, time

import plotly.graph_objects as go
import streamlit as st

from app_pages import common as C
from src import analytics, predict

C.require_data()
flights = C.load_flights()
scope = analytics.labelled(flights)
tlv = C.load_tlvflights()
profile = C.load_profile()

st.title("Analyze a flight")
st.write("Enter the flight you are thinking of booking. FlyCheck shows how similar flights "
         "performed historically and how busy the departure hall usually is when you arrive.")

airlines = C.airline_options(scope, tlv)
dests = C.destination_options(scope)
c1, c2, c3, c4, c5 = st.columns(5)
airline = c1.selectbox("Airline", list(airlines), format_func=airlines.get, key="an_airline")
dest = c2.selectbox("Destination", list(dests), format_func=dests.get, key="an_dest")
day = c3.date_input("Departure date", value=date(2027, 1, 14), key="an_date")
dep = c4.time_input("Departure time (Israel)", value=time(18, 30), step=300, key="an_time")
lead = c5.selectbox("Arrive before departure", [3.0, 3.5, 4.0], key="an_lead",
                    format_func=lambda h: f"{h:g} h" + (" (official guidance)" if h == 3 else ""))

ref = analytics.delay_reference(scope, airline=airline, airport=dest, hour=dep.hour)
cong = analytics.congestion_at(dep, lead, profile)

match = scope.loc[scope["airport_code"] == dest, "country"]
dest_country = match.iloc[0] if len(match) else None
when = datetime.combine(day, dep)

left, mid, right = st.columns(3)
with left.container(border=True):
    st.markdown("**Delay risk · ML model**")
    res = predict.predict_probability("delay", airline, dest, dest_country, when)
    if res["probability"] is not None:
        st.metric("Probability of a 30+ min delay", f"{res['probability']:.0%}")
        st.caption("Statistical estimate from historical patterns, not a live status. "
                   "Model evaluation: Explore data → Baselines & models.")
    else:
        st.info("No model estimate yet. On the held-out test days no model clearly beat the "
                "simple airline-rate baseline, so FlyCheck shows the historical rates instead.")
        with st.expander("Details"):
            st.caption(res["reason"])

    st.markdown("**Cancellation risk**")
    cres = predict.predict_probability("cancellation", airline, dest, dest_country, when)
    if cres["probability"] is not None:
        st.metric("Probability of cancellation", f"{cres['probability']:.1%}")
    elif tlv is not None and airline in set(tlv["airline_code"]):
        t = tlv.set_index("airline_code").loc[airline]
        st.metric(f"{airline} cancellations (TLVFlights)", f"{t['cancel_rate']:.1%}")
        st.caption(f"Historical rate over {int(t['total_flights']):,} flights. Cancellations are rare "
                   "and mostly follow security events or airline decisions that booking-time data "
                   "cannot foresee.")
    else:
        st.caption("No cancellation history for this airline.")

with mid.container(border=True):
    st.markdown("**Historical reference** (share delayed 30+ min)")
    a, d, h = ref["airline"], ref["destination"], ref["hour_band"]
    st.metric(f"{airline} in the archive", C.fmt_rate(a), help="Completed international departures")
    st.caption(f"n = {a['n']}" + (" · too few flights to rely on" if a["n"] < 10 else ""))
    if tlv is not None and airline in set(tlv["airline_code"]):
        t = tlv.set_index("airline_code").loc[airline]
        st.metric(f"{airline} on TLVFlights", f"{t['late_30_rate']:.0%}")
        st.caption(f"n = {int(t['total_flights']):,} flights, Dec 2025 – Nov 2026 (larger sample)")
    st.metric(f"To {dest}", C.fmt_rate(d))
    st.caption(f"n = {d['n']}")
    lo, hi = h["band"]
    st.metric(f"Departures {lo:02d}:00–{hi:02d}:00", C.fmt_rate(h))
    st.caption(f"all airlines, n = {h['n']}")

with right.container(border=True):
    st.markdown("**Departure hall when you arrive**")
    color = C.LEVEL_COLORS[cong["level"]]
    st.markdown(f"### :{color}[{cong['level'].upper()}] · {cong['score']}/100")
    prev_day = " (the evening before)" if cong["crosses_midnight"] else ""
    st.write(f"You arrive at **{cong['arrival_time']:%H:%M}**{prev_day} on {day:%a %d %b %Y}. "
             f"About **{cong['window_departures']:.0f}** international departures are usually "
             f"scheduled in the following three hours.")
    st.caption("A relative activity index (0–100) from 9 days of schedule in Jun–Jul 2026. "
               "It is not a queue or waiting-time estimate.")

# Hourly profile chart
colors = [C.INK if hr == dep.hour else C.SIGNAL if hr in cong["window_hours"] else C.SKY_SOFT
          for hr in range(24)]
fig = go.Figure(go.Bar(
    x=[f"{hr:02d}" for hr in range(24)], y=profile.values, marker_color=colors,
    hovertemplate="%{x}:00 · %{y:.1f} departures/hour<extra></extra>"))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                  xaxis_title="Hour of day (Israel time)", yaxis_title="Avg departures per hour")
st.subheader("Typical international departures by hour")
st.plotly_chart(fig, width="stretch")
st.caption("Yellow: the three hours after you arrive (the congestion window). "
           "Dark: your departure hour. Average of scheduled departures on 9 complete days "
           "(Jun 25–28 and Jul 2–6, 2026).")

with st.container(border=True):
    st.markdown("**Live status · planned**")
    st.caption("When the flight is already on the official board (about 3–4 days ahead), the app "
               "will show its terminal and check-in counters from data.gov.il, labelled as "
               "current status rather than a forecast.")
st.caption(f"Weekday and seasonal effects are not shown yet: the archive covers too few days. "
           f"Selected date: {day:%A}.")
