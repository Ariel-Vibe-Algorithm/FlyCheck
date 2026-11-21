"""Explore data (M2 dashboard): interactive historical analytics with filters."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_pages import common as C
from src import analytics

C.require_data()
flights = C.load_flights()
scope_all = analytics.labelled(flights)
tlv = C.load_tlvflights()
price = C.load_price_index()

WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

st.title("Explore the data")
st.write("Completed international departures from TLV in the open archive, after cleaning. "
         "Use the filters in the sidebar. Every rate is shown with its sample size (n).")

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.subheader("Filters")
    airline_names = C.airline_options(scope_all, None)
    sel_air = st.multiselect("Airlines", list(airline_names), format_func=airline_names.get,
                             placeholder="All airlines", key="ex_air")
    countries = sorted(scope_all["country"].dropna().str.title().unique())
    sel_cty = st.multiselect("Destination countries", countries, placeholder="All countries", key="ex_cty")
    hours = st.slider("Scheduled departure hour", 0, 23, (0, 23), key="ex_hours")
    sel_wd = st.multiselect("Weekdays", WEEKDAYS, placeholder="All weekdays", key="ex_wd")
    min_n = st.slider("Minimum flights per group in charts", 1, 30, 10, key="ex_min_n")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["hour"].between(*hours)
    if sel_air:
        mask &= df["airline_code"].isin(sel_air)
    if sel_cty:
        mask &= df["country"].str.title().isin(sel_cty)
    if sel_wd:
        mask &= df["weekday"].isin(sel_wd)
    return df[mask]


scope = apply_filters(scope_all)
departures = apply_filters(analytics.international(flights[flights["direction"] == "D"]))

if scope.empty:
    st.warning("No completed departures match these filters. Widen the filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- KPIs
summary = analytics.rate(scope)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Completed departures (n)", f"{summary['n']:,}")
k2.metric("Delayed 30+ min", C.fmt_rate(summary))
k3.metric("Median delay", f"{summary['median_delay']:.0f} min")
cancelled = int((departures["outcome"] == "cancelled").sum())
resolved = int(departures["outcome"].isin(["completed", "cancelled"]).sum())
k4.metric("Cancelled", f"{cancelled}", help=f"Out of {resolved} departures with a final status")

# ---------------------------------------------------------------- story: insight + baseline (M2)
def late_share(df: pd.DataFrame) -> float:
    return float(df["delayed_30"].astype(float).mean())


s_insight, s_baseline = st.columns(2)
with s_insight.container(border=True):
    st.markdown("**Insight: delays build up during the day**")
    morning, evening = scope[scope["hour"] < 12], scope[scope["hour"] >= 18]
    if len(morning) >= 20 and len(evening) >= 20:
        st.markdown(f"Departures scheduled before noon left 30+ min late **{late_share(morning):.0%}** of "
                    f"the time (n = {len(morning)}). After 18:00 it was **{late_share(evening):.0%}** "
                    f"(n = {len(evening)}).")
    else:
        st.caption("Too few morning or evening departures with these filters to compare. Widen the filters.")
    ly = scope_all[scope_all["airline_code"] == "LY"]
    ly_m, ly_e = ly[ly["hour"] < 12], ly[ly["hour"] >= 18]
    st.caption(f"It is not only the airline mix: El Al alone went from {late_share(ly_m):.0%} (n = {len(ly_m)}) "
               f"to {late_share(ly_e):.0%} (n = {len(ly_e)}). For a traveler who cares about delays, a "
               "morning flight is the safer bet.")

with s_baseline.container(border=True):
    st.markdown("**Baseline the M3 model must beat**")
    metrics = C.load_metrics()
    if metrics is None or "delay" not in metrics["tasks"]:
        st.caption("Run `python -m src.train` to compute the baselines.")
    else:
        task = metrics["tasks"]["delay"]
        best = task["best_baseline"]
        r = task["results"][best]["test"]
        overall = task["results"]["Overall rate"]["test"]
        st.markdown(f"**{best}**: ROC-AUC **{r['roc_auc']:.2f}**, Brier **{r['brier']:.3f}** on the "
                    f"{task['split']['test_rows']} latest departures (test days, never used for training).")
        st.caption(f"The simplest baseline, the overall rate for every flight, scores Brier "
                   f"{overall['brier']:.3f} and ROC-AUC 0.50. No model beats the airline rate yet; "
                   "see the Baselines & models tab.")

tab_delay, tab_groups, tab_activity, tab_context, tab_models, tab_data = st.tabs(
    ["Delays", "Airlines & destinations", "Airport activity", "Wider context",
     "Baselines & models", "Data table"])

with tab_models:
    C.render_model_report()

# ---------------------------------------------------------------- delays
with tab_delay:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("How late were departures?")
        clipped = scope["delay_minutes"].clip(upper=300)
        fig = px.histogram(clipped, nbins=26, labels={"value": "Minutes after schedule"},
                           color_discrete_sequence=[C.SKY])
        fig.add_vline(x=30, line_dash="dash", line_color=C.INK, annotation_text="30 min")
        fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Departures")
        st.plotly_chart(fig, width="stretch")
        st.caption("Delays above 300 minutes are shown in the last bar. "
                   "Times are the IAA board time, which is close to takeoff.")
    with c2:
        st.subheader("Delayed 30+ by departure time")
        band = scope.assign(band=(scope["hour"] // 3) * 3)
        stats = analytics.group_delay_stats(band, "band").sort_values("band")
        stats["label"] = stats["band"].map(lambda b: f"{b:02d}–{b + 3:02d}")
        fig = px.bar(stats, x="label", y="share_delayed_30", text=stats["n"].map(lambda n: f"n={n}"),
                     labels={"label": "Scheduled departure (Israel time)", "share_delayed_30": "Delayed 30+"},
                     color_discrete_sequence=[C.SKY])
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                          yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    st.subheader("Delayed 30+ by weekday")
    wd = analytics.group_delay_stats(scope, "weekday")
    wd["weekday"] = pd.Categorical(wd["weekday"], WEEKDAYS, ordered=True)
    wd = wd.sort_values("weekday")
    fig = px.bar(wd, x="weekday", y="share_delayed_30", text=wd["n"].map(lambda n: f"n={n}"),
                 labels={"weekday": "", "share_delayed_30": "Delayed 30+"}, color_discrete_sequence=[C.SKY])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.caption("Each weekday is covered by only one or two calendar days in the archive, so weekday "
               "differences may reflect those particular days rather than a weekly pattern.")

# ---------------------------------------------------------------- groups
with tab_groups:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Airlines")
        air = analytics.group_delay_stats(scope, ["airline_code", "airline_name"], min_n=min_n)
        if air.empty:
            st.info(f"No airline has at least {min_n} flights with these filters.")
        else:
            air["airline"] = air["airline_code"] + " · " + air["airline_name"].str.title()
            fig = px.bar(air.sort_values("share_delayed_30"), x="share_delayed_30", y="airline",
                         orientation="h", text=air.sort_values("share_delayed_30")["n"].map(lambda n: f"n={n}"),
                         labels={"share_delayed_30": "Delayed 30+", "airline": ""},
                         color_discrete_sequence=[C.SKY])
            fig.update_layout(height=max(260, 34 * len(air)), margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_tickformat=".0%", xaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Destinations")
        dst = analytics.group_delay_stats(scope, ["airport_code", "city", "country"], min_n=min_n)
        if dst.empty:
            st.info(f"No destination has at least {min_n} flights with these filters.")
        else:
            dst["city"] = dst["city"].str.title()
            dst["country"] = dst["country"].str.title()
            st.dataframe(dst, hide_index=True, width="stretch", height=420, column_config={
                "airport_code": "Airport", "city": "City", "country": "Country", "n": "n",
                "share_delayed_30": st.column_config.NumberColumn("Delayed 30+", format="percent"),
                "median_delay": st.column_config.NumberColumn("Median delay (min)", format="%.0f"),
            })
    st.caption(f"Groups with fewer than {min_n} flights are hidden (change this in the sidebar).")

# ---------------------------------------------------------------- activity
with tab_activity:
    st.subheader("Scheduled international flights per hour")
    days = analytics.complete_schedule_days(flights)
    dep_p = analytics.hourly_profile(flights, "D")
    arr_p = analytics.hourly_profile(flights, "A")
    fig = go.Figure([
        go.Bar(name="Departures", x=list(range(24)), y=dep_p.values, marker_color=C.SKY),
        go.Bar(name="Arrivals", x=list(range(24)), y=arr_p.values, marker_color=C.SKY_SOFT),
    ])
    fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Hour of day (Israel time)", yaxis_title="Average per hour",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Average over {len(days)} days with a complete schedule: "
               + ", ".join(f"{d:%b %d}" for d in days)
               + ". Airport-wide schedule; the sidebar filters do not apply here.")

    st.subheader("Departures by weekday and hour")
    sched = analytics.international(flights)
    sched = sched[(sched["direction"] == "D") & sched["service_date"].isin(days)]
    per_day = sched.groupby(["service_date", "weekday", "hour"]).size().reset_index(name="n")
    heat = per_day.groupby(["weekday", "hour"])["n"].mean().unstack(fill_value=0)
    heat = heat.reindex(index=[w for w in WEEKDAYS if w in heat.index], columns=range(24), fill_value=0)
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(x="Hour", y="", color="Departures"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each weekday here is one or two specific dates, so treat this as an illustration of "
               "the daily rhythm rather than a stable weekly pattern.")

# ---------------------------------------------------------------- context
with tab_context:
    st.subheader("Airline reliability, larger sample (TLVFlights)")
    if tlv is None:
        st.info("TLVFlights data not found. Run the download and processing scripts.")
    else:
        top = tlv.head(15)
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "airline_code": "Code", "airline_name": "Airline",
            "total_flights": st.column_config.NumberColumn("Flights", format="localized"),
            "on_time_rate": st.column_config.NumberColumn("On time (≤15 min)", format="percent"),
            "late_30_rate": st.column_config.NumberColumn("30+ min late", format="percent"),
            "cancel_rate": st.column_config.NumberColumn("Cancelled", format="percent"),
            "avg_delay_minutes": st.column_config.NumberColumn("Avg delay (min)", format="%.0f"),
        })
        st.caption(f"{int(tlv['total_flights'].sum()):,} flights, Dec 2025 – Nov 2026, airport closure "
                   "Feb 11 – May 15 excluded. Source: tlvflights.com. \"30+ min late\" is our calculation "
                   "from their 31+ minute buckets. They estimate gate time as takeoff minus 15 min, so "
                   "rates differ from the archive.")
    st.subheader("Price of flying abroad, monthly")
    if price is None:
        st.info("Price index not found. Run the download and processing scripts.")
    else:
        fig = px.line(price, x="month", y="index_value", labels={"month": "", "index_value": "Index"},
                      color_discrete_sequence=[C.SKY])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Israel CBS series 121390 \"Expenditures for travel abroad and local flights\", chained "
                   f"to one base ({price['base'].iloc[0]} = 100). A national index without routes or "
                   "airlines: price context, not a fare forecast.")

# ---------------------------------------------------------------- data
with tab_data:
    cols = ["scheduled", "updated", "airline_code", "airline_name", "flight_number", "airport_code",
            "city", "country", "terminal", "delay_minutes", "delayed_30", "codeshares"]
    st.dataframe(scope[cols], hide_index=True, width="stretch", height=420)
    st.download_button("Download filtered rows (CSV)", scope[cols].to_csv(index=False).encode("utf-8"),
                       file_name="flycheck_departures_filtered.csv", mime="text/csv")

C.data_note()
