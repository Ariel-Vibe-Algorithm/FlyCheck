"""Compare flights: user-entered options side by side, sortable, no 'best flight' verdict."""
from datetime import time

import pandas as pd
import streamlit as st

from app_pages import common as C
from src import analytics

C.require_data()
flights = C.load_flights()
scope = analytics.labelled(flights)
tlv = C.load_tlvflights()
profile = C.load_profile()
tlv_rates = {} if tlv is None else tlv.set_index("airline_code")["late_30_rate"].to_dict()
tlv_cancel = {} if tlv is None else tlv.set_index("airline_code")["cancel_rate"].to_dict()

st.title("Compare flights")
st.write("Add the options you are choosing between, then sort the results by what matters to "
         "you. FlyCheck does not pick a single best flight.")

airlines = C.airline_options(scope, tlv)
dests = C.destination_options(scope)

if "compare_rows" not in st.session_state:
    st.session_state.compare_rows = pd.DataFrame({
        "airline": ["LY", "W6", "IZ"],
        "destination": ["FCO", "FCO", "FCO"],
        "departure": [time(6, 15), time(11, 40), time(19, 5)],
        "price_ils": [None, None, None],
    })

st.caption("The three starting rows are examples. Edit them, delete them or add your own rows.")
edited = st.data_editor(
    st.session_state.compare_rows, num_rows="dynamic", width="stretch", key="compare_editor",
    column_config={
        "airline": st.column_config.SelectboxColumn("Airline", options=list(airlines), required=True),
        "destination": st.column_config.SelectboxColumn("Destination", options=list(dests), required=True),
        "departure": st.column_config.TimeColumn("Departure (Israel)", format="HH:mm", required=True),
        "price_ils": st.column_config.NumberColumn("Your price (₪, optional)", min_value=0, step=1),
    },
)
lead = st.radio("Arrive before departure", [3.0, 3.5, 4.0], horizontal=True,
                format_func=lambda h: f"{h:g} h", key="cmp_lead")

rows = []
for r in edited.dropna(subset=["airline", "destination", "departure"]).itertuples():
    ref = analytics.delay_reference(scope, airline=r.airline, airport=r.destination, hour=r.departure.hour)
    cong = analytics.congestion_at(r.departure, lead, profile)
    rows.append({
        "Flight": f"{r.airline} → {r.destination}",
        "Departs": r.departure.strftime("%H:%M"),
        "Airline 30+ late (archive)": ref["airline"]["share_delayed_30"],
        "n (airline)": ref["airline"]["n"],
        "Airline 30+ late (TLVFlights)": tlv_rates.get(r.airline),
        "Hour band 30+ late": ref["hour_band"]["share_delayed_30"],
        "Airline cancelled (TLVFlights)": tlv_cancel.get(r.airline),
        "Hall activity at arrival": cong["score"],
        "Level": cong["level"],
        "Price ₪": r.price_ils,
    })

if not rows:
    st.info("Add at least one option above to compare.")
    st.stop()

pct = st.column_config.NumberColumn(format="percent")
st.dataframe(
    pd.DataFrame(rows), hide_index=True, width="stretch",
    column_config={
        "Airline 30+ late (archive)": pct,
        "Airline 30+ late (TLVFlights)": pct,
        "Hour band 30+ late": pct,
        "Airline cancelled (TLVFlights)": st.column_config.NumberColumn(format="percent"),
        "Hall activity at arrival": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Price ₪": st.column_config.NumberColumn(format="₪%d"),
    },
)
st.caption("Click a column header to sort. Delay columns are historical rates, not predictions; "
           "the ML probability is added in M3. Prices are what you type. Conversion from other "
           "currencies with Bank of Israel rates is planned.")
