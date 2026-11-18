"""Airport guide: static, practical information with official links."""
import streamlit as st

IAA = "https://www.iaa.gov.il/en/airports/ben-gurion"

st.title("Airport guide")
st.write("Practical information for departing from Ben Gurion. Each item links to the official "
         "Israel Airports Authority (IAA) page. Details were retrieved on Nov 18, 2026 and can change.")

c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("**When to arrive**")
    st.markdown("### 3 hours")
    st.write("before international departures, per IAA. Domestic flights: 2 hours, and no later "
             "than 1 hour. FlyCheck never suggests arriving later than this.")
    st.markdown(f"[IAA security guidance]({IAA}/securitybgn/)")
with c2.container(border=True):
    st.markdown("**Terminals**")
    st.write("All international arrivals use Terminal 3. Terminal 1 reopened for some international "
             "departures on Jul 1, 2026, and some airlines use both terminals.")
    st.caption("Check your terminal on the live board or with your airline.")
with c3.container(border=True):
    st.markdown("**Parking**")
    st.markdown("| Lot | Price |\n|---|---|\n| Lot 15, open-air | ₪50/day |\n"
                "| Lot 15, covered | ₪90/day |\n| Short-term | ₪0.5/min |")
    st.markdown(f"[IAA parking page]({IAA}/parking-at-ben-gurion-airport/)")

c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("**Getting there**")
    st.write("Israel Railways stops at the airport station next to Terminal 3; buses serve "
             "Terminals 1 and 3.")
    st.markdown(f"[Train]({IAA}/directions/train/) · [Bus]({IAA}/directions/bus/)")
with c2.container(border=True):
    st.markdown("**Security & carry-on**")
    st.write("Banned items, lithium batteries and duty-free liquids are covered by the IAA "
             "guidance. Liquid limits depend on your airline.")
    st.markdown(f"[What you can carry]({IAA}/securitybgn/)")
with c3.container(border=True):
    st.markdown("**Lounges & accessibility**")
    st.write("Lounges in Terminals 1 and 3, accessibility services, family rooms and VAT refund.")
    st.markdown(f"[Lounges]({IAA}/lounges/) · [Accessibility]({IAA}/accessibility/)")

with st.container(border=True):
    st.markdown("**Planned extras**")
    st.markdown("- Weather at TLV and the destination for flights within 16 days (Open-Meteo, CC BY 4.0).\n"
                "- Distance and time difference to the destination (OurAirports).\n"
                "- Next trains to the airport for your date (Ministry of Transport GTFS).\n"
                "- Link to National Security Council travel warnings for the destination country.")
