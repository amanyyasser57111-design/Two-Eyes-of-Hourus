
import streamlit as st
import pandas as pd

# pyright: reportMissingImports=false
try:
    import plotly.express as px
except ImportError:
    px = None
    import altair as alt

st.set_page_config(
    page_title="HORUS | Lunar Navigator",
    page_icon="🌙",
    layout="wide"
)

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.stApp {
    background-color: #07111f;
    color: #ffffff;
}
h1, h2, h3 {
    color: #f5c76b !important;
}
[data-testid="stMetric"] {
    background: #14243a;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #34465e;
}
section[data-testid="stSidebar"] {
    background-color: #101e31;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("🌙 HORUS")
st.subheader("Lunar Landing Site Navigator")

st.caption(
    "Explore lunar south pole regions "
    "and evaluate potential landing locations."
)

st.divider()

# -------------------------
# LUNAR SITES
# -------------------------
sites = pd.DataFrame({
    "Site": [
        "Malapert Massif",
        "Shackleton Rim",
        "de Gerlache Rim 2",
        "Connecting Ridge"
    ],
    "Latitude": [-85.98, -89.92, -88.20, -89.05],
    "Longitude": [-0.42, -143.97, -64.42, -136.07]
})

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🚀 Mission Control")

selected_site = st.sidebar.selectbox(
    "Select Landing Site",
    sites["Site"]
)

st.sidebar.info(
    "HORUS evaluates sunlight, "
    "Earth visibility and terrain conditions."
)

# -------------------------
# SELECTED SITE
# -------------------------
site = sites[sites["Site"] == selected_site].iloc[0]

st.header("Mission Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Site", selected_site)

with col2:
    st.metric("Latitude", f"{site['Latitude']}°")

with col3:
    st.metric("Longitude", f"{site['Longitude']}°")

# -------------------------
# LUNAR COORDINATE MAP
# -------------------------
st.header("🗺️ Lunar South Pole Sites")

fig = px.scatter(
    sites,
    x="Longitude",
    y="Latitude",
    text="Site",
    hover_data=["Latitude", "Longitude"],
    title="Candidate Landing Site Coordinates"
)

fig.update_traces(
    marker=dict(size=14, color="#f5c76b"),
    textposition="top center"
)

fig.update_layout(
    paper_bgcolor="#07111f",
    plot_bgcolor="#101e31",
    font_color="white",
    xaxis_title="Longitude (degrees)",
    yaxis_title="Latitude (degrees)"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "This is a coordinate plot, not a lunar imagery map. "
    "Coordinates are approximate candidate-region references."
)

# -------------------------
# TRACKING
# -------------------------
st.header("☀️ Sun & 🌍 Earth Tracking")

col1, col2 = st.columns(2)

with col1:
    st.info("Sun tracking: Scientific data integration pending.")

with col2:
    st.info("Earth tracking: Scientific data integration pending.")

# -------------------------
# FOOTER
# -------------------------
st.divider()

st.caption(
    "HORUS | NASA Space Apps Challenge Project"
)
