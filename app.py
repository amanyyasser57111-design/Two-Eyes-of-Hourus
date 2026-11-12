import streamlit as st

st.set_page_config(
    page_title="HORUS",
    page_icon="🌙",
    layout="wide"
)

st.title("🌙 HORUS")
st.subheader("Lunar Landing Site Navigator")

st.markdown("""
Welcome to HORUS!

Explore the Moon and discover
the best locations for future missions.
""")

st.sidebar.title("Mission Control")

site = st.sidebar.selectbox(
    "Select Landing Site",
    [
        "Shackleton Crater",
        "Malapert Massif",
        "de Gerlache Crater"
    ]
)

st.header("Mission Overview")

st.write("Selected location:", site)

col1, col2 = st.columns(2)

with col1:
    st.metric("Sun Tracking", "Coming Soon")

with col2:
    st.metric("Earth Tracking", "Coming Soon")

st.success("HORUS Mission Dashboard is running!")