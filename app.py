# ---------------------------------------------------------------
# Lunar Landing Site Navigator - STARTER VERSION (Day 1)
# Shows how high the Sun and Earth are above the horizon at two
# Moon landing sites, so you can compare them side by side.
# Run it with:  streamlit run app.py
# ---------------------------------------------------------------

import numpy as np                      # numbers and arrays
import pandas as pd                     # tables
import streamlit as st                  # the web app
import plotly.graph_objects as go       # charts
from datetime import date               # for the date picker
from skyfield.api import Loader, PlanetaryConstants  # NASA/JPL astronomy math
from moon3d import make_moon_figure                  # the 3D animated Moon (moon3d.py)

st.set_page_config(page_title="Lunar Landing Site Navigator", layout="wide")

# --- 1. THE SITES -------------------------------------------------
# (latitude, longitude) in degrees. Positive latitude = north,
# positive longitude = east. These are APPROXIMATE: double-check
# the exact numbers before your final submission.
SITES = {
    "Shackleton crater rim (south pole)": (-89.9, 0.0),
    "Malapert massif (near south pole)": (-85.9, 0.0),
    "Chandrayaan-3 landing site": (-69.37, 32.32),
    "Apollo 11": (0.67, 23.47),
    "Apollo 15": (26.13, 3.63),
    "Apollo 17": (20.19, 30.77),
}

# --- 2. LOAD NASA DATA (downloads once, then is remembered) -------
@st.cache_resource
def load_data():
    loader = Loader("skyfield_data")        # data files are saved in this folder
    ts = loader.timescale()
    eph = loader("de421.bsp")               # positions of Sun, Earth, Moon
    pc = PlanetaryConstants()               # extra files that describe the Moon's rotation/wobble
    pc.read_text(loader("moon_080317.tf"))
    pc.read_text(loader("pck00008.tpc"))
    pc.read_binary(loader("moon_pa_de421_1900-2050.bpc"))
    frame = pc.build_frame_named("MOON_ME_DE421")
    return ts, eph, pc, frame

# --- 3. THE CORE MATH ---------------------------------------------
@st.cache_data
def compute(lat, lon, start, days, step_hours=2):
    """Return a table with Sun and Earth elevation (degrees above the
    horizon) every `step_hours` hours. Above 0 = visible."""
    ts, eph, pc, frame = load_data()
    n = int(days * 24 / step_hours)
    t0 = ts.utc(start.year, start.month, start.day)
    times = ts.tt_jd(t0.tt + np.arange(n) * step_hours / 24.0)

    # A point on the Moon's surface that rotates with the Moon
    spot = eph["moon"] + pc.build_latlon_degrees(frame, lat, lon)

    sun_alt = spot.at(times).observe(eph["sun"]).apparent().altaz()[0].degrees
    earth_alt = spot.at(times).observe(eph["earth"]).apparent().altaz()[0].degrees

    return pd.DataFrame({
        "time": times.utc_datetime(),
        "Sun elevation": sun_alt,
        "Earth elevation": earth_alt,
    })

def status_text(elevations, step_hours=2):
    """Plain-English status: when does it next rise or set?"""
    up = elevations > 0
    if up.all():
        return "Always above the horizon in this window"
    if not up.any():
        return "Never rises in this window"
    flip = np.where(up != up[0])[0][0] * step_hours   # hours until it changes
    return f"Sets in **{flip} h**" if up[0] else f"Rises in **{flip} h**"

# --- 3b. SUN AND EARTH DIRECTIONS (for the 3D Moon) ---------------
@st.cache_data
def directions(start, days, n_frames=60):
    """Direction of the Sun and Earth in the Moon's own (rotating) frame,
    at `n_frames` moments between `start` and `start + days`."""
    ts, eph, pc, frame = load_data()
    t0 = ts.utc(start.year, start.month, start.day)
    times = ts.tt_jd(t0.tt + np.linspace(0, days, n_frames, endpoint=False))
    from_moon = eph["moon"].at(times)
    out = {}
    for name in ("sun", "earth"):
        pos = from_moon.observe(eph[name]).apparent()
        try:
            x, y, z = pos.frame_xyz(frame).au
            v = np.column_stack([x, y, z])
        except AttributeError:                       # older Skyfield: do the rotation by hand
            rot = frame.rotation_at(times)           # shape (3, 3, N)
            v = np.einsum("ijn,jn->ni", rot, pos.position.au)
        out[name] = v / np.linalg.norm(v, axis=1, keepdims=True)
    labels = [d.strftime("%b %d") for d in times.utc_datetime()]
    return out["sun"], out["earth"], labels

# --- 4. ONE SITE PANEL (used twice, for side-by-side comparison) --
def site_panel(label, default_index, start, days):
    st.subheader(label)
    name = st.selectbox("Landing site", list(SITES), index=default_index, key=label)
    lat, lon = SITES[name]
    st.caption(f"Latitude {lat}°, longitude {lon}°")

    df = compute(lat, lon, start, days)

    # Key numbers
    sun_pct = (df["Sun elevation"] > 0).mean() * 100
    earth_pct = (df["Earth elevation"] > 0).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Sunlit time", f"{sun_pct:.0f}%")
    c2.metric("Earth visible", f"{earth_pct:.0f}%")
    c3.metric("Max Sun elevation", f"{df['Sun elevation'].max():.1f}°")

    st.write("☀️ Sun:", status_text(df["Sun elevation"].values))
    st.write("🌍 Earth:", status_text(df["Earth elevation"].values))

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["Sun elevation"], name="Sun",
                             line=dict(color="#FFB020", width=3)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["Earth elevation"], name="Earth",
                             line=dict(color="#4DA3FF", width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")   # the horizon
    fig.update_layout(template="plotly_dark", height=380,
                      yaxis_title="Elevation above horizon (°)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    return name, lat, lon

# --- 5. THE PAGE --------------------------------------------------
st.title("🌙 Lunar Landing Site Navigator")
st.write("Compare when the Sun (power) and Earth (communication) are above the horizon at two Moon sites.")

start = st.sidebar.date_input("Start date", value=date(2026, 12, 1))
days = st.sidebar.slider("Days to show", 5, 60, 30)
st.sidebar.caption("One lunar day is about 29.5 Earth days.")

try:
    left, right = st.columns(2)
    with left:
        site_a = site_panel("Site A", 0, start, days)
    with right:
        site_b = site_panel("Site B", 3, start, days)
except Exception as e:
    st.error("Something went wrong while computing. Copy this message and send it to Claude:")
    st.code(repr(e))
    st.stop()

# --- 6. THE 3D MOON ------------------------------------------------
st.divider()
st.header("🌑 The Moon in motion")
st.caption("Press Play. Bright = sunlit, dark = night. The yellow dot is the Sun and the blue dot is Earth "
           "(not to scale). Drag to rotate, scroll to zoom. Your two sites are the marked diamonds.")
try:
    sun_v, earth_v, labels = directions(start, days)
    sites = [(f"A: {site_a[0]}", site_a[1], site_a[2], "#ff4d6d"),
             (f"B: {site_b[0]}", site_b[1], site_b[2], "#00e5ff")]
    st.plotly_chart(make_moon_figure(sun_v, earth_v, labels, sites), use_container_width=True)
except Exception as e:
    st.error("The 3D Moon failed. Copy this message and send it to Claude:")
    st.code(repr(e))