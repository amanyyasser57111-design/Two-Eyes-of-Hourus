# ---------------------------------------------------------------
# Lunar Landing Site Navigator
# Compare two Moon landing sites: when are the Sun (power) and
# Earth (communication) above the horizon? Run with:
#     streamlit run app.py
# ---------------------------------------------------------------

import numpy as np                      # numbers and arrays
import pandas as pd                     # tables
import streamlit as st                  # the web app
import plotly.graph_objects as go       # charts
from datetime import date               # for the date picker
from skyfield.api import Loader, PlanetaryConstants  # NASA/JPL astronomy math
from moon3d import make_moon_figure                  # the 3D animated Moon (moon3d.py)

st.set_page_config(page_title="Lunar Landing Site Navigator", layout="wide")

STEP_HOURS = 2   # the app calculates one point every 2 hours

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
CUSTOM = "Custom coordinates"

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
def compute(lat, lon, start, days):
    """Table with Sun and Earth elevation + azimuth every STEP_HOURS hours.
    Elevation = degrees above the horizon (above 0 = visible)."""
    ts, eph, pc, frame = load_data()
    n = int(days * 24 / STEP_HOURS)
    t0 = ts.utc(start.year, start.month, start.day)
    times = ts.tt_jd(t0.tt + np.arange(n) * STEP_HOURS / 24.0)

    # A point on the Moon's surface that rotates with the Moon
    spot = eph["moon"] + pc.build_latlon_degrees(frame, lat, lon)
    here = spot.at(times)
    sun_alt, sun_az, _ = here.observe(eph["sun"]).apparent().altaz()
    earth_alt, earth_az, _ = here.observe(eph["earth"]).apparent().altaz()

    return pd.DataFrame({
        "time": times.utc_datetime(),
        "Sun elevation": sun_alt.degrees,
        "Earth elevation": earth_alt.degrees,
        "Sun azimuth": sun_az.degrees,
        "Earth azimuth": earth_az.degrees,
    })

def status_text(elevations):
    """Plain-English status: when does it next rise or set?"""
    up = elevations > 0
    if up.all():
        return "Always above the horizon in this window"
    if not up.any():
        return "Never rises in this window"
    flip = np.where(up != up[0])[0][0] * STEP_HOURS   # hours until it changes
    return f"Sets in **{flip} h**" if up[0] else f"Rises in **{flip} h**"

def longest_run(mask):
    """Longest stretch (in hours) where `mask` is True, e.g. the Sun is down."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best * STEP_HOURS

def fmt_hours(hours, window_hours):
    if hours == 0:
        return "none"
    if hours >= window_hours:
        return "whole window"
    return f"{hours / 24:.1f} days" if hours >= 48 else f"{hours:.0f} h"

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

# --- 4. SKY VIEW (looking up from the landing site) ---------------
def sky_plot(df, idx):
    """Circle = the whole sky. Center = straight up, edge = horizon.
    Dots = where the Sun/Earth are during the window. Star = the chosen moment."""
    fig = go.Figure()
    row = df.iloc[idx]
    for body, color in (("Sun", "#FFB020"), ("Earth", "#4DA3FF")):
        up = df[df[f"{body} elevation"] > 0]
        fig.add_trace(go.Scatterpolar(
            theta=up[f"{body} azimuth"], r=90 - up[f"{body} elevation"], mode="markers",
            marker=dict(size=4, color=color, opacity=0.45), name=f"{body} path"))
        if row[f"{body} elevation"] > 0:
            fig.add_trace(go.Scatterpolar(
                theta=[row[f"{body} azimuth"]], r=[90 - row[f"{body} elevation"]], mode="markers",
                marker=dict(size=16, color=color, symbol="star", line=dict(color="white", width=1)),
                name=f"{body} at chosen moment"))
    fig.update_layout(
        template="plotly_dark", height=380, margin=dict(l=30, r=30, t=20, b=20),
        legend=dict(orientation="h", y=-0.1),
        polar=dict(
            radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                            ticktext=["90° (overhead)", "60°", "30°", "0° (horizon)"]),
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickvals=[0, 90, 180, 270], ticktext=["N", "E", "S", "W"])))
    return fig

# --- 5. ONE SITE PANEL (used twice, for side-by-side comparison) --
def site_panel(label, default_index, start, days, idx):
    st.subheader(label)
    options = list(SITES) + [CUSTOM]
    name = st.selectbox("Landing site", options, index=default_index, key=label)
    if name == CUSTOM:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude (°, south is negative)", -90.0, 90.0, -85.0, 0.1, key=label + "lat")
        lon = c2.number_input("Longitude (°, east is positive)", -180.0, 180.0, 0.0, 0.1, key=label + "lon")
        lat = max(min(lat, 89.99), -89.99)      # exactly ±90 confuses the math
        name = f"Custom ({lat:.2f}°, {lon:.2f}°)"
    else:
        lat, lon = SITES[name]
    st.caption(f"Latitude {lat}°, longitude {lon}°")

    df = compute(lat, lon, start, days)
    window_hours = len(df) * STEP_HOURS
    sun_up = (df["Sun elevation"] > 0).values
    earth_up = (df["Earth elevation"] > 0).values

    # Key numbers
    c1, c2, c3 = st.columns(3)
    c1.metric("Sunlit time", f"{sun_up.mean() * 100:.0f}%",
              help="Share of the time window when the Sun is above the horizon (solar power possible).")
    c2.metric("Earth visible", f"{earth_up.mean() * 100:.0f}%",
              help="Share of the time window when Earth is above the horizon (direct radio contact possible).")
    c3.metric("Max Sun elevation", f"{df['Sun elevation'].max():.1f}°",
              help="Highest the Sun gets. Higher = more power per solar panel. Near 0° = long shadows and weak power.")
    c4, c5 = st.columns(2)
    c4.metric("Longest darkness", fmt_hours(longest_run(~sun_up), window_hours),
              help="Longest unbroken stretch with no Sun. The lander must survive this on batteries and heaters.")
    c5.metric("Longest comms gap", fmt_hours(longest_run(~earth_up), window_hours),
              help="Longest unbroken stretch with no direct line to Earth.")

    st.write("☀️ Sun:", status_text(df["Sun elevation"].values))
    st.write("🌍 Earth:", status_text(df["Earth elevation"].values))

    # Timeline chart
    st.markdown("**Timeline:** above the dashed line = visible")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["Sun elevation"], name="Sun",
                             line=dict(color="#FFB020", width=3)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["Earth elevation"], name="Earth",
                             line=dict(color="#4DA3FF", width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")   # the horizon
    fig.add_vline(x=df["time"].iloc[idx], line_color="white", opacity=0.4)  # chosen moment
    fig.update_layout(template="plotly_dark", height=340,
                      yaxis_title="Elevation above horizon (°)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Sky view at the moment chosen in the sidebar
    r = df.iloc[idx]
    st.markdown(f"**Sky view on {r['time']:%b %d, %H:%M} UTC** (looking up from the site)")
    st.write(f"☀️ Sun {r['Sun elevation']:.1f}° · 🌍 Earth {r['Earth elevation']:.1f}° above the horizon"
             " (negative = hidden below it)")
    st.plotly_chart(sky_plot(df, idx), use_container_width=True)
    return name, lat, lon

# --- 6. THE PAGE --------------------------------------------------
st.title("🌙 Lunar Landing Site Navigator")
st.write("Compare when the Sun (power) and Earth (communication) are above the horizon at two Moon sites.")

with st.expander("New here? What am I looking at?"):
    st.markdown("""
- **Why it matters:** a lander needs sunlight for power and a line of sight to Earth to talk to mission control.
  The Moon's south pole is extreme: the Sun barely climbs above the horizon, so a few kilometres or a few days can change everything.
- **Elevation** is how high something is above the horizon: 0° is the horizon, 90° is straight overhead.
- **Timeline chart:** when the Sun and Earth are visible at each site. Compare Site A and Site B side by side.
- **Sky view:** the whole sky seen from the site. Dots show where the Sun and Earth travel; the star is the moment you pick in the sidebar.
- **Longest darkness / comms gap:** the longest stretch without Sun or without a direct line to Earth, which decides how long equipment must survive on its own.
- **3D Moon:** press Play to watch daylight sweep across the Moon.
""")

start = st.sidebar.date_input("Start date", value=date(2026, 12, 1),
                              min_value=date(2000, 1, 1), max_value=date(2049, 12, 1))
days = st.sidebar.slider("Days to show", 5, 60, 30)
st.sidebar.caption("One lunar day is about 29.5 Earth days.")
offset = st.sidebar.slider("Pick a moment in the window (days from start)", 0.0, float(days) - 0.5, 0.0, 0.5)
idx = min(int(offset * 24 / STEP_HOURS), int(days * 24 / STEP_HOURS) - 1)
smooth = st.sidebar.checkbox("Smooth 3D animation (slower)", value=False)

try:
    left, right = st.columns(2)
    with left:
        site_a = site_panel("Site A", 0, start, days, idx)
    with right:
        site_b = site_panel("Site B", 3, start, days, idx)
except Exception as e:
    st.error("Something went wrong while computing. Copy this message and send it to Claude:")
    st.code(repr(e))
    st.stop()

# --- 7. THE 3D MOON ------------------------------------------------
st.divider()
st.header("🌑 The Moon in motion")
st.caption("Press Play. Bright = sunlit, dark = night. The yellow dot is the Sun and the blue dot is Earth "
           "(not to scale). Drag to rotate, scroll to zoom. Your two sites are the marked diamonds.")
try:
    frames_n = 90 if smooth else 40
    sun_v, earth_v, labels = directions(start, days, frames_n)
    sites = [(f"A: {site_a[0]}", site_a[1], site_a[2], "#ff4d6d"),
             (f"B: {site_b[0]}", site_b[1], site_b[2], "#00e5ff")]
    st.plotly_chart(make_moon_figure(sun_v, earth_v, labels, sites,
                                     n_lat=50 if smooth else 30, n_lon=100 if smooth else 60),
                    use_container_width=True)
except Exception as e:
    st.error("The 3D Moon failed. Copy this message and send it to Claude:")
    st.code(repr(e))

# --- 8. HONEST LIMITS ----------------------------------------------
st.info("**Known limitations:** the horizon is assumed flat, so real mountains and crater walls "
        "(especially near the south pole) would shorten daylight and Earth visibility compared with these numbers. "
        "Next step: add terrain from NASA's LOLA laser-altimeter data. Positions come from NASA/JPL's DE421 data "
        "(including the Moon's libration, its slow wobble). Site coordinates are approximate.")