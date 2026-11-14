# ---------------------------------------------------------------
# Lunar Landing Site Navigator
# Compare two Moon landing sites: when are the Sun (power) and
# Earth (communication) above the local terrain horizon? Run with:
#     streamlit run app.py
# ---------------------------------------------------------------

import numpy as np                      # numbers and arrays
import pandas as pd                     # tables
import streamlit as st                  # the web app
import plotly.graph_objects as go       # charts
from datetime import date               # for the date picker
from skyfield.api import Loader, PlanetaryConstants  # NASA/JPL SPICE ephemeris math
from moon3d import make_moon_figure                  # the 3D animated Moon (moon3d.py)

st.set_page_config(page_title="Lunar Landing Site Navigator", layout="wide")

STEP_HOURS = 2   # the app calculates one point every 2 hours

# --- 1. THE SITES (Curated from NASA Artemis/CLPS, LROC QuickMap & CSA) ---
# Format: "Site Name": (latitude_deg, longitude_deg, default_terrain_horizon_deg, mission_context)
# IMPORTANT: the terrain horizon numbers below are PLACEHOLDER ESTIMATES, not measurements.
# Replace them with values you read from Quickmap / Moon Trek. South pole sites have rim obstructions,
# whereas equatorial maria (Apollo) have near-flat horizons (~0.0°).
SITES = {
    "Shackleton Crater Rim (South Pole)": (
        -89.90, 0.00, 1.5,
        "Peak of eternal light candidate; extreme low sun angles (<2°) and steep crater rim shadows."
    ),
    "Artemis III: Connecting Ridge": (
        -89.45, -137.37, 1.8,
        "High-priority Artemis III candidate ridge connecting Shackleton and de Gerlache craters."
    ),
    "Artemis III: Nobile Rim 1": (
        -85.20, 55.00, 2.0,
        "NASA/CSA collaborative science target rich in permanently shadowed regions (PSRs) nearby."
    ),
    "Mons Mouton (VIPER / CLPS Target)": (
        -84.90, 31.00, 1.2,
        "Flat-topped plateau near the South Pole selected for rover mobility and solar access."
    ),
    "Malapert Massif (South Pole)": (
        -85.90, 0.00, 2.2,
        "High-altitude peak offering strong direct-to-Earth visibility lines."
    ),
    "Chandrayaan-3 (Shiv Shakti Point)": (
        -69.37, 32.32, 0.5,
        "High-latitude southern highlands landing site."
    ),
    "Apollo 11 (Mare Tranquillitatis)": (
        0.67, 23.47, 0.0,
        "Equatorial mare baseline: 14.7 Earth days of continuous high-sunlight, flat horizon."
    ),
    "Apollo 15 (Hadley-Apennine)": (
        26.13, 3.63, 1.0,
        "Mid-latitude site bordered by the Mons Hadley massif (~1° local mountain mask)."
    ),
}
CUSTOM = "Custom coordinates"

# --- 2. LOAD NASA/JPL SPICE DATA (Downloads once, cached locally) ---
@st.cache_resource
def load_data():
    """Loads official NASA/JPL DE421 planetary ephemerides and lunar libration kernels
    (from the same family of JPL ephemerides that the Horizons System serves)."""
    loader = Loader("skyfield_data")
    ts = loader.timescale()
    eph = loader("de421.bsp")               # NASA/JPL DE421 Sun, Earth, Moon positions
    pc = PlanetaryConstants()               # NASA NAIF SPICE kernels for Moon rotation & libration
    pc.read_text(loader("moon_080317.tf"))
    pc.read_text(loader("pck00008.tpc"))
    pc.read_binary(loader("moon_pa_de421_1900-2050.bpc"))
    frame = pc.build_frame_named("MOON_ME_DE421")
    return ts, eph, pc, frame

# --- 3. THE CORE MATH (JPL Horizons / DE421 Ephemeris Engine) -------
@st.cache_data
def compute(lat, lon, start, days):
    """Computes Sun and Earth elevation + azimuth every STEP_HOURS hours using JPL DE421."""
    ts, eph, pc, frame = load_data()
    n = int(days * 24 / STEP_HOURS)
    t0 = ts.utc(start.year, start.month, start.day)
    times = ts.tt_jd(t0.tt + np.arange(n) * STEP_HOURS / 24.0)

    # A point on the Moon's surface rotating with lunar libration
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

def status_text(elevations, horizon_deg):
    """Plain-English status relative to the local terrain horizon mask."""
    up = elevations > horizon_deg
    if up.all():
        return f"Always above {horizon_deg:.1f}° horizon in this window"
    if not up.any():
        return f"Never rises above {horizon_deg:.1f}° horizon in this window"
    flip = np.where(up != up[0])[0][0] * STEP_HOURS
    return f"Sets below terrain in **{flip} h**" if up[0] else f"Rises above terrain in **{flip} h**"

def longest_run(mask):
    """Longest stretch (in hours) where `mask` is True."""
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
        except AttributeError:
            rot = frame.rotation_at(times)
            v = np.einsum("ijn,jn->ni", rot, pos.position.au)
        out[name] = v / np.linalg.norm(v, axis=1, keepdims=True)
    labels = [d.strftime("%b %d") for d in times.utc_datetime()]
    return out["sun"], out["earth"], labels

# --- 4. SKY VIEW (looking up from the landing site) ---------------
def sky_plot(df, idx, horizon_deg):
    """Polar sky-dome plot showing Sun/Earth paths and the shaded local terrain mask."""
    fig = go.Figure()
    row = df.iloc[idx]

    # Shaded terrain horizon mask ring (if terrain > 0°)
    if horizon_deg > 0:
        theta_ring = np.linspace(0, 360, 180)
        fig.add_trace(go.Scatterpolar(
            theta=theta_ring,
            r=[90 - horizon_deg] * len(theta_ring),
            mode="lines",
            line=dict(color="#ff6b6b", width=1.5, dash="dot"),
            name=f"Terrain mask ({horizon_deg:.1f}°)"
        ))

    for body, color in (("Sun", "#FFB020"), ("Earth", "#4DA3FF")):
        up = df[df[f"{body} elevation"] > horizon_deg]
        fig.add_trace(go.Scatterpolar(
            theta=up[f"{body} azimuth"], r=90 - up[f"{body} elevation"], mode="markers",
            marker=dict(size=4, color=color, opacity=0.5), name=f"{body} visible path"))
        if row[f"{body} elevation"] > horizon_deg:
            fig.add_trace(go.Scatterpolar(
                theta=[row[f"{body} azimuth"]], r=[90 - row[f"{body} elevation"]], mode="markers",
                marker=dict(size=16, color=color, symbol="star", line=dict(color="white", width=1)),
                name=f"{body} at chosen moment"))

    fig.update_layout(
        template="plotly_dark", height=380, margin=dict(l=30, r=30, t=20, b=20),
        legend=dict(orientation="h", y=-0.15),
        polar=dict(
            radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                            ticktext=["90° (zenith)", "60°", "30°", "0° (flat)"]),
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
        lat = max(min(lat, 89.99), -89.99)
        default_horizon = 1.5
        context = "Custom coordinates — verify surrounding crater wall elevations in LROC QuickMap."
        display_name = f"Custom ({lat:.2f}°, {lon:.2f}°)"
    else:
        lat, lon, default_horizon, context = SITES[name]
        display_name = name

    st.caption(f"📍 **Coordinates:** {lat}°, {lon}° | {context}")

    # Direct deep-links to NASA QuickMap & Moon Trek centered on the exact site coordinates
    quickmap_url = f"https://quickmap.lroc.asu.edu/?extent={lon-1},{lat-1},{lon+1},{lat+1}&proj=22"
    moontrek_url = f"https://trek.nasa.gov/moon/#v=0.1&x={lon}&y={lat}&z=7"
    st.caption("If a link does not land on the right spot, open the site from the sources list below and search these coordinates.")
    st.markdown(f"🔍 Inspect terrain & elevation profile: [Open in NASA LROC QuickMap ↗]({quickmap_url}) · [Open in NASA Moon Trek ↗]({moontrek_url})")

    # Terrain Horizon Mask Slider (simulates LOLA/QuickMap crater rim obstruction)
    horizon_deg = st.slider(
        "Terrain horizon mask (° above flat; assumed value)",
        min_value=0.0, max_value=10.0, value=float(default_horizon), step=0.1,
        key=label + "horizon",
        help="PLACEHOLDER default, not a measurement. Replace it with the rim angle you read from Quickmap / Moon Trek (LOLA terrain). "
             "At the South Pole, a 2° rim can block low-angle sunlight for days."
    )

    df = compute(lat, lon, start, days)
    window_hours = len(df) * STEP_HOURS
    sun_up = (df["Sun elevation"] > horizon_deg).values
    earth_up = (df["Earth elevation"] > horizon_deg).values

    # Key operational metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Sunlit time", f"{sun_up.mean() * 100:.0f}%",
              help=f"Share of window where Sun clears the {horizon_deg:.1f}° terrain mask.")
    c2.metric("Earth visible", f"{earth_up.mean() * 100:.0f}%",
              help=f"Share of window where Earth clears the {horizon_deg:.1f}° terrain mask.")
    c3.metric("Max Sun elevation", f"{df['Sun elevation'].max():.1f}°",
              help="Peak solar angle. Near 0°–3° at the South Pole requires vertical solar arrays.")

    c4, c5 = st.columns(2)
    c4.metric("Longest darkness", fmt_hours(longest_run(~sun_up), window_hours),
              help="Longest continuous eclipse behind local terrain (battery survival requirement).")
    c5.metric("Longest comms gap", fmt_hours(longest_run(~earth_up), window_hours),
              help="Longest continuous loss of direct-to-Earth line of sight.")

    st.write("☀️ **Sun:**", status_text(df["Sun elevation"].values, horizon_deg))
    st.write("🌍 **Earth:**", status_text(df["Earth elevation"].values, horizon_deg))

    # Timeline chart
    st.markdown(f"**Timeline:** above the red dashed line ({horizon_deg:.1f}° terrain mask) = unobstructed")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["Sun elevation"], name="Sun",
                             line=dict(color="#FFB020", width=3)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["Earth elevation"], name="Earth",
                             line=dict(color="#4DA3FF", width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="0° flat horizon")
    if horizon_deg > 0:
        fig.add_hline(y=horizon_deg, line_dash="dash", line_color="#ff6b6b",
                      annotation_text=f"{horizon_deg:.1f}° terrain")
    fig.add_vline(x=df["time"].iloc[idx], line_color="white", opacity=0.4)
    fig.update_layout(template="plotly_dark", height=340,
                      yaxis_title="Elevation above flat horizon (°)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Sky view at chosen moment
    r = df.iloc[idx]
    st.markdown(f"**Sky view on {r['time']:%b %d, %H:%M} UTC** (looking up from site)")
    st.write(f"☀️ Sun {r['Sun elevation']:.1f}° · 🌍 Earth {r['Earth elevation']:.1f}° (Terrain mask: {horizon_deg:.1f}°)")
    st.plotly_chart(sky_plot(df, idx, horizon_deg), use_container_width=True)

    # CSV Export for Mission Planners
    export_df = df.copy()
    export_df["Terrain_Mask_Deg"] = horizon_deg
    export_df["Sun_Unobstructed"] = sun_up
    export_df["Earth_Unobstructed"] = earth_up
    st.download_button(
        label=f"📥 Download {label} Ephemeris & Visibility CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{label.lower().replace(' ', '_')}_ephemeris.csv",
        mime="text/csv",
        key=label + "download"
    )

    return display_name, lat, lon

# --- 6. THE PAGE --------------------------------------------------
st.title("🌙 Lunar Landing Site Navigator")
st.write("Rapid trade-off analysis for **NASA Artemis / CLPS** and **international partner (CSA)** lunar missions: "
         "compare solar power and direct-to-Earth communication windows across candidate sites and launch dates.")

with st.expander("📖 Data sources and how each one is used"):
    st.markdown("""
- **[JPL Horizons](https://ssd.jpl.nasa.gov/horizons/)**: NASA/JPL's ephemeris service. We compute Sun and Earth directions from the Moon with JPL's DE421 ephemeris and lunar libration (wobble) data, the same family of data Horizons serves, and use Horizons as the reference to check our numbers.
- **[LROC Quickmap](https://quickmap.lroc.asu.edu/)** and **[NASA Moon Trek](https://trek.nasa.gov/moon/)**: used to inspect south pole terrain (LOLA elevation) and read how high crater rims rise above each site. That angle feeds the terrain horizon mask.
- **[CSA Moon Exploration](https://www.asc-csa.gc.ca/eng/astronomy/moon-exploration/)**: background on Canada's role in lunar missions and Artemis, which frames the mission-planning use case.
""")

start = st.sidebar.date_input("Mission Start Date", value=date(2026, 12, 1),
                              min_value=date(2000, 1, 1), max_value=date(2049, 12, 1))
days = st.sidebar.slider("Analysis Window (Earth days)", 5, 60, 30)
st.sidebar.caption("One lunar day (synodic month) is ~29.5 Earth days.")
offset = st.sidebar.slider("Inspect specific moment (days from start)", 0.0, float(days) - 0.5, 0.0, 0.5)
idx = min(int(offset * 24 / STEP_HOURS), int(days * 24 / STEP_HOURS) - 1)
smooth = st.sidebar.checkbox("Smooth 3D animation (slower)", value=False)

try:
    left, right = st.columns(2)
    with left:
        site_a = site_panel("Site A", 0, start, days, idx)
    with right:
        site_b = site_panel("Site B", 1, start, days, idx)
except Exception as e:
    st.error("Something went wrong while computing:")
    st.code(repr(e))
    st.stop()

# --- 7. THE 3D MOON ------------------------------------------------
st.divider()
st.header("🌑 The Moon in motion")
st.caption("Press Play to watch the solar terminator sweep across the lunar surface. Yellow dot = Sun, Blue dot = Earth "
           "(not to scale). Your two candidate landing sites are marked as diamonds.")
try:
    frames_n = 90 if smooth else 40
    sun_v, earth_v, labels = directions(start, days, frames_n)
    sites = [(f"A: {site_a[0]}", site_a[1], site_a[2], "#ff4d6d"),
             (f"B: {site_b[0]}", site_b[1], site_b[2], "#00e5ff")]
    st.plotly_chart(make_moon_figure(sun_v, earth_v, labels, sites,
                                     n_lat=50 if smooth else 30, n_lon=100 if smooth else 60),
                    use_container_width=True)
except Exception as e:
    st.error("The 3D Moon failed to render:")
    st.code(repr(e))

# --- 8. DATA SOURCES & METHODOLOGY ---------------------------------
st.info("**Method and limits:** Sun and Earth positions use NASA/JPL DE421 with lunar libration. "
        "The terrain mask is a single assumed angle in every direction. It is not a measured 360° horizon, "
        "so treat daylight and comms numbers as estimates until the angles are read from LOLA terrain data. "
        "Site coordinates are approximate.")