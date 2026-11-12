"""
HORUS 2.1 — Lunar Landing Site Navigator
NASA Space Apps Challenge project

Scientific notes:
- Sun/Earth apparent azimuth and elevation are requested from JPL Horizons.
- Candidate coordinates are reference points, not certified landing ellipses.
- Illumination windows are geometric (elevation > 0°) and do not include terrain shadows.
- Solar energy values are idealized horizontal-surface proxies, not engineering power predictions.
- Communication visibility is geometric Earth-center elevation > 0°; no RF link budget,
  antenna pointing, occultation by local terrain, or relay-network analysis is performed.
- No landing safety or terrain score is fabricated. Terrain assessment requires validated
  LOLA/SLDEM elevation and hazard data for the exact region.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------
# Page configuration and visual system
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="HORUS 2.1 | Lunar Mission Control",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#080F1B"
PANEL = "#101B2B"
PANEL2 = "#142338"
GOLD = "#E8BC68"
GOLD2 = "#F5D99D"
TEXT = "#EDF2F8"
MUTED = "#9AAAC0"
BLUE = "#75B7E8"
GREEN = "#73C6A0"
RED = "#E58D8D"
PURPLE = "#B49AE6"
LINE = "rgba(164,184,211,.16)"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{ color-scheme: dark; }}
    html, body, [data-testid="stApp"], [data-testid="stMain"] {{
        direction:ltr !important; text-align:left !important;
    }}
    .stApp {{
        background:
          radial-gradient(ellipse at 80% -10%, rgba(56,93,132,.23), transparent 35%),
          radial-gradient(ellipse at 3% 35%, rgba(185,139,66,.07), transparent 28%), {BG};
        color:{TEXT}; font-family:'DM Sans',sans-serif;
    }}
    .block-container {{max-width:1540px;padding:1.7rem 2.6rem 3rem;}}
    h1,h2,h3,h4 {{color:{TEXT}!important;font-family:'Manrope',sans-serif!important;letter-spacing:-.025em;}}
    h1 {{font-size:clamp(2rem,3.8vw,3.2rem)!important;font-weight:800!important;}}
    h2 {{font-size:1.55rem!important;font-weight:750!important;}}
    h3 {{font-size:1.12rem!important;font-weight:700!important;}}
    p, [data-testid="stCaptionContainer"] {{color:{MUTED};}}
    .hero {{
      border:1px solid rgba(232,188,104,.28);border-radius:24px;padding:2.1rem 2.4rem;
      background:radial-gradient(circle at 88% 25%,rgba(232,188,104,.15),transparent 20%),
      linear-gradient(112deg,#172B43 0%,#101D2E 58%,#0D1929 100%);
      box-shadow:0 18px 55px rgba(0,0,0,.22);margin-bottom:1rem;
    }}
    .eyebrow,.kicker {{color:{GOLD};font-size:.68rem;font-weight:800;letter-spacing:.19em;text-transform:uppercase;}}
    .hero-title {{color:#fff;font:800 clamp(2.7rem,5vw,4.3rem) 'Manrope',sans-serif;letter-spacing:.08em;line-height:1.05;margin:.45rem 0;}}
    .hero-sub {{color:#D1DCE9;font-size:1rem;}}
    .chips {{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.25rem;}}
    .chip {{border:1px solid rgba(207,220,235,.2);background:rgba(5,13,23,.22);color:#D8E1EC;
      border-radius:99px;padding:.4rem .7rem;font-size:.65rem;font-weight:700;letter-spacing:.08em;}}
    .panel {{background:linear-gradient(145deg,rgba(24,42,64,.97),rgba(15,28,44,.97));
      border:1px solid {LINE};border-radius:16px;padding:1rem 1.15rem;min-height:112px;}}
    .metric-label {{color:{MUTED};font-size:.65rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;}}
    .metric-value {{color:{GOLD2};font:800 1.4rem 'Manrope',sans-serif;margin-top:.55rem;overflow-wrap:anywhere;}}
    .metric-note {{color:{MUTED};font-size:.72rem;margin-top:.2rem;}}
    .info-card {{background:rgba(20,35,56,.72);border:1px solid {LINE};border-radius:15px;padding:1rem 1.1rem;}}
    .info-title {{color:{GOLD2};font-weight:800;font-size:.9rem;margin-bottom:.45rem;}}
    .info-copy {{color:{MUTED};font-size:.8rem;line-height:1.65;}}
    .stButton>button,.stFormSubmitButton>button {{
      background:linear-gradient(135deg,{GOLD2},{GOLD})!important;color:#101827!important;
      border:0!important;border-radius:10px!important;font-weight:800!important;min-height:2.65rem;
    }}
    [data-testid="stForm"] {{background:rgba(18,32,50,.65);border:1px solid {LINE};border-radius:15px;padding:1rem;}}
    [data-testid="stMetric"] {{background:linear-gradient(145deg,#192D46,#112035);border:1px solid {LINE};border-radius:13px;padding:.85rem 1rem;}}
    [data-testid="stMetricLabel"] {{color:{MUTED}!important;}}
    [data-testid="stMetricValue"] {{color:{GOLD2}!important;font-family:'Manrope',sans-serif;font-weight:800!important;}}
    [data-testid="stWidgetLabel"] p {{color:{TEXT}!important;font-weight:600!important;}}
    [data-baseweb="select"]>div,[data-baseweb="input"]>div {{background:#172A41!important;color:{TEXT}!important;border-color:rgba(164,184,211,.22)!important;}}
    [data-testid="stSidebar"] {{background:linear-gradient(180deg,#111F31,#091321)!important;border-right:1px solid {LINE};}}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{color:{GOLD2}!important;}}
    hr {{border-color:{LINE}!important;}}
    .footer {{text-align:center;border-top:1px solid {LINE};margin-top:2rem;padding:1.4rem .5rem .3rem;}}
    .footer b {{color:{GOLD};letter-spacing:.28em;font-family:'Manrope',sans-serif;}}
    @media(max-width:900px){{.block-container{{padding:1rem .9rem 2rem}}.hero{{padding:1.5rem 1.25rem}}}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Reference candidate sites (coordinates remain approximate references)
# ---------------------------------------------------------------------
REFERENCE_SITES = {
    "Malapert Massif": {"lat": -85.98, "lon": -0.42, "kind": "South-polar candidate region", "source_note": "Approximate reference coordinate"},
    "Shackleton Rim": {"lat": -89.92, "lon": -143.97, "kind": "South-polar candidate region", "source_note": "Approximate reference coordinate"},
    "de Gerlache Rim 2": {"lat": -88.20, "lon": -64.42, "kind": "South-polar candidate region", "source_note": "Approximate reference coordinate"},
    "Connecting Ridge": {"lat": -89.05, "lon": -136.07, "kind": "South-polar candidate region", "source_note": "Approximate reference coordinate"},
    "Apollo 11 (Tranquility Base)": {"lat": 0.6741, "lon": 23.4729, "kind": "Historic landing site", "source_note": "Approximate historic landing coordinate"},
    "Apollo 12": {"lat": -3.0124, "lon": -23.4216, "kind": "Historic landing site", "source_note": "Approximate historic landing coordinate"},
    "Apollo 15": {"lat": 26.1322, "lon": 3.6339, "kind": "Historic landing site", "source_note": "Approximate historic landing coordinate"},
    "Apollo 17": {"lat": 20.1908, "lon": 30.7717, "kind": "Historic landing site", "source_note": "Approximate historic landing coordinate"},
}

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "HORUS-Lunar-Site-Navigator/2.1 (educational research project)"})

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def metric_panel(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="panel"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def plot_theme(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#122238",
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=12),
        title=dict(font=dict(size=15, color=TEXT), x=.02, xanchor="left"),
        margin=dict(l=55, r=25, t=65, b=55), height=height,
        hoverlabel=dict(bgcolor="#1B304A", font_color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(164,184,211,.10)", automargin=True, title_standoff=12)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(164,184,211,.10)", automargin=True, title_standoff=12)
    return fig


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _find_column(header: list[str], tokens: tuple[str, ...]) -> int | None:
    normalized = [h.lower().replace(" ", "").replace("_", "").replace("-", "") for h in header]
    for i, name in enumerate(normalized):
        if all(token in name for token in tokens):
            return i
    return None


def parse_horizons_csv(raw: str) -> pd.DataFrame:
    """Parse the Horizons observer table using its header labels, not fixed column positions."""
    if "$$SOE" not in raw or "$$EOE" not in raw:
        raise ValueError("JPL Horizons returned no ephemeris rows. Check the coordinates and date.")
    before, body = raw.split("$$SOE", 1)
    body = body.split("$$EOE", 1)[0]
    header = None
    for line in before.splitlines():
        if "Azimuth" in line and "Elevation" in line:
            try:
                candidate = next(csv.reader([line.strip()]))
                if len(candidate) >= 3:
                    header = candidate
            except Exception:
                pass
    rows = []
    for line in body.strip().splitlines():
        if not line.strip():
            continue
        try:
            parsed_row = next(csv.reader([line.strip()]))
            # Horizons CSV output may include a trailing delimiter; discard empty tail cells.
            while parsed_row and not parsed_row[-1].strip():
                parsed_row.pop()
            rows.append(parsed_row)
        except Exception:
            continue
    if not rows:
        raise ValueError("The Horizons ephemeris table was empty.")
    az_i = _find_column(header or [], ("azi",))
    el_i = _find_column(header or [], ("elev",))
    # Horizons observer tables normally end in azimuth/elevation; fallback only if
    # the service header format changes.
    if az_i is None or el_i is None:
        az_i, el_i = -2, -1
    time_i = 0
    parsed = []
    for row in rows:
        try:
            az = float(row[az_i].strip())
            el = float(row[el_i].strip())
            stamp = row[time_i].strip()
            parsed.append({"Raw time": stamp, "Azimuth (deg)": az, "Elevation (deg)": el})
        except (ValueError, IndexError):
            continue
    if not parsed:
        raise ValueError("Could not parse azimuth/elevation values from Horizons.")
    frame = pd.DataFrame(parsed)
    # Horizons timestamps commonly look like 2026-Nov-12 00:00.
    frame["Time"] = pd.to_datetime(frame["Raw time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["Time"]).sort_values("Time").drop_duplicates("Time")
    if frame.empty:
        raise ValueError("Could not interpret timestamps returned by Horizons.")
    return frame[["Time", "Azimuth (deg)", "Elevation (deg)"]].reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_horizons_track(target: str, longitude: float, latitude: float,
                         start_iso: str, duration_hours: int, step_minutes: int) -> pd.DataFrame:
    """Fetch a time series of apparent topocentric azimuth/elevation from JPL Horizons."""
    start = datetime.strptime(start_iso, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    stop = start + timedelta(hours=duration_hours)
    # Horizons geodetic SITE_COORD longitude is east-positive.
    east_lon = longitude % 360.0
    params = {
        "format": "json",
        "COMMAND": f"'{target}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "coord@301",
        "COORD_TYPE": "GEODETIC",
        "SITE_COORD": f"{east_lon:.6f},{latitude:.6f},0",
        "START_TIME": f"'{start.strftime('%Y-%m-%d %H:%M')}'",
        "STOP_TIME": f"'{stop.strftime('%Y-%m-%d %H:%M')}'",
        "STEP_SIZE": f"'{step_minutes} m'",
        "QUANTITIES": "'4'",
        "CSV_FORMAT": "YES",
        "ANG_FORMAT": "DEG",
        "EXTRA_PREC": "YES",
    }
    response = SESSION.get(HORIZONS_URL, params=params, timeout=75)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(str(payload["error"]))
    result = payload.get("result", "")
    frame = parse_horizons_csv(result)
    frame["Target"] = "Sun" if target == "10" else "Earth"
    return frame


def interpolate_crossing(t0: pd.Timestamp, e0: float, t1: pd.Timestamp, e1: float) -> pd.Timestamp:
    if e1 == e0:
        return t1
    fraction = float(np.clip((0.0 - e0) / (e1 - e0), 0, 1))
    return t0 + (t1 - t0) * fraction


def analyze_windows(frame: pd.DataFrame, threshold: float = 0.0) -> dict[str, Any]:
    """Calculate sampled above-horizon windows and interpolate horizon crossings."""
    data = frame.sort_values("Time").reset_index(drop=True).copy()
    if len(data) < 2:
        return {"visible_hours": 0.0, "windows": [], "next_set": None, "next_rise": None}
    times = data["Time"].tolist()
    elev = data["Elevation (deg)"].astype(float).tolist()
    windows = []
    is_up = elev[0] > threshold
    start = times[0] if is_up else None
    next_rise = None
    next_set = None
    for i in range(1, len(data)):
        prev_up = elev[i-1] > threshold
        curr_up = elev[i] > threshold
        if not prev_up and curr_up:
            cross = interpolate_crossing(times[i-1], elev[i-1]-threshold, times[i], elev[i]-threshold)
            if next_rise is None:
                next_rise = cross
            start = cross
        elif prev_up and not curr_up:
            cross = interpolate_crossing(times[i-1], elev[i-1]-threshold, times[i], elev[i]-threshold)
            if next_set is None:
                next_set = cross
            if start is None:
                start = times[i-1]
            windows.append((start, cross))
            start = None
    if elev[-1] > threshold and start is not None:
        windows.append((start, times[-1]))
    visible_seconds = sum(max(0, (b-a).total_seconds()) for a, b in windows)
    return {
        "visible_hours": visible_seconds / 3600,
        "windows": windows,
        "next_set": next_set,
        "next_rise": next_rise,
    }


def format_duration(hours: float | None) -> str:
    if hours is None or not np.isfinite(hours):
        return "—"
    total = max(0, int(round(hours * 60)))
    return f"{total // 60}h {total % 60:02d}m"


def hours_until(now: datetime, future: pd.Timestamp | None) -> float | None:
    if future is None:
        return None
    return max(0.0, (future.to_pydatetime() - now).total_seconds() / 3600)


def make_sky_dome(track: pd.DataFrame, title: str) -> go.Figure:
    """3D horizon dome: azimuth/elevation mapped to a unit hemisphere."""
    fig = go.Figure()
    # Hemisphere mesh (z >= 0)
    phi = np.linspace(0, 2*np.pi, 45)
    theta = np.linspace(0, np.pi/2, 20)
    pp, tt = np.meshgrid(phi, theta)
    xx = np.cos(tt) * np.sin(pp)
    yy = np.cos(tt) * np.cos(pp)
    zz = np.sin(tt)
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz, opacity=.12, showscale=False,
        colorscale=[[0, "#24415D"], [1, "#75B7E8"]],
        name="Visible sky hemisphere", hoverinfo="skip"
    ))
    # Horizon ring
    ring = np.linspace(0, 2*np.pi, 180)
    fig.add_trace(go.Scatter3d(
        x=np.sin(ring), y=np.cos(ring), z=np.zeros_like(ring),
        mode="lines", line=dict(color="#E8BC68", width=5), name="Local horizon"
    ))
    for target, color in [("Sun", GOLD), ("Earth", BLUE)]:
        part = track[track["Target"] == target].copy()
        if part.empty:
            continue
        az = np.deg2rad(part["Azimuth (deg)"].to_numpy())
        el = np.deg2rad(part["Elevation (deg)"].to_numpy())
        radius = np.cos(el)
        x = radius * np.sin(az)
        y = radius * np.cos(az)
        z = np.sin(el)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines+markers",
            marker=dict(size=2.8, color=color), line=dict(color=color, width=5),
            name=target, text=part["Time"].dt.strftime("%Y-%m-%d %H:%M UTC"),
            hovertemplate=f"{target}<br>%{{text}}<br>Azimuth: %{{customdata[0]:.1f}}°<br>Elevation: %{{customdata[1]:.1f}}°<extra></extra>",
            customdata=np.column_stack([part["Azimuth (deg)"], part["Elevation (deg)"]])
        ))
    fig.update_layout(
        title=title, scene=dict(
            bgcolor="#122238",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="cube", camera=dict(eye=dict(x=1.55, y=1.55, z=1.05))
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
        margin=dict(l=0,r=0,t=55,b=0), height=510,
        legend=dict(orientation="h", y=1.02, x=0)
    )
    return fig


def create_animation(track: pd.DataFrame) -> go.Figure:
    """Animated 3D sky plot with hourly frames."""
    data = track.copy()
    data["Az rad"] = np.deg2rad(data["Azimuth (deg)"])
    data["El rad"] = np.deg2rad(data["Elevation (deg)"])
    data["x"] = np.cos(data["El rad"]) * np.sin(data["Az rad"])
    data["y"] = np.cos(data["El rad"]) * np.cos(data["Az rad"])
    data["z"] = np.sin(data["El rad"])
    data["Frame"] = data["Time"].dt.strftime("%Y-%m-%d %H:%M UTC")
    # Keep frames to a manageable count, while preserving the selected time span.
    frames = sorted(data["Frame"].unique())
    if len(frames) > 48:
        chosen = set(frames[::max(1, len(frames)//48)])
        chosen.add(frames[-1])
        data = data[data["Frame"].isin(chosen)]
    fig = px.scatter_3d(
        data, x="x", y="y", z="z", color="Target", animation_frame="Frame",
        color_discrete_map={"Sun": GOLD, "Earth": BLUE},
        hover_data={"Azimuth (deg)":":.1f","Elevation (deg)":":.1f","Frame":True,"x":False,"y":False,"z":False},
        title="Animated Sun & Earth positions — local sky dome"
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
        scene=dict(bgcolor="#122238", xaxis=dict(range=[-1,1],title="East / West"),
                   yaxis=dict(range=[-1,1],title="North / South"),
                   zaxis=dict(range=[-1,1],title="Elevation"), aspectmode="cube"),
        height=560, margin=dict(l=0,r=0,t=55,b=0)
    )
    return fig


def build_site_table(names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        item = REFERENCE_SITES[name]
        rows.append({"Site": name, "Latitude (°)": item["lat"], "Longitude (°)": item["lon"], "Type": item["kind"]})
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------
# NASA LOLA south-polar DEM: native in-app 2D map + 3D terrain patch
# ---------------------------------------------------------------------
LOLA_DEM_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF"
LOLA_PRODUCT_URL = "https://pgda.gsfc.nasa.gov/products/90"
MOON_RADIUS_M = 1_737_400.0


@st.cache_data(ttl=86400, show_spinner=False)
def load_lola_patch(latitude: float, longitude: float, radius_km: float = 18.0,
                    samples: int = 150) -> dict[str, Any]:
    """Read a small window from NASA's cloud-optimized LOLA DEM using HTTP range reads.

    The source product is an 80 m/pixel south-polar stereographic COG covering 80–90°S.
    Only a small window is read; the full ~181 MB GeoTIFF is not downloaded by HORUS.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds
    from pyproj import CRS, Transformer

    if latitude > -80.0:
        raise ValueError("The selected coordinate is north of the LOLA 80–90°S product coverage.")

    # The input geographic coordinates are lunar planetocentric coordinates on a
    # spherical Moon. The destination projection is read from the authoritative COG.
    lunar_geographic = CRS.from_proj4(
        f"+proj=longlat +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +no_defs +type=crs"
    )
    vsi_url = "/vsicurl/" + LOLA_DEM_URL
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="5000000",
    ):
        with rasterio.open(vsi_url) as ds:
            if ds.crs is None:
                raise ValueError("NASA LOLA GeoTIFF has no readable projection metadata.")
            transformer = Transformer.from_crs(lunar_geographic, ds.crs, always_xy=True)
            center_x, center_y = transformer.transform(float(longitude) % 360.0, float(latitude))
            half = float(radius_km) * 1000.0
            raw_window = from_bounds(
                center_x-half, center_y-half, center_x+half, center_y+half,
                transform=ds.transform
            )
            full = Window(0, 0, ds.width, ds.height)
            window = raw_window.intersection(full)
            if window.width < 2 or window.height < 2:
                raise ValueError("The selected coordinate is outside the available LOLA raster footprint.")
            out_h = max(30, min(int(samples), int(window.height)))
            out_w = max(30, min(int(samples), int(window.width)))
            arr = ds.read(
                1, window=window, out_shape=(out_h, out_w),
                resampling=Resampling.average, masked=True
            ).astype("float64")
            win_transform = ds.window_transform(window)
            xs = win_transform.c + (np.arange(out_w) + 0.5) * win_transform.a
            ys = win_transform.f + (np.arange(out_h) + 0.5) * win_transform.e
            z = np.asarray(arr.filled(np.nan), dtype=float)
            if ds.nodata is not None:
                z[np.isclose(z, ds.nodata)] = np.nan
            # Slope is calculated from the sampled DEM grid; this is a visualization
            # estimate, not a replacement for NASA's dedicated slope raster.
            dy = abs(float(win_transform.e)) * max(1, window.height / out_h)
            dx = abs(float(win_transform.a)) * max(1, window.width / out_w)
            safe = np.where(np.isfinite(z), z, np.nanmedian(z))
            gy, gx = np.gradient(safe, dy, dx)
            slope = np.degrees(np.arctan(np.hypot(gx, gy)))
            return {
                "x": xs.tolist(), "y": ys.tolist(), "z": z.tolist(),
                "slope": slope.tolist(), "center_x": float(center_x),
                "center_y": float(center_y), "pixel_m": float(max(dx, dy)),
                "source": LOLA_DEM_URL,
                "bounds": [float(np.nanmin(z)), float(np.nanmax(z))],
            }


def lunar_xy(latitude: float, longitude: float, target_crs: Any) -> tuple[float, float]:
    """Project lunar planetocentric coordinates into the LOLA product CRS."""
    from pyproj import CRS, Transformer
    lunar_geographic = CRS.from_proj4(
        f"+proj=longlat +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +no_defs +type=crs"
    )
    tr = Transformer.from_crs(lunar_geographic, target_crs, always_xy=True)
    return tr.transform(float(longitude) % 360.0, float(latitude))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_lunar_wac_mosaic(zoom: int = 2) -> list[list[list[int]]]:
    """Fetch and stitch NASA/JPL Moon Trek LROC WAC global WMTS tiles."""
    from PIL import Image
    import concurrent.futures

    base = "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm"
    cols, rows = 2 * (2 ** zoom), 1 * (2 ** zoom)
    tile_size = 256
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), (8, 15, 27))

    def get_tile(rc: tuple[int, int]):
        row, col = rc
        url = f"{base}/{zoom}/{row}/{col}.jpg"
        response = requests.get(url, timeout=15, headers={"User-Agent": "HORUS-Lunar-Site-Navigator/2.1"})
        response.raise_for_status()
        return row, col, Image.open(io.BytesIO(response.content)).convert("RGB")

    coords = [(r, c) for r in range(rows) for c in range(cols)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for row, col, tile in pool.map(get_tile, coords):
            if tile.size != (tile_size, tile_size):
                tile = tile.resize((tile_size, tile_size))
            canvas.paste(tile, (col * tile_size, row * tile_size))
    # Downsample for efficient browser rendering and the interactive 3D mesh.
    preview = canvas.resize((360, 180), Image.Resampling.BILINEAR)
    return np.asarray(preview, dtype=np.uint8).tolist()


def add_global_lunar_views(candidate_df: pd.DataFrame, observer_name: str,
                           observer_lat: float, observer_lon: float) -> None:
    """Render NASA WAC global imagery and a textured interactive globe in HORUS."""
    st.markdown("### Global lunar imagery & 3D globe")
    st.caption("NASA/JPL Moon Trek LROC WAC global mosaic, served as map tiles and rendered inside HORUS.")
    map_tab, globe_tab = st.tabs(["Interactive global map", "3D lunar globe"])
    tile_url = "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
    markers = []
    for _, row in candidate_df.iterrows():
        markers.append({"name": str(row["Site"]), "lat": float(row["Latitude (°)"]), "lon": float(row["Longitude (°)"])})
    markers.append({"name": str(observer_name), "lat": float(observer_lat), "lon": float(observer_lon), "active": True})
    marker_json = __import__("json").dumps(markers)
    with map_tab:
        html = f"""<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>html,body,#map{{height:590px;width:100%;margin:0;background:#080f1b}}.leaflet-container{{background:#080f1b;color:#edf2f8}}.leaflet-control-attribution{{background:rgba(8,15,27,.8)!important;color:#cbd5e1!important}}.leaflet-control-attribution a{{color:#e8bc68!important}}</style></head><body><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
        const map=L.map('map',{{crs:L.CRS.EPSG4326,center:[{observer_lat},{observer_lon}],zoom:2,minZoom:0,maxZoom:7,worldCopyJump:false}});
        L.tileLayer('{tile_url}',{{tileSize:256,noWrap:true,minZoom:0,maxZoom:7,attribution:'NASA/JPL Moon Trek · LROC WAC Global Mosaic'}}).addTo(map);
        const sites={marker_json}; sites.forEach(s=>{{const active=!!s.active;const m=L.circleMarker([s.lat,s.lon],{{radius:active?8:6,color:active?'#f5d99d':'#75b7e8',weight:2,fillColor:active?'#e8bc68':'#75b7e8',fillOpacity:.95}}).addTo(map);m.bindPopup('<b>'+s.name+'</b><br>'+s.lat.toFixed(4)+'°, '+s.lon.toFixed(4)+'°');}});
        L.control.scale({{imperial:false}}).addTo(map);
        </script></body></html>"""
        components.html(html, height=610, scrolling=False)
    with globe_tab:
        try:
            with st.spinner("Loading NASA/JPL global lunar image tiles for the 3D globe…"):
                rgb = np.asarray(fetch_lunar_wac_mosaic(2), dtype=np.uint8)
            nlat, nlon = rgb.shape[:2]
            latitudes = np.linspace(90, -90, nlat)
            longitudes = np.linspace(-180, 180, nlon, endpoint=False)
            latg, long = np.meshgrid(np.deg2rad(latitudes), np.deg2rad(longitudes), indexing="ij")
            xx = np.cos(latg) * np.cos(long)
            yy = np.cos(latg) * np.sin(long)
            zz = np.sin(latg)
            vertex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r,g,b in rgb.reshape(-1,3)]
            ii=[]; jj=[]; kk=[]
            for r in range(nlat-1):
                for c in range(nlon):
                    c2=(c+1)%nlon
                    a=r*nlon+c; b=r*nlon+c2; d=(r+1)*nlon+c; e=(r+1)*nlon+c2
                    ii.extend([a,b,d,b,e,d]); jj.extend([b,d,e,e,d,b]); kk.extend([d,e,a,d,a,e])
            globe=go.Figure(go.Mesh3d(x=xx.ravel(),y=yy.ravel(),z=zz.ravel(),i=ii,j=jj,k=kk,
                                      vertexcolor=vertex_colors,flatshading=False,lighting=dict(ambient=.62,diffuse=.72,specular=.12,roughness=.9),
                                      hoverinfo="skip",name="NASA LROC WAC image mosaic"))
            for _, site in candidate_df.iterrows():
                latr=np.deg2rad(float(site["Latitude (°)"])); lonr=np.deg2rad(float(site["Longitude (°)"]))
                globe.add_trace(go.Scatter3d(x=[1.012*np.cos(latr)*np.cos(lonr)],y=[1.012*np.cos(latr)*np.sin(lonr)],z=[1.012*np.sin(latr)],
                                             mode="markers+text",marker=dict(size=4,color="#f5d99d"),text=[str(site["Site"])],
                                             textposition="top center",name=str(site["Site"])))
            globe.update_layout(title="NASA LROC WAC global image mosaic · interactive globe",
                                scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                                           bgcolor="#080f1b",aspectmode="cube",camera=dict(eye=dict(x=1.5,y=1.5,z=1.1))),
                                paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT),height=650,
                                margin=dict(l=0,r=0,t=55,b=0),legend=dict(orientation="h",y=-.03))
            st.plotly_chart(globe,use_container_width=True,config={"displayModeBar":True})
            st.caption("Global imagery is the LROC WAC mosaic; polar elevation is provided separately by the LOLA DEM patch below.")
        except Exception as exc:
            st.error("The NASA global image tiles could not be assembled for the 3D globe in this session.")
            st.caption(f"Technical detail: {str(exc)[:350]}")
            st.info("The interactive map above still requests the same NASA/JPL tiles directly in the browser.")


def add_lunar_visuals(candidate_df: pd.DataFrame, observer_name: str,
                      observer_lat: float, observer_lon: float) -> None:
    """Render a real LOLA south-polar elevation patch directly inside HORUS."""
    section(
        "03 / INTEGRATED LUNAR DATA",
        "Lunar surface explorer — NASA LOLA terrain",
        "The south-polar terrain below is sampled directly from NASA's cloud-optimized LOLA GeoTIFF. Only the selected local window is fetched."
    )
    st.markdown(
        f'<div class="info-card"><div class="info-title">Active terrain focus</div>'
        f'<div class="info-copy"><b>{observer_name.replace("<", "&lt;").replace(">", "&gt;")}</b> · '
        f'{observer_lat:.4f}° latitude · {observer_lon:.4f}° longitude. '
        'Candidate coordinates are reference points, not certified landing ellipses.</div></div>',
        unsafe_allow_html=True,
    )
    add_global_lunar_views(candidate_df, observer_name, observer_lat, observer_lon)

    if observer_lat > -80.0:
        st.warning("NASA's selected LOLA south-polar DEM covers 80–90°S. Choose a south-polar candidate or custom coordinate at 80°S or farther south to load measured terrain. The global coordinate list remains available below.")
        st.link_button("NASA LOLA dataset details", LOLA_PRODUCT_URL)
        return

    radius_km = st.select_slider("Terrain patch radius", options=[5, 10, 18, 25, 35], value=18, format_func=lambda v: f"{v} km")
    with st.spinner("Reading the selected window from NASA's LOLA elevation GeoTIFF…"):
        try:
            patch = load_lola_patch(observer_lat, observer_lon, float(radius_km), 150)
        except Exception as exc:
            st.error("The in-app NASA LOLA terrain request could not be completed. This can happen if the host blocks remote GeoTIFF range requests or the data service is temporarily unavailable.")
            st.caption(f"Technical detail: {str(exc)[:500]}")
            st.link_button("Open NASA LOLA dataset documentation", LOLA_PRODUCT_URL)
            st.code(LOLA_DEM_URL, language="text")
            return

    z = np.asarray(patch["z"], dtype=float)
    slope = np.asarray(patch["slope"], dtype=float)
    x = np.asarray(patch["x"], dtype=float) / 1000.0
    y = np.asarray(patch["y"], dtype=float) / 1000.0
    valid = np.isfinite(z)
    if not valid.any():
        st.warning("No valid LOLA elevation pixels were returned for this window.")
        return
    center_elev = float(np.nanmedian(z[max(0,len(y)//2-2):len(y)//2+3,
                                      max(0,len(x)//2-2):len(x)//2+3]))
    max_slope = float(np.nanpercentile(slope[valid], 95))
    a,b,c,d = st.columns(4)
    with a: metric_panel("LOLA ELEVATION RANGE", f"{np.nanmin(z):.0f}–{np.nanmax(z):.0f} m", "Within displayed patch")
    with b: metric_panel("PATCH CENTER ELEVATION", f"{center_elev:.0f} m", "Median of center pixels")
    with c: metric_panel("95TH PERCENTILE SLOPE", f"{max_slope:.1f}°", "Derived from sampled DEM")
    with d: metric_panel("SOURCE GRID", f"{patch['pixel_m']:.0f} m", "Approx. sampled pixel spacing")

    tabs = st.tabs(["2D Elevation Map", "3D Terrain Model", "Slope Preview", "Data & Coverage"])
    with tabs[0]:
        fig = go.Figure(go.Heatmap(
            x=x, y=y, z=z, colorscale="Earth", colorbar=dict(title="Elevation (m)"),
            hovertemplate="X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Elevation: %{z:.1f} m<extra>NASA LOLA</extra>"
        ))
        fig.add_trace(go.Scatter(x=[patch["center_x"]/1000], y=[patch["center_y"]/1000],
                                 mode="markers+text", marker=dict(size=12,color="#F5D99D",symbol="star"),
                                 text=[observer_name], textposition="top center", name="Active site"))
        fig.update_layout(title="Measured lunar elevation · south-polar stereographic",
                          xaxis_title="LOLA X (km)", yaxis_title="LOLA Y (km)",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#122238",
                          font=dict(color=TEXT), height=560, margin=dict(l=30,r=30,t=60,b=35))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    with tabs[1]:
        # Surface mesh uses the actual DEM heights, with vertical exaggeration only for visibility.
        fig3 = go.Figure(go.Surface(
            x=x, y=y, z=z, surfacecolor=z, colorscale="Earth",
            colorbar=dict(title="Elevation (m)"), connectgaps=False,
            hovertemplate="X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Elevation: %{z:.1f} m<extra>NASA LOLA DEM</extra>"
        ))
        fig3.update_layout(title="3D LOLA elevation surface · measured DEM pixels",
                           scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Elevation (m)",
                                      bgcolor="#122238", aspectmode="manual",
                                      aspectratio=dict(x=1,y=1,z=.22)),
                           paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
                           height=620, margin=dict(l=0,r=0,t=60,b=0))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": True})
        st.caption("The vertical axis is visually compressed; the surface elevations themselves are not altered. Rotate and zoom to inspect the local terrain.")

    with tabs[2]:
        fs = go.Figure(go.Heatmap(x=x, y=y, z=slope, colorscale="Turbo",
                                  colorbar=dict(title="Slope (°)"),
                                  hovertemplate="X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Estimated slope: %{z:.2f}°<extra>Derived from LOLA DEM</extra>"))
        fs.update_layout(title="Local slope estimate derived from sampled LOLA elevations",
                         xaxis_title="LOLA X (km)", yaxis_title="LOLA Y (km)",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#122238",
                         font=dict(color=TEXT), height=540)
        st.plotly_chart(fs, use_container_width=True, config={"displayModeBar": True})
        st.warning("This is a first-order slope estimate from the displayed resampled DEM window. It is not the dedicated NASA LDSM product and must not be used as a certified landing hazard assessment.")

    with tabs[3]:
        st.markdown("**Data provenance**")
        st.markdown(
            "- Product: NASA/GSFC LOLA south-polar large-area DEM, `LDEM_80S_80MPP_ADJ.TIF`\n"
            "- Coverage: 80°S to 90°S\n"
            "- Native grid: 80 m/pixel; HORUS reads a smaller window and resamples it for interactive plotting\n"
            "- Elevation values: meters; projection metadata is read from the GeoTIFF itself\n"
            "- The 2D and 3D views are rendered by HORUS/Plotly, not an embedded NASA webpage."
        )
        st.markdown(f"[NASA PGDA product page]({LOLA_PRODUCT_URL}) · [Direct cloud-optimized GeoTIFF]({LOLA_DEM_URL})")
        st.caption("Citation: Barker et al. (2023), Planetary Science Journal 4, 183. DOI: 10.3847/PSJ/acf3e1.")

    st.download_button(
        "Download displayed LOLA patch (CSV)",
        data=pd.DataFrame(z, index=y, columns=x).to_csv().encode("utf-8"),
        file_name="HORUS_LOLA_elevation_patch.csv", mime="text/csv"
    )
    st.caption("Coverage note: this measured terrain layer is available for the south-polar 80–90°S region only. Other latitudes are not filled with synthetic elevation.")


# ---------------------------------------------------------------------
# Sidebar: user level and data configuration
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌙 HORUS 2.1")
    st.caption("LUNAR MISSION CONTROL")
    st.divider()
    user_mode = st.radio("Interface mode", ["Mission Planner", "Explorer / Educator"], index=0)
    st.markdown("#### SITE SELECTION")
    site_names = list(REFERENCE_SITES.keys())
    primary_site = st.selectbox("Primary landing site", site_names, index=0)
    compare_sites = st.multiselect(
        "Compare additional sites",
        [n for n in site_names if n != primary_site],
        default=["Shackleton Rim", "de Gerlache Rim 2"],
        max_selections=5,
    )
    st.divider()
    st.markdown("#### CUSTOM COORDINATES")
    use_custom = st.toggle("Use a custom observer location", value=False)
    if use_custom:
        custom_name = st.text_input("Custom site label", "My lunar site")
        custom_lat = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=-89.0, step=0.01, format="%.4f")
        custom_lon = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, format="%.4f")
        observer_name = custom_name.strip() or "Custom site"
        observer_lat, observer_lon = float(custom_lat), float(custom_lon)
    else:
        observer_name = primary_site
        observer_lat = float(REFERENCE_SITES[primary_site]["lat"])
        observer_lon = float(REFERENCE_SITES[primary_site]["lon"])
    st.divider()
    st.markdown("#### ANALYSIS WINDOW")
    duration_hours = st.select_slider("Timeline duration", options=[6, 12, 24, 36, 48, 72], value=24, format_func=lambda x: f"{x} hours")
    step_minutes = st.select_slider("Sampling interval", options=[5, 10, 15, 30, 60], value=15, format_func=lambda x: f"{x} min")
    st.caption("Smaller intervals provide finer crossing estimates but require more API data.")

# ---------------------------------------------------------------------
# Header and mission overview
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">LUNAR EXPLORATION SYSTEM · SOUTH POLE OPERATIONS</div>
      <div class="hero-title">🌙 HORUS</div>
      <div class="hero-sub">Lunar Landing Site Navigator · Rapid Site-Date Feasibility Explorer</div>
      <div class="chips">
        <span class="chip">JPL HORIZONS EPHEMERIS</span>
        <span class="chip">SITE COMPARISON</span>
        <span class="chip">ILLUMINATION WINDOWS</span>
        <span class="chip">EARTH VISIBILITY</span>
      </div>
    </div>
    """, unsafe_allow_html=True
)

section("01 / MISSION OVERVIEW", "Mission snapshot", "Select a candidate site, a UTC observation time, and a mission analysis window.")
overview = st.columns(4)
with overview[0]:
    metric_panel("ACTIVE OBSERVER", observer_name, "Selected lunar coordinates")
with overview[1]:
    metric_panel("LATITUDE", f"{observer_lat:.4f}°", "Planetocentric latitude input")
with overview[2]:
    metric_panel("LONGITUDE", f"{observer_lon:.4f}°", "East/west longitude input")
with overview[3]:
    metric_panel("ANALYSIS WINDOW", f"{duration_hours} hours", f"Sampling every {step_minutes} minutes")

# ---------------------------------------------------------------------
# Date/time controls
# ---------------------------------------------------------------------
section("02 / MISSION SCHEDULING", "Choose landing date & time (UTC)", "Explore how changing the landing time changes sunlight and Earth visibility.")
now_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
with st.form("mission_time_form"):
    d1, d2, d3 = st.columns([1,1,1.25])
    with d1:
        landing_date = st.date_input("Landing date (UTC)", value=now_utc.date(), key="landing_date")
    with d2:
        landing_time = st.time_input("Landing time (UTC)", value=now_utc.time().replace(tzinfo=None), key="landing_time")
    with d3:
        st.text_input("Observer", value=observer_name, disabled=True)
    run_analysis = st.form_submit_button("↗  RUN SITE-DATE ANALYSIS", use_container_width=True)

start_dt = datetime.combine(landing_date, landing_time).replace(tzinfo=timezone.utc)
start_iso = iso_utc(start_dt)

# ---------------------------------------------------------------------
# Lunar map/globe and candidate comparison
# ---------------------------------------------------------------------
all_selected = [primary_site] + compare_sites
candidate_df = build_site_table(all_selected)
if use_custom:
    candidate_df = pd.concat([candidate_df, pd.DataFrame([{
        "Site": observer_name, "Latitude (°)": observer_lat, "Longitude (°)": observer_lon, "Type": "Custom observer"
    }])], ignore_index=True)

add_lunar_visuals(candidate_df, observer_name, observer_lat, observer_lon)

section("04 / SITE COMPARISON", "Candidate coordinates", "Compare the selected sites and inspect their geographic positions.")
comparison = candidate_df.copy()
comparison["Latitude offset from South Pole (°)"] = 90 - comparison["Latitude (°)"].abs()
st.dataframe(comparison, use_container_width=True, hide_index=True,
             column_config={"Latitude (°)":st.column_config.NumberColumn(format="%.4f"),
                            "Longitude (°)":st.column_config.NumberColumn(format="%.4f"),
                            "Latitude offset from South Pole (°)":st.column_config.NumberColumn(format="%.4f")})
st.caption("Latitude offset is a geographic descriptor only. It is not a landing suitability or safety score.")

# ---------------------------------------------------------------------
# Run ephemeris analysis
# ---------------------------------------------------------------------
if run_analysis:
    with st.spinner("Querying NASA/JPL Horizons for the Sun and Earth tracks…"):
        try:
            sun_track = fetch_horizons_track("10", observer_lon, observer_lat, start_iso, duration_hours, step_minutes)
            earth_track = fetch_horizons_track("399", observer_lon, observer_lat, start_iso, duration_hours, step_minutes)
            combined = pd.concat([sun_track, earth_track], ignore_index=True).sort_values("Time")
            st.session_state["horus_analysis"] = {
                "site": observer_name, "lat": observer_lat, "lon": observer_lon,
                "start": start_iso, "duration": duration_hours, "step": step_minutes,
                "track": combined,
            }
            st.session_state.pop("horus_error", None)
        except requests.exceptions.Timeout:
            st.session_state["horus_error"] = "JPL Horizons timed out. Try a longer sampling interval or a shorter analysis window."
            st.session_state.pop("horus_analysis", None)
        except requests.exceptions.RequestException as exc:
            st.session_state["horus_error"] = f"Network/API request failed: {exc}"
            st.session_state.pop("horus_analysis", None)
        except Exception as exc:
            st.session_state["horus_error"] = f"Could not process the ephemeris response: {exc}"
            st.session_state.pop("horus_analysis", None)

if st.session_state.get("horus_error"):
    st.error(st.session_state["horus_error"])

analysis = st.session_state.get("horus_analysis")
if analysis:
    if analysis["site"] != observer_name or analysis["lat"] != observer_lat or analysis["lon"] != observer_lon:
        st.warning("The results below belong to a previously selected observer. Run the analysis again for the current site.")
    if analysis["start"] != start_iso:
        st.warning("The results below belong to a previously selected UTC date/time. Run the analysis again to refresh.")
    track = analysis["track"].copy()
    sun_data = track[track["Target"] == "Sun"].copy()
    earth_data = track[track["Target"] == "Earth"].copy()
    sun_windows = analyze_windows(sun_data, 0.0)
    earth_windows = analyze_windows(earth_data, 0.0)
    sun_operational = analyze_windows(sun_data, 5.0)
    current = datetime.strptime(analysis["start"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    sun_start_el = float(sun_data.iloc[0]["Elevation (deg)"])
    earth_start_el = float(earth_data.iloc[0]["Elevation (deg)"])
    next_sunset_h = hours_until(current, sun_windows["next_set"])
    next_earthset_h = hours_until(current, earth_windows["next_set"])
    next_sunrise_h = hours_until(current, sun_windows["next_rise"])
    next_earthrise_h = hours_until(current, earth_windows["next_rise"])

    section("05 / POWER & COMMUNICATION", "Illumination and Earth visibility", "Horizons apparent elevation sampled through the selected mission window.")
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        metric_panel("SUN ABOVE HORIZON", format_duration(sun_windows["visible_hours"]), "Geometric sunlight in selected window")
    with k2:
        metric_panel("SUN ABOVE 5°", format_duration(sun_operational["visible_hours"]), "Higher-elevation illumination proxy")
    with k3:
        metric_panel("EARTH ABOVE HORIZON", format_duration(earth_windows["visible_hours"]), "Geometric Earth-center visibility")
    with k4:
        metric_panel("NEXT SUNSET", format_duration(next_sunset_h), "From selected UTC start time")

    c1,c2 = st.columns(2)
    with c1:
        st.metric("Sun elevation at landing time", f"{sun_start_el:.2f}°", "Above horizon" if sun_start_el > 0 else "Below horizon")
        st.metric("Next sunrise", format_duration(next_sunrise_h), "If a rise occurs in this analysis window")
    with c2:
        st.metric("Earth elevation at landing time", f"{earth_start_el:.2f}°", "Geometrically visible" if earth_start_el > 0 else "Below horizon")
        st.metric("Next Earthset", format_duration(next_earthset_h), "If a set occurs in this analysis window")
        st.metric("Next Earthrise", format_duration(next_earthrise_h), "If a rise occurs in this analysis window")

    # Solar power proxy: ideal horizontal receiver, no atmospheric attenuation, no terrain.
    sun_data["Horizontal irradiance factor"] = np.maximum(0, np.sin(np.deg2rad(sun_data["Elevation (deg)"])))
    sun_data["Operational light (>5°)"] = sun_data["Elevation (deg)"] > 5
    dt_hours = step_minutes / 60.0
    equivalent_sun_hours = float(np.trapezoid(sun_data["Horizontal irradiance factor"], dx=dt_hours))
    st.markdown('<div class="info-card"><div class="info-title">Solar power proxy</div>'
                f'<div class="info-copy">Idealized horizontal-panel equivalent full-sun exposure over this window: '
                f'<b style="color:{GOLD2}">{equivalent_sun_hours:.2f} hours</b>. '
                'This integrates max(0, sin(solar elevation)) and is a geometry-only proxy; it is not electrical energy, '
                'does not assume a real panel orientation/efficiency, and does not include terrain shadows or thermal constraints.</div></div>',
                unsafe_allow_html=True)

    # Timeline
    section("06 / TEMPORAL DYNAMICS", "Sun & Earth elevation timeline", "The 0° line is the geometric horizon; positive elevation indicates geometric visibility.")
    timeline = track.copy()
    timeline["Object"] = timeline["Target"]
    timeline_fig = px.line(timeline, x="Time", y="Elevation (deg)", color="Object",
                           color_discrete_map={"Sun":GOLD,"Earth":BLUE},
                           title="Apparent elevation above the local horizon")
    timeline_fig.add_hline(y=0, line_dash="dash", line_color=RED, annotation_text="Horizon (0°)")
    timeline_fig.add_hline(y=5, line_dash="dot", line_color=GREEN, annotation_text="5° solar threshold")
    timeline_fig.update_layout(xaxis_title="UTC time", yaxis_title="Elevation (°)")
    plot_theme(timeline_fig, 430)
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar":False})

    # Tabular windows
    def windows_frame(name: str, result: dict[str, Any]) -> list[dict[str,str]]:
        out=[]
        for idx,(a,b) in enumerate(result["windows"],1):
            out.append({"Object":name,"Window":str(idx),"Rise / start (UTC)":a.strftime("%Y-%m-%d %H:%M"),
                        "Set / end (UTC)":b.strftime("%Y-%m-%d %H:%M"),
                        "Duration":format_duration((b-a).total_seconds()/3600)})
        return out
    win_rows = windows_frame("Sun",sun_windows)+windows_frame("Earth",earth_windows)
    section("06 / OPERATIONAL WINDOWS", "Visibility windows", "Intervals are estimated from sampled elevation crossings with linear interpolation.")
    if win_rows:
        st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No complete above-horizon intervals were found in the selected analysis window. A partial interval may begin before or end after the selected window.")

    # 3D horizon dome and animation
    section("07 / LOCAL SKY VISUALIZATION", "Local celestial horizon dome", "Azimuth sets the horizontal direction; elevation determines height above or below the local horizon.")
    sky_fig = make_sky_dome(track, "Sun & Earth paths in the local sky")
    st.plotly_chart(sky_fig, use_container_width=True, config={"displayModeBar":False})
    with st.expander("▶ Animate celestial positions over time"):
        st.caption("Animation samples the computed ephemeris points. Use the Play control and timeline slider.")
        anim = create_animation(track)
        st.plotly_chart(anim, use_container_width=True, config={"displayModeBar":False})

    # Downloadable scientific output
    st.download_button("⬇ Download ephemeris data (CSV)", data=track.to_csv(index=False).encode("utf-8"),
                       file_name="horus_ephemeris.csv", mime="text/csv")

# ---------------------------------------------------------------------
# Site-date comparison for the selected curated sites
# ---------------------------------------------------------------------
section("08 / RAPID SITE COMPARISON", "Compare site geometry for the selected landing time",
        "This comparison requests a single-time Sun/Earth ephemeris for each selected reference coordinate.")
if st.button("Calculate Sun/Earth elevation for selected comparison sites", use_container_width=False):
    rows=[]
    targets=[("Sun","10"),("Earth","399")]
    with st.spinner("Comparing selected sites through JPL Horizons…"):
        for name in all_selected[:4]:
            coord=REFERENCE_SITES[name]
            row={"Site":name,"Latitude (°)":coord["lat"],"Longitude (°)":coord["lon"]}
            try:
                for label,target in targets:
                    one=fetch_horizons_track(target,coord["lon"],coord["lat"],start_iso,1,60)
                    row[f"{label} elevation (°)"]=float(one.iloc[0]["Elevation (deg)"])
                rows.append(row)
            except Exception as exc:
                row["Data status"]=str(exc)[:120]
                rows.append(row)
    st.session_state["horus_site_compare"] = pd.DataFrame(rows)

if "horus_site_compare" in st.session_state:
    comp=st.session_state["horus_site_compare"]
    st.dataframe(comp,use_container_width=True,hide_index=True)
    valid=comp.dropna(subset=["Sun elevation (°)","Earth elevation (°)"]) if "Sun elevation (°)" in comp else pd.DataFrame()
    if not valid.empty:
        melted=valid.melt(id_vars="Site",value_vars=["Sun elevation (°)","Earth elevation (°)"],
                          var_name="Object",value_name="Elevation (°)")
        f=px.bar(melted,x="Site",y="Elevation (°)",color="Object",barmode="group",
                 color_discrete_map={"Sun elevation (°)":GOLD,"Earth elevation (°)":BLUE},
                 title="Single-time Sun/Earth elevation by site")
        f.add_hline(y=0,line_dash="dash",line_color=RED)
        plot_theme(f,380)
        st.plotly_chart(f,use_container_width=True,config={"displayModeBar":False})

# ---------------------------------------------------------------------
# Accessibility: explainer mode and scientific limitations
# ---------------------------------------------------------------------
if user_mode == "Explorer / Educator":
    section("LEARN / LUNAR ENVIRONMENT", "Why landing-site timing matters", "A short guide for students and non-specialist users.")
    a,b,c=st.columns(3)
    with a:
        st.markdown('<div class="info-card"><div class="info-title">Sunlight</div><div class="info-copy">The Sun’s elevation changes the geometry of illumination. Low Sun angles create long shadows and can make terrain visibility and solar power planning more difficult.</div></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="info-card"><div class="info-title">Earth visibility</div><div class="info-copy">A site needs a geometrical line of sight to Earth for direct communication. Local ridges, antennas, and relay satellites can change the real communication picture.</div></div>',unsafe_allow_html=True)
    with c:
        st.markdown('<div class="info-card"><div class="info-title">Terrain</div><div class="info-copy">Slope, rocks, craters, roughness, and local shadows matter to landing safety. HORUS does not claim to assess these without validated high-resolution terrain data.</div></div>',unsafe_allow_html=True)

with st.expander("Scientific methodology, assumptions & limitations"):
    st.markdown("""
    **Ephemeris source:** NASA/JPL Horizons API, observer ephemerides for the Moon-centered geodetic observer (`coord@301`).

    **Coordinates:** The curated coordinates are approximate reference points. Longitude is converted to east-positive for the Horizons request. Verify coordinates against authoritative mission and lunar geospatial sources before operational use.

    **Sunlight:** Geometric sunlight is counted when apparent solar-center elevation is above 0°. The additional 5° duration is a user-facing operational threshold, not a universal engineering requirement. The horizontal irradiance factor is `max(0, sin(elevation))`; it is a dimensionless geometry proxy.

    **Communication:** Earth visibility is based on apparent Earth-center elevation above 0°. It is not a radio-frequency link budget and does not account for antenna gain, frequency, data rate, local terrain masking, relay satellites, or spacecraft attitude.

    **Terrain / landing safety:** No slope, roughness, boulder density, crater hazard, bearing strength, thermal stability, or landing dispersion analysis is calculated. No overall landing score is produced. Add these only when validated LOLA/SLDEM and hazard layers are integrated.

    **Sampling:** Horizon crossing times are estimated by linear interpolation between samples. A finer interval improves temporal resolution but does not eliminate model or terrain uncertainties.

    **Use:** HORUS is an educational rapid-evaluation prototype, not a certified mission-design or landing-safety system.
    """)

st.markdown(
    f'<div class="footer"><b>HORUS 2.1</b><div class="metric-note">LUNAR LANDING SITE NAVIGATOR · NASA SPACE APPS CHALLENGE</div></div>',
    unsafe_allow_html=True
)
