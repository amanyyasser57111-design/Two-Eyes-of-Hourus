# ---------------------------------------------------------------
# moon3d.py - builds the animated 3D Moon (pure Plotly + numpy).
# app.py gives it the Sun/Earth directions and it returns a figure.
# ---------------------------------------------------------------
import numpy as np
import plotly.graph_objects as go

# Night side = deep navy, then purple, orange, and pale gold where the Sun is high
COLORSCALE = [
    [0.00, "#04060f"],
    [0.12, "#14205a"],
    [0.35, "#6a3fc4"],
    [0.60, "#ff7a45"],
    [0.85, "#ffc85a"],
    [1.00, "#fff6d6"],
]


def latlon_to_xyz(lat_deg, lon_deg, r=1.0):
    """Moon latitude/longitude -> x, y, z (x points to lon 0, z to the north pole)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return (r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat))


def make_moon_figure(sun_vecs, earth_vecs, labels, sites):
    """
    sun_vecs, earth_vecs : arrays (N, 3), unit vectors in the Moon-fixed frame
    labels               : N strings (one date label per animation frame)
    sites                : list of (name, lat, lon, color) to mark on the Moon
    """
    n_frames = len(labels)

    # The Moon's surface: a grid of points on a sphere
    lat = np.radians(np.linspace(-90, 90, 50))
    lon = np.radians(np.linspace(-180, 180, 100))
    LON, LAT = np.meshgrid(lon, lat)
    X, Y, Z = np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)

    # A little fake "terrain" so the Moon isn't perfectly smooth
    texture = 0.90 + 0.10 * np.sin(9 * X + 4 * Y) * np.sin(7 * Z + 3 * X)

    def shade(sun):
        # Brightness = how directly the surface faces the Sun (0 = night)
        return np.clip(np.clip(X * sun[0] + Y * sun[1] + Z * sun[2], 0, 1) ** 0.8 * texture, 0, 1)

    # Trace 0: the Moon. Trace 1: Sun. Trace 2: Earth. Trace 3: sites. Trace 4: stars.
    moon = go.Surface(x=X, y=Y, z=Z, surfacecolor=shade(sun_vecs[0]),
                      colorscale=COLORSCALE, cmin=0, cmax=1, showscale=False,
                      lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                      hoverinfo="skip")

    def sun_trace(v):
        return go.Scatter3d(x=[2.1 * v[0]], y=[2.1 * v[1]], z=[2.1 * v[2]], mode="markers",
                            marker=dict(size=16, color="#ffd23f", line=dict(color="#fff3b0", width=4)),
                            name="Sun", hoverinfo="name")

    def earth_trace(v):
        return go.Scatter3d(x=[1.6 * v[0]], y=[1.6 * v[1]], z=[1.6 * v[2]], mode="markers",
                            marker=dict(size=9, color="#3fa7ff", line=dict(color="#bfe3ff", width=3)),
                            name="Earth", hoverinfo="name")

    sx, sy, sz, names, colors = [], [], [], [], []
    for name, la, lo, color in sites:
        x, y, z = latlon_to_xyz(la, lo, 1.03)
        sx.append(x); sy.append(y); sz.append(z); names.append(name); colors.append(color)
    site_trace = go.Scatter3d(x=sx, y=sy, z=sz, mode="markers+text", text=names,
                              textfont=dict(color="white", size=12), textposition="top center",
                              marker=dict(size=7, color=colors, symbol="diamond",
                                          line=dict(color="white", width=2)),
                              hoverinfo="text", name="Sites")

    rng = np.random.default_rng(1)                      # fixed seed = same stars every time
    s = rng.normal(size=(500, 3))
    s = 2.3 * s / np.linalg.norm(s, axis=1, keepdims=True)
    stars = go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="markers",
                         marker=dict(size=1.6, color="white", opacity=0.8),
                         hoverinfo="skip", showlegend=False)

    # One animation frame per time step: only the Moon colors, Sun and Earth change
    frames = [go.Frame(name=str(i),
                       data=[go.Surface(surfacecolor=shade(sun_vecs[i])),
                             sun_trace(sun_vecs[i]), earth_trace(earth_vecs[i])],
                       traces=[0, 1, 2])
              for i in range(n_frames)]

    fig = go.Figure(data=[moon, sun_trace(sun_vecs[0]), earth_trace(earth_vecs[0]),
                          site_trace, stars], frames=frames)

    south = dict(x=0.0, y=0.55, z=-1.35)
    near = dict(x=1.6, y=0.0, z=0.25)
    play = dict(frame=dict(duration=90, redraw=True), fromcurrent=True,
                transition=dict(duration=0), mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=False), mode="immediate",
                 transition=dict(duration=0))
    hidden = dict(visible=False)

    fig.update_layout(
        template="plotly_dark", height=700, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black", showlegend=False, uirevision="keep-camera",
        scene=dict(xaxis=dict(range=[-2.4, 2.4], **hidden),
                   yaxis=dict(range=[-2.4, 2.4], **hidden),
                   zaxis=dict(range=[-2.4, 2.4], **hidden),
                   aspectmode="cube", bgcolor="black", camera=dict(eye=south)),
        updatemenus=[
            dict(type="buttons", direction="left", x=0.02, y=0.06, showactive=False,
                 buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                          dict(label="⏸ Pause", method="animate", args=[[None], pause])]),
            dict(type="buttons", direction="left", x=0.02, y=0.99, showactive=False,
                 buttons=[dict(label="South pole view", method="relayout",
                               args=[{"scene.camera.eye": south}]),
                          dict(label="Near side view", method="relayout",
                               args=[{"scene.camera.eye": near}])]),
        ],
        sliders=[dict(active=0, x=0.2, len=0.78, y=0.04,
                      currentvalue=dict(prefix="Date: ", font=dict(color="white")),
                      steps=[dict(label=labels[i], method="animate",
                                  args=[[str(i)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))])
                             for i in range(n_frames)])],
    )
    return fig