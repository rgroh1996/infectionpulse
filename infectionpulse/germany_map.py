"""Offline Germany maps for regional forecasts and state-level observations.

State observations are genuine state measurements; forecasts remain macroregional.
Plain SVG Scatter polygons avoid geographic CDN assets and map-tile requests.
"""

import json
import math
from datetime import date, timedelta
from functools import lru_cache
from html import escape
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

BOUNDARIES = Path(__file__).resolve().parents[1] / "examples/germany_states.geojson"
COLORS = {
    "low": "#70BFA1",
    "moderate": "#EBC769",
    "high": "#E88780",
    "unknown": "#CBD5E1",
}
LEVEL_LABELS = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "unknown": "No fresh data",
}
REGION_NAMES = {
    "north_west": "Norden (West)",
    "central_west": "Mitte (West)",
    "east": "Osten",
    "south": "Süden",
}
LABEL_POINTS = {
    "north_west": (9.4, 53.1),
    "central_west": (7.9, 50.5),
    "east": (13.1, 51.7),
    "south": (10.5, 48.7),
}
COPYRIGHT = "© EuroGeographics for the administrative boundaries"
EXPECTED_STATES = {
    "DE1",
    "DE2",
    "DE3",
    "DE4",
    "DE5",
    "DE6",
    "DE7",
    "DE8",
    "DE9",
    "DEA",
    "DEB",
    "DEC",
    "DED",
    "DEE",
    "DEF",
    "DEG",
}


@lru_cache(maxsize=1)
def load_boundaries() -> dict:
    """Load the vendored public geometry; never fetch data during rendering."""
    data = json.loads(BOUNDARIES.read_text())
    features = data.get("features", [])
    if (
        len(features) != 16
        or {f["properties"]["NUTS_ID"] for f in features} != EXPECTED_STATES
    ):
        raise ValueError(
            "Germany boundary bundle must contain all 16 states exactly once"
        )
    if any(f["properties"].get("region_id") not in REGION_NAMES for f in features):
        raise ValueError("Each state must map to one official GrippeWeb macroregion")
    return data


def project(lon: float, lat: float) -> tuple[float, float]:
    """Spherical Lambert azimuthal equal-area projection, coordinates in km."""
    longitude, latitude = math.radians(lon - 10.5), math.radians(lat)
    center = math.radians(51)
    k = math.sqrt(
        2
        / (
            1
            + math.sin(center) * math.sin(latitude)
            + math.cos(center) * math.cos(latitude) * math.cos(longitude)
        )
    )
    return (
        6371 * k * math.cos(latitude) * math.sin(longitude),
        6371
        * k
        * (
            math.cos(center) * math.sin(latitude)
            - math.sin(center) * math.cos(latitude) * math.cos(longitude)
        ),
    )


def _polygons(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError("Only polygon state boundaries are supported")


def _geometry_path(geometry):
    x, y, area = [], [], 0.0
    for polygon in _polygons(geometry):
        for index, ring in enumerate(polygon):
            coordinates = [project(lon, lat) for lon, lat in ring]
            signed_area = (
                sum(
                    a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(coordinates, coordinates[1:])
                )
                / 2
            )
            # Opposite winding keeps enclaves/holes distinct in the SVG compound path.
            if (index == 0 and signed_area < 0) or (index > 0 and signed_area > 0):
                coordinates.reverse()
            if index == 0:
                area += abs(signed_area)
            x.extend([point[0] for point in coordinates] + [None])
            y.extend([point[1] for point in coordinates] + [None])
    return x, y, area


def _usable(row):
    if (
        not row
        or row.get("freshness") != "fresh"
        or row.get("level") not in ("low", "moderate", "high")
    ):
        return False
    try:
        quantiles = [float(row[key]) for key in ("q10", "q50", "q90")]
        start, end = (
            date.fromisoformat(str(row["week_start"])),
            date.fromisoformat(str(row["week_end"])),
        )
        return (
            all(math.isfinite(q) and q >= 0 for q in quantiles)
            and quantiles == sorted(quantiles)
            and start.weekday() == 0
            and end == start + timedelta(days=6)
        )
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def build_germany_map(rows: list[dict]) -> go.Figure:
    """Render the Germany outline from one regional row per displayed target week.

    Required row fields: region_id, level, q10/q50/q90, week_start/week_end,
    freshness. Missing, stale, malformed or duplicate regional rows render grey.
    """
    regional = {}
    for row in rows:
        region = row.get("region_id")
        if region in REGION_NAMES:
            regional[region] = row if region not in regional else None
    states = []
    all_x, all_y = [], []
    for feature in load_boundaries()["features"]:
        x, y, area = _geometry_path(feature["geometry"])
        states.append((area, feature, x, y))
        all_x.extend(value for value in x if value is not None)
        all_y.extend(value for value in y if value is not None)
    figure = go.Figure()
    # Small city states are drawn last, making Berlin, Hamburg and Bremen visible.
    for _, feature, x, y in sorted(states, key=lambda state: state[0], reverse=True):
        properties = feature["properties"]
        region = properties["region_id"]
        row = regional.get(region)
        usable = _usable(row)
        level = row["level"] if usable else "unknown"
        description = f"<b>{escape(properties['NUTS_NAME'])}</b><br>GrippeWeb region: {REGION_NAMES[region]}<br>"
        if usable:
            description += f"<b>{LEVEL_LABELS[level]} regional activity</b><br>Weekly ARE forecast: {float(row['q50']):,.0f} per 100,000<br>10th–90th percentiles: {float(row['q10']):,.0f}–{float(row['q90']):,.0f}<br>Week: {escape(str(row['week_start']))} – {escape(str(row['week_end']))}<br>Same regional estimate across these states"
        else:
            description += "<b>No fresh regional forecast</b><br>Grey means unavailable, not low activity"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=COLORS[level],
                line={"color": "#FFFFFF", "width": 1.1},
                name=properties["NUTS_NAME"],
                showlegend=False,
                hoveron="fills",
                hovertemplate=description + "<extra></extra>",
                meta={
                    "kind": "state",
                    "state_id": properties["NUTS_ID"],
                    "region_id": region,
                    "level": level,
                    "forecast_geography": "macroregion",
                },
            )
        )
    for level, label in LEVEL_LABELS.items():
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": COLORS[level], "size": 12, "symbol": "square"},
                name=label,
                legendgroup=level,
                showlegend=True,
                hoverinfo="skip",
                meta={"kind": "legend", "level": level},
            )
        )
    label_x, label_y, labels = [], [], []
    for region, point in LABEL_POINTS.items():
        x, y = project(*point)
        label_x.append(x)
        label_y.append(y)
        labels.append(REGION_NAMES[region])
    figure.add_trace(
        go.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            text=labels,
            textfont={"size": 12, "color": "#243447"},
            hoverinfo="skip",
            showlegend=False,
            meta={"kind": "region_labels"},
        )
    )
    figure.update_layout(
        height=560,
        margin={"l": 5, "r": 5, "t": 5, "b": 75},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": "#334155"},
        xaxis={
            "visible": False,
            "range": [min(all_x) - 25, max(all_x) + 25],
            "fixedrange": True,
        },
        yaxis={
            "visible": False,
            "range": [min(all_y) - 25, max(all_y) + 25],
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": True,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.015,
            "yanchor": "top",
            "font": {"size": 11},
            "itemclick": False,
            "itemdoubleclick": False,
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": "#1F2937", "size": 12}},
        annotations=[
            {
                "text": "Regional ARE outlook · state borders for orientation",
                "x": 0.5,
                "y": -0.105,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 10, "color": "#64748B"},
            },
            {
                "text": COPYRIGHT,
                "x": 0.5,
                "y": -0.15,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 9, "color": "#64748B"},
            },
        ],
        meta={
            "geographic_resolution": "four GrippeWeb macroregions",
            "boundary_source": "Eurostat/GISCO NUTS 2024",
            "external_tiles": False,
        },
        uirevision="germany-macroregions-v1",
    )
    return figure


def build_state_observation_map(rows: list[dict], indicator: str) -> go.Figure:
    """Map genuine state measurements, never disaggregate the regional forecast."""
    names = {"influenza": "Influenza", "rsv": "RSV"}
    if indicator not in names:
        raise ValueError("State observation map supports influenza and RSV")
    selected = {}
    for row in rows:
        if row.get("indicator") != indicator:
            continue
        code = row.get("location_id")
        selected[code] = row if code not in selected else None

    def usable(row):
        if (
            not row
            or row.get("freshness") != "fresh"
            or row.get("geographic_level") != "state"
        ):
            return False
        try:
            value = float(row["value"])
            start = date.fromisoformat(row["week_start"])
            end = date.fromisoformat(row["observed_through"])
            return (
                math.isfinite(value)
                and value >= 0
                and start.weekday() == 0
                and end == start + timedelta(days=6)
            )
        except (TypeError, ValueError, KeyError):
            return False

    # Different report periods must not silently share one comparison map.
    periods = {row["week_start"] for row in selected.values() if usable(row)}
    if len(periods) > 1:
        selected = {}
    values = [float(row["value"]) for row in selected.values() if usable(row)]
    upper = max(1.0, max(values, default=0))
    figure = build_germany_map([])
    figure.data = tuple(trace for trace in figure.data if trace.meta["kind"] == "state")
    codes = {
        f["properties"]["NUTS_ID"]: f["properties"]["state_code"]
        for f in load_boundaries()["features"]
    }
    for trace in figure.data:
        state = codes[trace.meta["state_id"]]
        row = selected.get(state)
        valid = usable(row)
        value = float(row["value"]) if valid else None
        trace.fillcolor = (
            sample_colorscale("OrRd", [value / upper])[0]
            if valid
            else COLORS["unknown"]
        )
        trace.meta = {
            "kind": "state",
            "state_id": trace.meta["state_id"],
            "location_id": state,
            "geographic_resolution": "state",
            "indicator": indicator,
            "value": value,
            "observation": True,
        }
        text = f"<b>{escape(trace.name)}</b><br>{names[indicator]} · reported observations<br>"
        if valid:
            text += f"<b>{value:.2f} reported cases per 100,000</b><br>Week: {escape(row['week_start'])} – {escape(row['observed_through'])}<br>State-level figure; not a forecast or infection probability"
        else:
            text += "No current comparable state report<br>Grey means unavailable, not zero cases"
        trace.hovertemplate = text + "<extra></extra>"
    if values:
        figure.add_trace(
            go.Scatter(
                x=[None, None],
                y=[None, None],
                mode="markers",
                showlegend=False,
                marker={
                    "color": [0, upper],
                    "colorscale": "OrRd",
                    "cmin": 0,
                    "cmax": upper,
                    "showscale": True,
                    "colorbar": {
                        "title": {"text": "Reported cases<br>per 100,000"},
                        "thickness": 12,
                        "len": 0.65,
                    },
                },
                hoverinfo="skip",
                meta={"kind": "scale", "units": "reported cases per 100000 per week"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"color": COLORS["unknown"], "size": 12, "symbol": "square"},
            name="No current report",
            hoverinfo="skip",
            meta={"kind": "legend"},
        )
    )
    figure.layout.annotations = (
        {
            "text": f"{names[indicator]} · state observations, not forecasts",
            "x": 0.5,
            "y": -0.105,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 10, "color": "#64748B"},
        },
        {
            "text": COPYRIGHT,
            "x": 0.5,
            "y": -0.15,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 9, "color": "#64748B"},
        },
    )
    figure.update_layout(
        meta={
            "geographic_resolution": "16 federal states",
            "kind": "observations",
            "indicator": indicator,
            "external_tiles": False,
        },
        uirevision=f"germany-state-{indicator}-v1",
    )
    return figure
