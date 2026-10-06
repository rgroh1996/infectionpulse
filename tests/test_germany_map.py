import hashlib
import json
from collections import Counter

import pytest

from infectionpulse.germany_map import (
    BOUNDARIES,
    COLORS,
    COPYRIGHT,
    EXPECTED_STATES,
    LEVEL_LABELS,
    REGION_NAMES,
    build_germany_map,
    build_state_observation_map,
    load_boundaries,
)


def row(region_id="east", **changes):
    return {
        "region_id": region_id,
        "level": "high",
        "q10": 3000,
        "q50": 5000,
        "q90": 8000,
        "week_start": "2026-09-21",
        "week_end": "2026-09-27",
        "freshness": "fresh",
        **changes,
    }


def states(figure):
    return [trace for trace in figure.data if trace.meta["kind"] == "state"]


def test_bundle_contains_real_german_states_with_verified_provenance():
    features = load_boundaries()["features"]
    assert len(features) == 16
    assert {feature["properties"]["NUTS_ID"] for feature in features} == EXPECTED_STATES
    assert {feature["properties"]["CNTR_CODE"] for feature in features} == {"DE"}
    assert Counter(feature["properties"]["region_id"] for feature in features) == {
        "north_west": 4,
        "central_west": 4,
        "east": 6,
        "south": 2,
    }
    # Check membership independently against the official state-code allocation.
    state_groups = {
        "north_west": {"01", "02", "03", "04"},
        "central_west": {"05", "06", "07", "10"},
        "east": {"11", "12", "13", "14", "15", "16"},
        "south": {"08", "09"},
    }
    for feature in features:
        properties = feature["properties"]
        assert properties["state_code"] in state_groups[properties["region_id"]]
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        assert feature["geometry"]["coordinates"]
    metadata = json.loads(BOUNDARIES.with_suffix(".sources.json").read_text())
    assert (
        hashlib.sha256(BOUNDARIES.read_bytes()).hexdigest() == metadata["bundle_sha256"]
    )
    assert metadata["copyright_notice"] == COPYRIGHT
    assert metadata["source_url"].startswith("https://gisco-services.ec.europa.eu/")


def test_each_state_uses_its_macroregion_forecast_and_explicit_hover():
    levels = dict(zip(REGION_NAMES, ["low", "moderate", "high", "low"]))
    figure = build_germany_map(
        [row(region, level=level) for region, level in levels.items()]
    )
    traces = states(figure)
    assert {trace.meta["state_id"] for trace in traces} == EXPECTED_STATES
    for trace in traces:
        region = trace.meta["region_id"]
        assert trace.fillcolor == COLORS[levels[region]]
        assert trace.meta["forecast_geography"] == "macroregion"
        assert REGION_NAMES[region] in trace.hovertemplate
        assert "5,000 per 100,000" in trace.hovertemplate
        assert "2026-09-21" in trace.hovertemplate
        assert "2026-09-27" in trace.hovertemplate
        assert "Same regional estimate" in trace.hovertemplate


@pytest.mark.parametrize(
    "changes",
    [
        {"freshness": "stale"},
        {"freshness": "unknown"},
        {"level": "unknown"},
        {"q50": None},
        {"q90": float("nan")},
        {"q10": -1},
        {"q90": 4000},
        {"week_start": "2026-09-22"},
        {"week_end": "2026-09-28"},
    ],
)
def test_stale_or_invalid_forecast_is_grey_never_low(changes):
    for trace in states(build_germany_map([row(**changes)])):
        assert trace.fillcolor == COLORS["unknown"]
        assert trace.meta["level"] == "unknown"
        assert "Grey means unavailable, not low activity" in trace.hovertemplate
        assert "Weekly ARE forecast" not in trace.hovertemplate


def test_missing_and_duplicate_regions_fail_closed():
    for rows in ([], [row(), row()]):
        assert all(
            trace.fillcolor == COLORS["unknown"]
            for trace in states(build_germany_map(rows))
        )


def test_complete_legend_copyright_and_no_network_map_layers():
    figure = build_germany_map([])
    legend = [trace for trace in figure.data if trace.showlegend]
    assert [trace.name for trace in legend] == list(LEVEL_LABELS.values())
    assert [trace.marker.color for trace in legend] == list(COLORS.values())
    assert all(trace.type == "scatter" for trace in figure.data)
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.meta["external_tiles"] is False
    assert COPYRIGHT in [annotation.text for annotation in figure.layout.annotations]
    serialized = figure.to_json()
    assert "https://" not in serialized
    assert "mapbox" not in figure.layout.to_plotly_json()
    assert "geo" not in figure.layout.to_plotly_json()


def observed(state="11", **changes):
    return {
        "indicator": "influenza",
        "location_id": state,
        "geographic_level": "state",
        "value": 0.2,
        "freshness": "fresh",
        "week_start": "2026-09-07",
        "observed_through": "2026-09-13",
        **changes,
    }


def test_state_map_uses_distinct_measurements_within_same_macroregion():
    figure = build_state_observation_map(
        [observed("11", value=0.2), observed("12", value=0.8)], "influenza"
    )
    by_state = {t.meta["location_id"]: t for t in states(figure)}
    assert len(by_state) == 16
    assert by_state["11"].fillcolor != by_state["12"].fillcolor
    assert by_state["11"].meta["value"] == 0.2
    assert by_state["12"].meta["value"] == 0.8
    assert "0.20 reported cases per 100,000" in by_state["11"].hovertemplate
    assert "not a forecast" in by_state["11"].hovertemplate
    assert by_state["09"].fillcolor == COLORS["unknown"]
    assert figure.layout.meta["geographic_resolution"] == "16 federal states"
    assert all(t.type == "scatter" for t in figure.data)
    assert COPYRIGHT in [a.text for a in figure.layout.annotations]


@pytest.mark.parametrize(
    "changes",
    [
        {"freshness": "stale"},
        {"value": float("nan")},
        {"value": -1},
        {"value": None},
        {"geographic_level": "macroregion"},
        {"observed_through": "2026-09-14"},
    ],
)
def test_invalid_state_reports_are_grey(changes):
    figure = build_state_observation_map([observed(**changes)], "influenza")
    assert all(t.fillcolor == COLORS["unknown"] for t in states(figure))
    assert all(t.meta["value"] is None for t in states(figure))


def test_state_map_does_not_mix_weeks_or_duplicate_measurements():
    for rows in (
        [observed(), observed()],
        [
            observed(),
            observed("12", week_start="2026-08-31", observed_through="2026-09-06"),
        ],
    ):
        assert all(
            t.meta["value"] is None
            for t in states(build_state_observation_map(rows, "influenza"))
        )
    zero = next(
        t
        for t in states(build_state_observation_map([observed(value=0)], "influenza"))
        if t.meta["location_id"] == "11"
    )
    assert zero.meta["value"] == 0 and zero.fillcolor != COLORS["unknown"]
    with pytest.raises(ValueError):
        build_state_observation_map([], "are")
