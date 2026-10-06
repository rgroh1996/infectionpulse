import copy

import pandas as pd
import pytest
from check_live_forecasts import assess_record, summarize


def fixture():
    forecast = {
        "forecast_id": "saved:east",
        "indicator": "are",
        "location_id": "east",
        "issued_at": "2026-09-17T10:00:00Z",
        "target_week_start": "2026-09-21",
        "target_week_end": "2026-09-27",
        "q10": 5,
        "q50": 10,
        "q90": 15,
        "reference_p80": 12,
    }
    observations = pd.DataFrame(
        {
            "location_id": ["east"],
            "week_start": pd.to_datetime(["2026-09-21"]),
            "value": [20.0],
        }
    )
    source = {
        "source_snapshot_id": "new_release",
        "published_at": "2026-10-01T08:00Z",
        "source_url": "https://example.test/release",
    }
    return forecast, "2026-09-17T10:01:00Z", observations, source, "2026-10-05T10:00Z"


def test_preliminary_score_preserves_saved_forecast_and_reports_missed_high_event():
    args = fixture()
    original = copy.deepcopy(args[0])
    row = assess_record(*args, persistence=7)
    assert args[0] == original
    assert row["status"] == "preliminary_latest_release"
    assert row["absolute_error"] == 10
    assert row["persistence_absolute_error"] == 13
    assert row["interval_covered"] is False
    assert row["predicted_high"] is False and row["observed_high"] is True
    assert row["mature_label_earliest_at"].startswith("2026-11-02")


@pytest.mark.parametrize("change", ["issued", "generated", "unknown"])
def test_backdated_or_unknown_generation_is_excluded(change):
    forecast, generated, observations, source, now = fixture()
    if change == "issued":
        forecast["issued_at"] = "2026-09-21T00:00:00Z"
    elif change == "generated":
        generated = "2026-09-21T00:00:00Z"
    else:
        generated = None
    row = assess_record(forecast, generated, observations, source, now)
    assert row["observed"] is None
    assert row["status"] in (
        "forecast_created_after_target_started",
        "generation_or_issue_time_unknown",
    )


def test_state_row_actual_generation_overrides_earlier_bundle_time():
    forecast, generated, observations, source, now = fixture()
    forecast["generated_at"] = "2026-09-22T10:00Z"
    row = assess_record(forecast, generated, observations, source, now)
    assert row["status"] == "forecast_created_after_target_started"


def test_future_outcome_release_is_rejected():
    forecast, generated, observations, source, _ = fixture()
    with pytest.raises(ValueError, match="evaluation cutoff"):
        assess_record(forecast, generated, observations, source, "2026-09-30T10:00Z")


def test_earlier_week_never_substitutes_for_unpublished_outcome():
    forecast, generated, observations, source, now = fixture()
    observations["week_start"] = pd.Timestamp("2026-09-14")
    row = assess_record(forecast, generated, observations, source, now)
    assert row["status"] == "pending_publication"
    assert row["observed"] is None and row["absolute_error"] is None


def test_missing_region_and_null_outcomes_are_not_zero():
    forecast, generated, observations, source, now = fixture()
    for frame in (
        observations.assign(location_id="south"),
        observations.assign(value=float("nan")),
    ):
        row = assess_record(forecast, generated, frame, source, now)
        assert row["status"] == "missing_region_outcome" and row["observed"] is None
    zero = assess_record(
        forecast, generated, observations.assign(value=0.0), source, now
    )
    assert zero["observed"] == 0 and zero["absolute_error"] == 10


def test_duplicate_outcomes_fail_instead_of_selecting_favorable_row():
    forecast, generated, observations, source, now = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        assess_record(
            forecast, generated, pd.concat([observations, observations]), source, now
        )


def test_unfinished_week_is_not_scored_even_if_source_has_a_value():
    forecast, generated, observations, source, _ = fixture()
    source["published_at"] = "2026-09-24T08:00Z"
    row = assess_record(forecast, generated, observations, source, "2026-09-27T20:00Z")
    assert row["status"] == "target_week_not_complete" and row["observed"] is None


def test_summary_uses_same_outcomes_for_persistence_comparison():
    first = assess_record(*fixture(), persistence=7)
    unpaired = dict(
        first,
        forecast_id="other",
        location_id="south",
        absolute_error=100,
        squared_error=10000,
        persistence=None,
        persistence_absolute_error=None,
    )
    pending = dict(
        first, indicator="influenza", status="pending_publication", observed=None
    )
    summary = summarize([first, unpaired, pending])
    are, flu = summary
    assert are["mae"] == 55 and are["paired_tabpfn_mae"] == 10
    assert are["persistence_mae"] == 13 and are["persistence_paired_rows"] == 1
    assert flu["mae"] is None and flu["observed_forecasts"] == 0


def test_invalid_forecast_interval_fails():
    args = list(fixture())
    args[0]["q90"] = 1
    with pytest.raises(ValueError, match="quantiles"):
        assess_record(*args)
