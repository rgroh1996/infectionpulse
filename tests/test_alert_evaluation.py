import json

import numpy as np
import pandas as pd
import pytest

from infectionpulse.alert_evaluation import (
    counts,
    freeze_protocol,
    labels,
    matched_predictions,
    paired_intervals,
    scores,
)


def sample():
    data = pd.DataFrame(
        {
            "target_start": pd.date_range("2025-01-06", periods=4, freq="W-MON"),
            "region_id": "a",
            "eligible": True,
            "target_incidence": [20, 20, 0, 0],
            "reference_p80": 10,
            "seasonal_reference_p80": 10,
            "q50": [20, 0, 20, 0],
            "probability": [0.8, 0.2, 0.8, 0.2],
        }
    )
    frame = pd.concat(
        [data.assign(model=model) for model in ["TabPFN-3.5", "LightGBM full-history"]],
        ignore_index=True,
    )
    return frame, data[["target_start", "region_id", "eligible"]]


def test_confusion_counts_distinguish_false_alarm_denominators():
    result = scores(counts([1, 1, 0, 0, 0], [1, 0, 1, 0, 0]))
    assert result["alerts"] == 2
    assert result["missed_events"] == 1
    assert result["precision"] == result["recall"] == 0.5
    assert result["false_positive_rate"] == 1 / 3
    assert result["false_discovery_rate"] == 0.5
    assert result["false_alerts_per_100_region_weeks"] == 20


def test_undefined_metrics_are_not_fabricated_as_zero():
    result = scores([0, 0, 0, 4])
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["false_positive_rate"] == 0
    json.dumps(result, allow_nan=False)


def test_primary_boundary_matches_existing_policy_and_secondary_event_is_strict():
    frame, _ = sample()
    frame.loc[:, ["q50", "target_incidence"]] = 10
    frame.loc[:, "probability"] = 0.5
    event, alert = labels(frame, "high_burden")
    assert event.all() and alert.all()
    event, alert = labels(frame, "seasonal_event")
    assert not event.any() and alert.all()


def test_shared_coverage_excludes_missing_without_counting_negative():
    frame, coverage = sample()
    frame = frame.drop(index=7)
    matched, report = matched_predictions(
        frame, coverage, ["TabPFN-3.5", "LightGBM full-history"]
    )
    assert len(matched) == 6
    assert report.matched_region_weeks.tolist() == [3, 3]
    assert report.unavailable_region_weeks.tolist() == [0, 1]
    assert report.unmatched_usable_predictions.tolist() == [1, 0]


def test_invalid_probability_abstains_for_both_policy_comparisons():
    frame, coverage = sample()
    frame.loc[0, "probability"] = np.nan
    matched, report = matched_predictions(
        frame, coverage, ["TabPFN-3.5", "LightGBM full-history"]
    )
    assert len(matched) == 6
    assert report.unavailable_region_weeks.tolist() == [1, 0]


@pytest.mark.parametrize(
    "column", ["reference_p80", "seasonal_reference_p80", "target_incidence"]
)
def test_disagreeing_truth_or_reference_fails(column):
    frame, coverage = sample()
    frame.loc[0, column] += 1
    with pytest.raises(ValueError, match="disagree"):
        matched_predictions(frame, coverage, ["TabPFN-3.5", "LightGBM full-history"])


def test_duplicate_and_missing_model_fail():
    frame, coverage = sample()
    with pytest.raises(ValueError, match="Duplicate"):
        matched_predictions(
            pd.concat([frame, frame.iloc[:1]]),
            coverage,
            ["TabPFN-3.5", "LightGBM full-history"],
        )
    with pytest.raises(ValueError, match="Missing required"):
        matched_predictions(frame, coverage, ["TabPFN-3.5", "missing"])


def test_unplanned_forecasts_fail():
    frame, coverage = sample()
    with pytest.raises(ValueError, match="unplanned"):
        matched_predictions(
            frame, coverage.iloc[1:], ["TabPFN-3.5", "LightGBM full-history"]
        )


def test_identical_models_have_zero_paired_difference_even_with_missing_calendar_week():
    frame, coverage = sample()
    frame, _ = matched_predictions(
        frame, coverage, ["TabPFN-3.5", "LightGBM full-history"]
    )
    calendar = pd.date_range(frame.target_start.min(), periods=5, freq="W-MON")
    intervals = paired_intervals(
        frame, "high_burden", calendar, {"block_weeks": 4, "replicates": 30, "seed": 42}
    )
    assert all(item["difference"] == 0 for item in intervals)
    assert all(item["ci95"] == [0, 0] for item in intervals)


def test_freeze_is_immutable_and_discloses_prior_test_inspection(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ["predictions.parquet", "coverage_rows.parquet", "manifest.json"]:
        (run / name).write_bytes(b"input")
    target = tmp_path / "protocol.json"
    first = freeze_protocol(run, target, n_bootstrap=20)
    assert first["evidence_status"] == "exploratory_retrospective"
    assert "inspected" in first["disclosure"]
    assert freeze_protocol(run, target, n_bootstrap=20) == first
    with pytest.raises(ValueError, match="Frozen"):
        freeze_protocol(run, target, n_bootstrap=30)
    (run / "predictions.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen"):
        freeze_protocol(run, target, n_bootstrap=20)
