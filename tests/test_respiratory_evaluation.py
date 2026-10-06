import numpy as np
import pandas as pd
import pytest

from infectionpulse.respiratory_evaluation import (
    LOCKED_TARGETS,
    SeasonalReference,
    baseline_predictions,
    completed_job,
    paired_block_bootstrap,
    probability_readiness,
    select_context,
    tune_tree,
)


def history():
    dates = pd.date_range("2012-01-02", "2024-01-01", freq="W-MON")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "target_start": dates,
                    "region_id": region,
                    "issue_at": dates.tz_localize("UTC") - pd.Timedelta(days=3),
                    "truth_available_at": dates.tz_localize("UTC")
                    + pd.Timedelta(days=41),
                    "target_incidence": np.arange(len(dates)) + 1,
                    "incidence": np.arange(len(dates)) + 1,
                }
            )
            for region in ["a", "b", "c", "d"]
        ],
        ignore_index=True,
    )


def test_locked_calendar_and_nested_purged_contexts():
    assert len(LOCKED_TARGETS) == 104
    h = history()
    small, cal, pool = select_context(h, 25)
    large, _, _ = select_context(h, 1000)
    assert cal.target_start.nunique() == 52
    assert small.truth_available_at.max() <= cal.issue_at.min()

    def keys(f):
        return set(zip(f.target_start, f.region_id))

    assert keys(small) <= keys(large)
    assert keys(pool).isdisjoint(keys(cal))


def test_reference_does_not_depend_on_later_outcomes():
    h = history().rename(columns={"target_start": "week_start"})
    reference = SeasonalReference(h)
    altered = h.copy()
    altered.loc[altered.week_start > "2022-08-28", "incidence"] = 999999
    assert reference.annual == SeasonalReference(altered).annual
    assert reference.seasonal == SeasonalReference(altered).seasonal


def test_baseline_cannot_read_test_labels():
    h = history()
    reference = SeasonalReference(h.rename(columns={"target_start": "week_start"}))
    _, cal, _ = select_context(h, 25)
    test = h.tail(8).copy()
    raw = baseline_predictions("Persistence", cal, test, reference)
    test["target_incidence"] = 99999
    pd.testing.assert_frame_equal(
        raw, baseline_predictions("Persistence", cal, test, reference)
    )


def test_paired_bootstrap_preserves_pairing_and_missing_models():
    h = history().tail(52).copy()
    h["target_incidence"] = 10.0
    tab = h.assign(model="TabPFN-3.5", q50=9.0)
    other = h.assign(model="Persistence", q50=7.0)
    result = paired_block_bootstrap(
        pd.concat([tab, other]), "Persistence", n_bootstrap=50
    )
    assert result["mae_improvement"] == 2.0
    np.testing.assert_allclose(result["ci95"], [2, 2])
    assert paired_block_bootstrap(tab, "missing")["paired_rows"] == 0


def test_locked_outcomes_cannot_enter_tuning():
    h = history().tail(10).copy()
    h["target_start"] = pd.Timestamp("2025-01-06")
    with pytest.raises(ValueError, match="Locked"):
        tune_tree("LightGBM", [(h, h)], ["incidence"])


def test_probability_release_gate_is_measured_not_hardcoded():
    labels = np.tile([0, 1], 50)
    frame = pd.DataFrame(
        {
            "model": "TabPFN-3.5",
            "target_incidence": labels * 20.0,
            "seasonal_reference_p80": 10.0,
            "q10": 0.0,
            "q50": labels * 20.0,
            "q90": 25.0,
            "interval_low": 0.0,
            "interval_high": 25.0,
            "probability": labels * 0.98 + 0.01,
            "probability_raw": labels * 0.98 + 0.01,
            "climatology_probability": 0.5,
        }
    )
    assert probability_readiness(frame)["ready"]
    frame["probability"] = 0.5
    assert not probability_readiness(frame)["ready"]


def test_missing_prediction_artifact_never_counts_as_free_resume(tmp_path):
    (tmp_path / "result.json").write_text(
        '{"model":"TabPFN-3.5","query_rows":212,"calibration_rows":208}'
    )
    assert not completed_job(tmp_path)
