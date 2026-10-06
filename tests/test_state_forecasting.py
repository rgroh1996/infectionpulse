"""State-pilot temporal and coverage contracts; no hosted or native tree calls."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import state_benchmark as cli

import infectionpulse.respiratory_model as respiratory_model
import infectionpulse.state_forecasting as state


def observations(latest="2025-07-21", weeks=160, codes=("01", "02")):
    calendar = pd.date_range(end=latest, periods=weeks, freq="W-MON")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "location_id": code,
                    "state_name": f"State {code}",
                    "week_start": calendar,
                    "observed_through": calendar + pd.Timedelta(days=6),
                    "value": np.arange(weeks, dtype=float) + 1000 * int(code),
                    "cases": np.arange(weeks),
                }
            )
            for code in codes
        ],
        ignore_index=True,
    )


def provenance(target="2025-08-04"):
    return {
        "published_at": (
            state.issue_for_target(target) - pd.Timedelta(days=1)
        ).isoformat(),
        "source_snapshot_id": "a" * 40,
        "sha256": "b" * 64,
    }


def test_issue_time_respects_berlin_daylight_saving_transition():
    assert state.issue_for_target("2025-03-31") == pd.Timestamp("2025-03-28T09:00:00Z")
    assert state.issue_for_target("2025-04-07") == pd.Timestamp("2025-04-04T08:00:00Z")
    assert state.issue_for_target("2026-01-05") == pd.Timestamp("2026-01-02T09:00:00Z")
    with pytest.raises(ValueError, match="Monday"):
        state.issue_for_target("2025-08-05")
    with pytest.raises(ValueError, match="timezone-naive"):
        state.issue_for_target(pd.Timestamp("2025-08-04", tz="UTC"))


def test_calendar_lags_preserve_missing_week_and_state_boundaries():
    data = observations(weeks=60)
    missing = pd.Timestamp("2025-07-14")
    data = data[~(data.location_id.eq("01") & data.week_start.eq(missing))]
    rows = state.feature_rows(data, 2)
    latest = rows[rows.latest_observed_week.eq(pd.Timestamp("2025-07-21"))].set_index(
        "location_id"
    )
    assert pd.isna(latest.loc["01", "inc_lag_1"])
    assert latest.loc["01", "inc_lag_2"] == 1057
    assert latest.loc["02", "inc_lag_1"] == 2058
    assert latest.loc["01", "inc_lag_52"] == 1007
    assert latest.loc["01", "seasonal_naive_incidence"] == 1009
    assert latest.loc["01", "state_01"] == 1
    assert latest.loc["01", "state_02"] == 0
    assert pd.isna(latest.loc["01", "latest_change"])


@pytest.mark.parametrize("steps", [1, 4, 0, -1])
def test_feature_horizon_only_allows_two_or_three_calendar_steps(steps):
    with pytest.raises(ValueError, match="two/three"):
        state.feature_rows(observations(), steps)


def test_duplicate_observations_fail_before_training():
    data = observations()
    with pytest.raises(ValueError, match="Duplicate"):
        state.feature_rows(pd.concat([data, data.iloc[:1]]), 2)


@pytest.mark.parametrize("latest,steps", [("2025-07-21", 2), ("2025-07-14", 3)])
def test_job_supports_actual_reporting_gap_and_purges_training_labels(latest, steps):
    job = state.build_job(observations(latest), provenance(), "2025-08-04")
    assert job["status"] == "ready"
    assert job["steps"] == steps
    assert job["query"].forecast_steps.eq(steps).all()
    assert job["query"].target_start.eq(pd.Timestamp("2025-08-04")).all()
    assert job["query"].target_incidence.isna().all()
    day = pd.Timestamp("2025-08-01")
    assert (
        job["pool"].target_start + pd.Timedelta(days=6) <= day - pd.Timedelta(days=35)
    ).all()
    assert job["pool"].target_start.max() == pd.Timestamp("2025-06-16")
    assert not set(job["query"].target_start).intersection(job["pool"].target_start)


def test_unsupported_four_step_lag_abstains_instead_of_extrapolating():
    result = state.build_job(observations("2025-07-07"), provenance(), "2025-08-04")
    assert result["status"] == "unsupported_reporting_lag"
    assert result["steps"] == 4


def test_missing_current_state_does_not_reuse_older_week_as_current():
    data = observations()
    data = data[
        ~(data.location_id.eq("01") & data.week_start.eq(pd.Timestamp("2025-07-21")))
    ]
    job = state.build_job(data, provenance(), "2025-08-04")
    assert job["status"] == "ready"
    assert job["query"].location_id.tolist() == ["02"]


def test_missing_incidence_is_not_zero_and_missing_training_label_is_excluded():
    data = observations()
    data.loc[
        data.location_id.eq("01") & data.week_start.eq(pd.Timestamp("2025-07-21")),
        "value",
    ] = np.nan
    data.loc[
        data.location_id.eq("02") & data.week_start.eq(pd.Timestamp("2025-06-02")),
        "value",
    ] = np.nan
    job = state.build_job(data, provenance(), "2025-08-04")
    assert job["query"].location_id.tolist() == ["02"]
    assert not (
        job["pool"].location_id.eq("02")
        & job["pool"].target_start.eq(pd.Timestamp("2025-06-02"))
    ).any()
    assert job["pool"].target_incidence.notna().all()


def test_incomplete_current_week_and_future_labels_cannot_change_query_features():
    data = observations()
    baseline = state.build_job(data, provenance(), "2025-08-04")
    future = observations(latest="2025-08-04", weeks=2).assign(value=99999.0)
    altered = state.build_job(pd.concat([data, future]), provenance(), "2025-08-04")
    pd.testing.assert_frame_equal(baseline["query"], altered["query"])
    pd.testing.assert_frame_equal(baseline["context"], altered["context"])


def test_context_sampling_is_deterministic_bounded_and_from_purged_pool():
    data = observations(weeks=200, codes=state.STATE_CODES)
    a = state.build_job(data, provenance(), "2025-08-04")
    b = state.build_job(data, provenance(), "2025-08-04")
    assert len(a["context"]) == 2000
    assert len(a["pool"]) > 2000
    pd.testing.assert_frame_equal(a["context"], b["context"])
    keys = ["location_id", "target_start"]
    assert set(map(tuple, a["context"][keys].to_numpy())) <= set(
        map(tuple, a["pool"][keys].to_numpy())
    )


def test_future_publication_fails_and_stale_or_absent_release_abstains():
    issue = state.issue_for_target("2025-08-04")
    with pytest.raises(ValueError, match="after issue"):
        state.build_job(
            observations(),
            {"published_at": (issue + pd.Timedelta(seconds=1)).isoformat()},
            "2025-08-04",
        )
    result = state.build_job(
        observations(),
        {"published_at": (issue - pd.Timedelta(days=10, seconds=1)).isoformat()},
        "2025-08-04",
    )
    assert result["status"] == "stale_release"
    assert (
        state.build_job(None, {"unavailable": True}, "2025-08-04")["status"]
        == "no_historical_release"
    )


def test_issue_must_precede_target():
    with pytest.raises(ValueError, match="precede target"):
        state.build_job(
            observations(), provenance(), "2025-08-04", issue="2025-08-04T00:00:00Z"
        )


def test_cached_future_vintage_is_rejected_before_reading_payload(tmp_path):
    issue = state.issue_for_target("2025-08-04")
    path = tmp_path / "influenza/issues" / f"{issue.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"published_at": (issue + pd.Timedelta(seconds=1)).isoformat()})
    )
    with pytest.raises(ValueError, match="Future snapshot"):
        state.fetch_vintage(tmp_path, "influenza", issue)


def test_download_pins_query_to_issue_and_rejects_future_release(monkeypatch, tmp_path):
    issue = state.issue_for_target("2025-08-04")
    requests = []

    def get(url, **kwargs):
        requests.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [
                {
                    "sha": "a" * 40,
                    "commit": {
                        "committer": {
                            "date": (issue + pd.Timedelta(seconds=1)).isoformat()
                        }
                    },
                }
            ],
        )

    monkeypatch.setattr(state, "_session", lambda: SimpleNamespace(get=get))
    with pytest.raises(ValueError, match="future release"):
        state.fetch_vintage(tmp_path, "influenza", issue, download=True)
    assert len(requests) == 1
    assert requests[0][1]["params"]["until"] == issue.isoformat()
    assert requests[0][1]["params"]["path"] == "IfSG_Influenzafaelle.tsv"


def test_no_cached_vintage_does_not_fall_back_to_latest(tmp_path):
    (tmp_path / "influenza").mkdir()
    (tmp_path / "influenza/latest.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Missing pinned"):
        state.fetch_vintage(tmp_path, "influenza", "2025-08-01T08:00:00Z")


def score_data():
    truth = pd.DataFrame(
        {
            "indicator": ["influenza"] * 3,
            "location_id": ["01", "02", "03"],
            "target_start": pd.Timestamp("2025-08-04"),
            "target_incidence": [10.0, 20.0, 30.0],
        }
    )
    forecast = truth.drop(columns="target_incidence").assign(
        q50=[11.0, 21.0, 100.0],
        q10=[8.0, 18.0, 25.0],
        q90=[12.0, 22.0, 110.0],
        rsv_revision_regime="pre_correction",
    )
    predictions = pd.concat(
        [
            forecast.assign(model="TabPFN-3.5"),
            forecast.iloc[:2].assign(model="LightGBM full-history"),
        ],
        ignore_index=True,
    )
    return predictions, truth


def test_summary_scores_models_on_common_rows_not_their_own_coverage():
    predictions, truth = score_data()
    summary, scored = state.summarize(
        predictions, truth, ["TabPFN-3.5", "LightGBM full-history"]
    )
    assert len(scored) == 4
    assert summary.rows.tolist() == [2, 2]
    assert summary.mae.tolist() == [1.0, 1.0]
    assert summary.coverage_80.tolist() == [1.0, 1.0]


def test_nonfinite_forecast_and_missing_truth_drop_same_row_for_every_model():
    predictions, truth = score_data()
    predictions.loc[0, "q50"] = np.inf
    truth.loc[truth.location_id.eq("03"), "target_incidence"] = np.nan
    summary, scored = state.summarize(
        predictions, truth, ["TabPFN-3.5", "LightGBM full-history"]
    )
    assert summary.rows.tolist() == [1, 1]
    assert scored.location_id.eq("02").all()


def test_unexpected_model_cannot_substitute_missing_required_competitor():
    predictions, truth = score_data()
    predictions.loc[predictions.model.eq("LightGBM full-history"), "model"] = (
        "unexpected"
    )
    try:
        summary, scored = state.summarize(
            predictions, truth, ["TabPFN-3.5", "LightGBM full-history"]
        )
    except ValueError:
        return
    assert summary.empty and scored.empty


@pytest.mark.parametrize("low,high", [(-np.inf, np.inf), (20.0, 1.0)])
def test_invalid_intervals_do_not_count_as_valid_coverage(low, high):
    predictions, truth = score_data()
    predictions.loc[predictions.model.eq("TabPFN-3.5"), ["q10", "q90"]] = [low, high]
    try:
        summary, _ = state.summarize(
            predictions, truth, ["TabPFN-3.5", "LightGBM full-history"]
        )
    except ValueError:
        return
    tab = summary[summary.model.eq("TabPFN-3.5")].iloc[0]
    assert tab.interval_rows == 0
    assert pd.isna(tab.coverage_80)


def test_point_only_baselines_do_not_invent_intervals():
    predictions, truth = score_data()
    predictions.loc[predictions.model.eq("LightGBM full-history"), ["q10", "q90"]] = (
        np.nan
    )
    summary, _ = state.summarize(
        predictions, truth, ["TabPFN-3.5", "LightGBM full-history"]
    )
    baseline = summary[summary.model.eq("LightGBM full-history")].iloc[0]
    assert baseline.interval_rows == 0
    assert pd.isna(baseline.coverage_80)


def test_summary_rejects_duplicates_and_separates_diseases_and_rsv_regimes():
    predictions, truth = score_data()
    with pytest.raises(ValueError, match="Duplicate model"):
        state.summarize(
            pd.concat([predictions, predictions.iloc[:1]]),
            truth,
            ["TabPFN-3.5", "LightGBM full-history"],
        )
    rsv = predictions.assign(indicator="rsv")
    rsv.loc[rsv.location_id.eq("02"), "rsv_revision_regime"] = "post_correction"
    summary, _ = state.summarize(
        pd.concat([predictions, rsv]),
        pd.concat([truth, truth.assign(indicator="rsv")]),
        ["TabPFN-3.5", "LightGBM full-history"],
    )
    assert set(summary.indicator) == {"influenza", "rsv"}
    assert set(summary[summary.indicator.eq("rsv")].regime) == {
        "all",
        "pre_correction",
        "post_correction",
    }
    assert (
        summary[summary.indicator.eq("rsv") & summary.regime.ne("all")].rows.eq(1).all()
    )


def test_test_period_labels_cannot_enter_tree_tuning(monkeypatch):
    job = state.build_job(observations(), provenance(), "2025-08-04")
    job["query"]["target_incidence"] = 100.0
    fitted = []

    class FakeTree:
        def fit(self, x, y):
            fitted.append(len(y))
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(respiratory_model, "make_tree", lambda *args: FakeTree())
    with pytest.raises(ValueError, match="[Dd]evelopment|[Tt]est|[Ll]ocked"):
        state.tree_tuning([job])
    assert not fitted


def test_prepare_development_labels_use_pretest_vintage_not_latest_truth(
    monkeypatch, tmp_path
):
    """Exercise prepare orchestration with sentinel label values and no model calls."""
    root = tmp_path / "data"
    latest = {
        "published_at": "2026-09-17T07:00:00Z",
        "source_snapshot_id": "latest",
        "sha256": "latest",
    }
    for indicator in ("influenza", "rsv"):
        (root / indicator).mkdir(parents=True)
        (root / indicator / "latest.json").write_text(json.dumps(latest))
    future_truth = pd.DataFrame(
        {
            "location_id": ["01"] * 16,
            "state_name": "State",
            "week_start": pd.to_datetime(state.DEVELOPMENT + state.TARGETS),
            "observed_through": pd.to_datetime(state.DEVELOPMENT + state.TARGETS)
            + pd.Timedelta(days=6),
            "value": 999.0,
            "cases": 1,
        }
    )
    pretest_truth = future_truth.copy().assign(value=123.0)
    captured = []
    monkeypatch.setattr(cli, "DATA", root)
    monkeypatch.setattr(cli, "_read_snapshot", lambda *args: b"latest sentinel")
    monkeypatch.setattr(
        cli, "parse_pathogen_snapshot", lambda content: future_truth.copy()
    )

    def fetch(root, indicator, issue, **kwargs):
        return pretest_truth.copy(), {
            "published_at": issue.isoformat(),
            "source_snapshot_id": str(issue),
            "sha256": "past",
        }

    def build(data, metadata, target):
        return {
            "status": "ready",
            "target": target,
            "query": pd.DataFrame(
                {
                    "location_id": ["01"],
                    "target_start": pd.to_datetime([target]),
                    "target_incidence": np.nan,
                }
            ),
        }

    def tune(function, jobs, **kwargs):
        captured.extend(job["query"].target_incidence.tolist() for job in jobs)
        return {"selected": {}, "development_scores": []}

    monkeypatch.setattr(cli, "fetch_vintage", fetch)
    monkeypatch.setattr(cli, "build_job", build)
    monkeypatch.setattr(cli, "isolated", tune)
    args = SimpleNamespace(
        reports=tmp_path / "reports", output=tmp_path / "models", download=False
    )
    folder, plan = cli.prepare(args)
    assert captured and all(values == [123.0] for values in captured)
    for indicator in ("influenza", "rsv"):
        assert pd.Timestamp(
            plan["sources"][f"{indicator}/development_truth"]["published_at"]
        ) == state.issue_for_target(state.TARGETS[0])
    assert pd.read_parquet(folder / "truth.parquet").target_incidence.eq(999.0).all()
    assert all(
        job["query"].target_incidence.isna().all() for job in plan["jobs"].values()
    )
