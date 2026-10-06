"""Experimental state notification forecasts with actual publication vintages.

Targets are weekly reported flu/RSV incidence, never personal infection odds.
Training histories are reconstructed within each issue-time snapshot; test
features use the actual historical release. Missing weeks remain missing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from infectionpulse.pathogen_data import (
    SOURCES,
    STATE_CODES,
    _read_snapshot,
    _session,
    _utc,
    _write_json,
    parse_pathogen_snapshot,
)

FEATURES = [
    "incidence",
    "inc_lag_1",
    "inc_lag_2",
    "inc_lag_3",
    "inc_lag_4",
    "inc_lag_52",
    "latest_change",
    "sin_week",
    "cos_week",
    "forecast_steps",
] + [f"state_{code}" for code in STATE_CODES]
DEVELOPMENT = ["2024-11-04", "2025-01-06", "2025-03-03", "2025-05-05"]
TARGETS = [
    "2025-08-04",
    "2025-09-01",
    "2025-10-06",
    "2025-11-03",
    "2025-12-01",
    "2026-01-05",
    "2026-02-02",
    "2026-03-02",
    "2026-04-06",
    "2026-05-04",
    "2026-06-01",
    "2026-07-06",
]
MODEL_NAMES = [
    "Persistence",
    "Seasonal naive",
    "LightGBM",
    "LightGBM full-history",
    "TabPFN-3.5",
]
CONFIGS = [
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 240, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 240, "learning_rate": 0.03, "max_depth": 3},
]


def protocol():
    return {
        "schema_version": 1,
        "status": "frozen_before_state_model_comparisons",
        "primary_indicator": "influenza",
        "secondary_indicator": "rsv",
        "target": "all-age state notification-week incidence per 100000",
        "geography": "16 German states; no individual infection outcome",
        "development_targets": DEVELOPMENT,
        "test_targets": TARGETS,
        "sampling": "first Monday of each month; 12 origins, not a weekly backtest",
        "issue": "Friday 10:00 Europe/Berlin before target Monday",
        "max_publication_age_days": 10,
        "allowed_observation_to_target_steps": [2, 3],
        "training_label_maturity_days": 35,
        "context_rows": 2000,
        "seed": 42,
        "features": FEATURES,
        "models": MODEL_NAMES,
        "tree_configs": CONFIGS,
        "tuning": "minimum pooled development MAE, separately by context regime and disease",
        "primary_metric": "MAE, separately by disease on common state-target rows",
        "secondary_metrics": ["RMSE", "q10-q90 coverage", "mean interval width"],
        "uncertainty": "descriptive pilot only; 12 correlated seasonal origins, no superiority significance claim",
        "rsv_revision_break": "2026-02-23",
        "rsv_reporting": "report all test rows and separate pre/post correction issue regimes",
        "deployment": "research artifacts only; does not drive the personal precaution score",
        "limitations": [
            "Issue-time training histories include revisions already present in that release, not first-report historical features.",
            "Mature labels are pinned to a later snapshot; this evaluates eventual notifications, not all infections.",
            "RSV releases omitted infections March 2025-February 2026; RKI corrected history on 23 February 2026.",
            "Raw predictive intervals have no coverage guarantee or public probability-readiness claim.",
            "Missing incidence is never converted to zero; seasonal-naive missing values use explicit persistence fallback.",
        ],
    }


def freeze_protocol(path):
    path = Path(path)
    expected = protocol()
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["protocol"] != expected:
            raise ValueError(
                "State protocol changed; use a new version/output directory"
            )
        return saved
    saved = {"frozen_at": pd.Timestamp.now(tz="UTC").isoformat(), "protocol": expected}
    _write_json(path, saved)
    return saved


def issue_for_target(target):
    target = pd.Timestamp(target)
    if target.weekday() != 0 or target.tzinfo is not None:
        raise ValueError("Target must be a timezone-naive Monday")
    return (
        (target - pd.Timedelta(days=3) + pd.Timedelta(hours=10))
        .tz_localize("Europe/Berlin")
        .tz_convert("UTC")
    )


def fetch_vintage(root, indicator, issue, *, download=False):
    """Pin the most recent data-file commit at/before the issue; no latest fallback."""
    root, issue = Path(root), _utc(issue)
    key = issue.strftime("%Y%m%dT%H%M%SZ")
    index = root / indicator / "issues" / f"{key}.json"
    if index.exists():
        metadata = json.loads(index.read_text())
        if metadata.get("unavailable"):
            return None, metadata
        if _utc(metadata["published_at"]) > issue:
            raise ValueError("Future snapshot in issue index")
        return parse_pathogen_snapshot(_read_snapshot(root, metadata)), metadata
    if not download:
        raise FileNotFoundError(
            f"Missing pinned {indicator} vintage for {issue}; prepare --download"
        )
    spec, session = SOURCES[indicator], _session()
    response = session.get(
        f"https://api.github.com/repos/{spec['repository']}/commits",
        params={"path": spec["filename"], "until": issue.isoformat(), "per_page": 1},
        timeout=30,
    )
    response.raise_for_status()
    commits = response.json()
    if not isinstance(commits, list):
        raise ValueError("Invalid RKI commit inventory")
    if not commits:
        metadata = {"unavailable": True, "issue_at": issue.isoformat()}
        _write_json(index, metadata)
        return None, metadata
    commit = commits[0]
    sha = commit["sha"]
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid commit SHA")
    published = _utc(commit["commit"]["committer"]["date"])
    if published > issue:
        raise ValueError("RKI returned a future release")
    relative = Path(indicator) / "snapshots" / f"{sha}.tsv.gz"
    raw_path = root / relative
    url = f"https://raw.githubusercontent.com/{spec['repository']}/{sha}/{spec['filename']}"
    if raw_path.exists():
        content = gzip.decompress(raw_path.read_bytes())
    else:
        raw = session.get(url, timeout=60)
        raw.raise_for_status()
        content = raw.content
    parsed = parse_pathogen_snapshot(content)
    metadata = {
        "indicator": indicator,
        "issue_at": issue.isoformat(),
        "source_snapshot_id": sha,
        "published_at": published.isoformat(),
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_file": str(relative),
        "source_url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "attribution": "Robert Koch-Institut",
        "license": "CC-BY-4.0",
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raw_path.write_bytes(gzip.compress(content, mtime=0))
    _write_json(index, metadata)
    return parsed, metadata


def feature_rows(observations, steps):
    """Exact calendar joins; a missing week must not shift every subsequent lag."""
    if steps not in (2, 3):
        raise ValueError("Only validated two/three observation-step designs supported")
    data = observations.copy()
    if data.duplicated(["location_id", "week_start"]).any():
        raise ValueError("Duplicate state/week")
    if not data.week_start.dt.weekday.eq(0).all():
        raise ValueError("Observation weeks must begin Monday")
    out = data[["location_id", "state_name", "week_start", "value"]].rename(
        columns={"week_start": "latest_observed_week", "value": "incidence"}
    )
    for lag in (1, 2, 3, 4, 52):
        lookup = data[["location_id", "week_start", "value"]].copy()
        lookup["week_start"] += pd.Timedelta(weeks=lag)
        lookup = lookup.rename(
            columns={"week_start": "latest_observed_week", "value": f"inc_lag_{lag}"}
        )
        out = out.merge(
            lookup,
            on=["location_id", "latest_observed_week"],
            how="left",
            validate="one_to_one",
        )
    out["target_start"] = out.latest_observed_week + pd.Timedelta(weeks=steps)
    target = data[["location_id", "week_start", "value"]].rename(
        columns={"week_start": "target_start", "value": "target_incidence"}
    )
    out = out.merge(
        target, on=["location_id", "target_start"], how="left", validate="one_to_one"
    )
    seasonal = data[["location_id", "week_start", "value"]].copy()
    seasonal["week_start"] += pd.Timedelta(weeks=52)
    seasonal = seasonal.rename(
        columns={"week_start": "target_start", "value": "seasonal_naive_incidence"}
    )
    out = out.merge(
        seasonal, on=["location_id", "target_start"], how="left", validate="one_to_one"
    )
    out["latest_change"] = out.incidence - out.inc_lag_1
    weeks = out.target_start.dt.isocalendar().week.astype(float)
    out["sin_week"], out["cos_week"] = (
        np.sin(2 * np.pi * weeks / 52.1775),
        np.cos(2 * np.pi * weeks / 52.1775),
    )
    out["forecast_steps"] = steps
    for code in STATE_CODES:
        out[f"state_{code}"] = out.location_id.eq(code).astype(float)
    return out.sort_values(["target_start", "location_id"]).reset_index(drop=True)


def build_job(observations, provenance, target, *, issue=None):
    target = pd.Timestamp(target)
    issue = _utc(issue) if issue is not None else issue_for_target(target)
    if target.tzinfo is not None or target.weekday() != 0:
        raise ValueError("Target must be a timezone-naive Monday")
    if issue >= target.tz_localize("Europe/Berlin").tz_convert("UTC"):
        raise ValueError("Issue must precede target")
    if provenance.get("unavailable"):
        return {"status": "no_historical_release", "target": str(target.date())}
    published = _utc(provenance["published_at"])
    if published > issue:
        raise ValueError("Feature release is after issue")
    if issue - published > pd.Timedelta(days=10):
        return {"status": "stale_release", "target": str(target.date())}
    day = issue.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    complete = observations[observations.observed_through < day].copy()
    if complete.empty:
        return {"status": "no_complete_week", "target": str(target.date())}
    latest = complete.week_start.max()
    steps = int((target - latest).days // 7)
    if steps not in (2, 3):
        return {
            "status": "unsupported_reporting_lag",
            "target": str(target.date()),
            "steps": steps,
        }
    rows = feature_rows(complete, steps)
    query = rows[rows.target_start.eq(target) & rows.incidence.notna()].copy()
    # Query target labels must never accompany hosted test features.
    query["target_incidence"] = np.nan
    pool = rows[
        rows.incidence.notna()
        & rows.target_incidence.notna()
        & (rows.target_start + pd.Timedelta(days=6) <= day - pd.Timedelta(days=35))
    ].copy()
    if len(pool) < 100 or query.empty:
        return {"status": "insufficient_history_or_query", "target": str(target.date())}
    order = np.random.default_rng(42).permutation(len(pool))[:2000]
    context = (
        pool.iloc[order]
        .sort_values(["target_start", "location_id"])
        .reset_index(drop=True)
    )
    return {
        "status": "ready",
        "target": str(target.date()),
        "issue_at": issue.isoformat(),
        "provenance": provenance,
        "steps": steps,
        "training_provenance": "reconstructed_within_actual_issue_snapshot",
        "query": query.reset_index(drop=True),
        "context": context,
        "pool": pool.reset_index(drop=True),
    }


def tree_tuning(jobs, *, full_history=False):
    """Fit medians/model only on each development origin's purged history."""
    from infectionpulse.respiratory_model import feature_values, make_tree

    for job in jobs:
        if job["target"] not in DEVELOPMENT:
            raise ValueError(
                "Tree tuning requires development targets; test labels forbidden"
            )
        if (
            job["status"] == "ready"
            and not job["query"].target_start.eq(pd.Timestamp(job["target"])).all()
        ):
            raise ValueError(
                "Development query targets do not match the declared origin"
            )
    scores = []
    for config in CONFIGS:
        errors = []
        for job in jobs:
            if job["status"] != "ready" or job["query"].target_incidence.isna().all():
                continue
            train = job["pool"] if full_history else job["context"]
            test = job["query"].dropna(subset=["target_incidence"])
            x = feature_values(train, FEATURES)
            medians = x.median().fillna(0)
            xt = feature_values(test, FEATURES)
            y = np.log1p(train.target_incidence) - np.log1p(train.incidence)
            model = make_tree("LightGBM", 0.5, config).fit(
                np.column_stack([x.fillna(medians), x.isna().astype(float)]), y
            )
            pred = np.maximum(
                0,
                np.expm1(
                    model.predict(
                        np.column_stack([xt.fillna(medians), xt.isna().astype(float)])
                    )
                    + np.log1p(test.incidence)
                ),
            )
            errors.extend(np.abs(pred - test.target_incidence).tolist())
        if not errors:
            raise ValueError("No development labels; refusing untuned benchmark")
        scores.append(
            {"config": config, "mae": float(np.mean(errors)), "rows": len(errors)}
        )
    return {
        "selected": min(scores, key=lambda row: row["mae"])["config"],
        "development_scores": scores,
    }


def run_model(job, name, config, folder):
    from infectionpulse.respiratory_model import RespiratoryModel

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    query = job["query"]
    if name in ("Persistence", "Seasonal naive"):
        point = (
            query.incidence
            if name == "Persistence"
            else query.seasonal_naive_incidence.fillna(query.incidence)
        )
        result = pd.DataFrame({"q50": point})
        # Honest point baselines: do not invent a zero-width uncertainty interval.
        result["q10"], result["q90"] = np.nan, np.nan
    else:
        train = job["pool"] if name.endswith("full-history") else job["context"]
        model = RespiratoryModel(name.replace(" full-history", ""), FEATURES, config)
        model.fit(train)
        result = model.raw(query, np.ones(len(query)), folder / "api_cache")[
            ["q10", "q50", "q90"]
        ]
        model.save(folder / "model.joblib")
        _write_json(
            folder / "model_metadata.json",
            {
                "model": name,
                "api_version": model.api_version,
                "training_rows": len(train),
                "features": FEATURES,
                "interval_status": "raw_uncalibrated",
                "personal_probability": None,
                "inference_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            },
        )
    result = result.reset_index(drop=True)
    for column in (
        "location_id",
        "state_name",
        "target_start",
        "latest_observed_week",
        "forecast_steps",
        "incidence",
        "seasonal_naive_incidence",
    ):
        result[column] = query[column].to_numpy()
    result["model"], result["issue_at"] = name, job["issue_at"]
    result["source_snapshot_id"] = job["provenance"]["source_snapshot_id"]
    result["rsv_revision_regime"] = (
        "post_correction"
        if _utc(job["issue_at"]) >= _utc("2026-02-23")
        else "pre_correction"
    )
    temp = folder / "predictions.tmp.parquet"
    result.to_parquet(temp, index=False)
    temp.replace(folder / "predictions.parquet")
    return result


def summarize(predictions, truth, expected_models):
    """Score only common finite rows; separate diseases and RSV revision regimes."""
    if not expected_models or len(set(expected_models)) != len(expected_models):
        raise ValueError("Expected a nonempty unique model list")
    if not set(predictions.model).issubset(expected_models):
        raise ValueError("Unexpected model in matched comparison")
    keys = ["indicator", "location_id", "target_start"]
    if predictions.duplicated([*keys, "model"]).any():
        raise ValueError("Duplicate model prediction")
    if truth.duplicated(keys).any():
        raise ValueError("Duplicate truth")
    scored = predictions.merge(
        truth[keys + ["target_incidence"]], on=keys, validate="many_to_one"
    )
    valid = scored[np.isfinite(scored.q50) & np.isfinite(scored.target_incidence)]
    counts = valid.groupby(keys).model.nunique()
    common = counts[counts.eq(len(expected_models))].reset_index()[keys]
    valid = valid.merge(common, on=keys, validate="many_to_one")
    rows = []
    for indicator, disease in valid.groupby("indicator"):
        regimes = ["all"] + (
            ["pre_correction", "post_correction"] if indicator == "rsv" else []
        )
        for regime in regimes:
            subset = (
                disease
                if regime == "all"
                else disease[disease.rsv_revision_regime.eq(regime)]
            )
            for name, group in subset.groupby("model"):
                error = group.q50 - group.target_incidence
                interval = (
                    np.isfinite(group.q10)
                    & np.isfinite(group.q90)
                    & (group.q10 >= 0)
                    & (group.q10 <= group.q50)
                    & (group.q50 <= group.q90)
                )
                bands = group[interval]
                rows.append(
                    {
                        "indicator": indicator,
                        "regime": regime,
                        "model": name,
                        "rows": len(group),
                        "origins": group.target_start.nunique(),
                        "mae": float(error.abs().mean()),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "interval_rows": int(interval.sum()),
                        "coverage_80": None
                        if bands.empty
                        else float(
                            (
                                (bands.q10 <= bands.target_incidence)
                                & (bands.target_incidence <= bands.q90)
                            ).mean()
                        ),
                        "mean_interval_width": None
                        if bands.empty
                        else float((bands.q90 - bands.q10).mean()),
                    }
                )
    return pd.DataFrame(rows), valid
