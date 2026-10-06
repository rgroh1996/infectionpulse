"""Publication-aware, paired respiratory evaluation; no hosted calls on import."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
)

from infectionpulse.respiratory_model import (
    SEARCH_CONFIGS,
    RespiratoryModel,
    feature_values,
    make_tree,
)

LOCKED_TARGETS = pd.date_range("2024-08-05", "2026-07-27", freq="W-MON")
DEVELOPMENT_TARGETS = pd.date_range("2022-09-05", "2024-07-29", freq="W-MON")
DEVELOPMENT_ORIGINS = ["2023-01-09", "2023-07-10"]
CONTEXT_LADDER = [25, 50, 100, 500, 2000]
REFERENCE_START, REFERENCE_END = "2017-09-04", "2022-08-28"
BASELINES = ["Persistence", "Seasonal naive", "Recent mean", "Recent trend"]
LEARNED = ["LightGBM", "XGBoost", "TabPFN-3.5"]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def completed_job(folder):
    """The same conservative completeness predicate for quoting and execution."""
    folder = Path(folder)
    if (
        not (folder / "result.json").exists()
        or not (folder / "predictions.parquet").exists()
    ):
        return False
    try:
        report = json.loads((folder / "result.json").read_text())
        rows = pd.read_parquet(folder / "predictions.parquet")
        expected = report.get(
            "prediction_rows", report["query_rows"] - report["calibration_rows"]
        )
        return bool(
            len(rows) == expected
            and rows.model.eq(report["model"]).all()
            and np.isfinite(rows[["q10", "q50", "q90"]]).all().all()
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


class SeasonalReference:
    """Fixed predevelopment references; historical activity, not clinical risk."""

    def __init__(self, observations):
        data = observations[
            observations.week_start.between(REFERENCE_START, REFERENCE_END)
            & observations.region_id.ne("national")
        ].copy()
        if data.empty:
            raise ValueError("Missing fixed 2017–2022 reference observations")
        self.reference_end = REFERENCE_END
        self.annual, self.seasonal = {}, {}
        for region, group in data.groupby("region_id"):
            self.annual[region] = np.quantile(group.incidence, [0.5, 0.8]).tolist()
            weeks = group.week_start.dt.isocalendar().week.to_numpy(dtype=int)
            for week in range(1, 54):
                delta = np.abs(weeks - week)
                selected = group.incidence[np.minimum(delta, 53 - delta) <= 4]
                if len(selected) < 20:
                    raise ValueError(f"Insufficient reference: {region} week {week}")
                self.seasonal[(region, week)] = float(selected.quantile(0.8))

    def thresholds(self, frame):
        return np.array(
            [
                self.seasonal[(r, int(w))]
                for r, w in zip(
                    frame.region_id, frame.target_start.dt.isocalendar().week
                )
            ]
        )

    def annotate(self, frame):
        result = frame.copy()
        result["seasonal_reference_p80"] = self.thresholds(frame)
        result["reference_p50"] = [self.annual[r][0] for r in frame.region_id]
        result["reference_p80"] = [self.annual[r][1] for r in frame.region_id]
        return result


def select_context(history, context_size=1000, seed=42):
    """Reserve 52 complete target weeks; purge fit labels unavailable at calibration."""
    if not isinstance(context_size, int) or context_size < 5:
        raise ValueError("Context size must be an integer of at least five")
    history = history.sort_values(["target_start", "region_id"]).copy()
    history["effective_label_available_at"] = history.truth_available_at
    # Before the archive begins, a retrospective feature reconstruction cannot
    # possess a genuine earlier publication timestamp. Preserve the real one
    # and explicitly mark this simulated maturity cutoff in a separate field.
    if "feature_provenance" in history:
        reconstructed = history.feature_provenance.eq("reconstructed_pre_archive")
        simulated = (
            (history.target_start + pd.Timedelta(days=42))
            .dt.tz_localize("Europe/Berlin")
            .dt.tz_convert("UTC")
        )
        history.loc[reconstructed, "effective_label_available_at"] = simulated[
            reconstructed
        ]
    weeks = np.sort(history.target_start.unique())
    if len(weeks) < 78:
        raise ValueError("Need 78 historical weeks including 52 calibration weeks")
    calibration_start = weeks[-52]
    calibration = history[history.target_start >= calibration_start].copy()
    boundary = calibration.issue_at.min()
    pool = history[
        (history.target_start < calibration_start)
        & (history.effective_label_available_at <= boundary)
    ].copy()
    if len(pool) < context_size:
        raise ValueError(f"Only {len(pool)} context rows, requested {context_size}")
    # A single seeded permutation makes context sizes genuinely nested.
    order = np.random.default_rng(seed).permutation(len(pool))
    context = pool.iloc[order[:context_size]].sort_values(["target_start", "region_id"])
    return (
        context.reset_index(drop=True),
        calibration.reset_index(drop=True),
        pool.reset_index(drop=True),
    )


def baseline_point(name, frame):
    if name == "Persistence":
        return frame.incidence.to_numpy()
    if name == "Seasonal naive":
        if "seasonal_naive_incidence" not in frame:
            raise ValueError(
                "Seasonal baseline requires target-week-minus-52 observation"
            )
        return frame.seasonal_naive_incidence.fillna(frame.incidence).to_numpy()
    if name == "Recent mean":
        return frame.rolling_mean3.to_numpy()
    if name == "Recent trend":
        # Two observation steps because Friday's last observation is normally
        # the previous week, and the target is next Monday's complete week.
        steps = (
            (frame.target_start - frame.latest_observed_week).dt.days / 7
        ).to_numpy()
        return np.maximum(
            0, frame.incidence.to_numpy() + steps * frame.latest_change.to_numpy()
        )
    raise ValueError(name)


def baseline_predictions(name, calibration, test, reference):
    residuals = calibration.target_incidence.to_numpy() - baseline_point(
        name, calibration
    )
    point = baseline_point(name, test)
    quantiles = np.maximum(
        0, point[:, None] + np.quantile(residuals, [0.1, 0.5, 0.9])[None, :]
    )
    # Persistence and seasonal naive keep their actual point forecast; do not
    # quietly turn the baseline into a fitted bias-corrected point estimator.
    quantiles[:, 1] = point
    quantiles[:, 0] = np.minimum(quantiles[:, 0], point)
    quantiles[:, 2] = np.maximum(quantiles[:, 2], point)
    p = np.mean(
        residuals[:, None] + point[None, :] > reference.thresholds(test)[None, :],
        axis=0,
    )
    return pd.DataFrame(
        {
            "q10": quantiles[:, 0],
            "q50": point,
            "q90": quantiles[:, 2],
            "interval_low": quantiles[:, 0],
            "interval_high": quantiles[:, 2],
            "probability_raw": p,
            "probability": p,
            "calibration_status": "empirical_calibration_residuals",
        }
    )


def seasonal_event_frequency(calibration, test, reference):
    """Smoothed recent seasonal event frequency, estimated without test labels."""
    labels = calibration.target_incidence.to_numpy() > reference.thresholds(calibration)
    cal_weeks = calibration.target_start.dt.isocalendar().week.to_numpy(dtype=int)
    values = []
    for week in test.target_start.dt.isocalendar().week:
        distance = np.abs(cal_weeks - int(week))
        selected = labels[np.minimum(distance, 53 - distance) <= 4]
        values.append((int(selected.sum()) + 1) / (len(selected) + 2))
    return np.asarray(values)


def metrics(frame):
    labeled = frame[frame.target_incidence.notna() & frame.q50.notna()]
    if labeled.empty:
        return {"rows": 0}
    y = labeled.target_incidence.to_numpy()
    event = (y > labeled.seasonal_reference_p80.to_numpy()).astype(int)
    p = labeled.probability.to_numpy()
    result = {
        "rows": len(y),
        "events": int(event.sum()),
        "mae": float(mean_absolute_error(y, labeled.q50)),
        "rmse": float(np.sqrt(mean_squared_error(y, labeled.q50))),
        "pinball": float(
            np.mean(
                [
                    mean_pinball_loss(y, labeled[f"q{int(q * 100)}"], alpha=q)
                    for q in [0.1, 0.5, 0.9]
                ]
            )
        ),
        "coverage80": float(
            np.mean((y >= labeled.interval_low) & (y <= labeled.interval_high))
        ),
        "interval_width": float((labeled.interval_high - labeled.interval_low).mean()),
        "raw_coverage80": float(np.mean((y >= labeled.q10) & (y <= labeled.q90))),
        "raw_interval_width": float((labeled.q90 - labeled.q10).mean()),
        "brier": float(brier_score_loss(event, p)),
        "brier_raw": float(brier_score_loss(event, labeled.probability_raw)),
        "event_rate": float(event.mean()),
        "average_precision": float(average_precision_score(event, p))
        if event.sum()
        else None,
    }
    if "climatology_probability" in labeled:
        result["brier_climatology"] = float(
            brier_score_loss(event, labeled.climatology_probability)
        )
    return result


def paired_block_bootstrap(predictions, competitor, n_bootstrap=2000, seed=42):
    """Four adjacent issue weeks per resampling block, all regions kept together."""
    keys = ["target_start", "region_id"]
    tab = predictions[predictions.model == "TabPFN-3.5"][
        keys + ["q50", "target_incidence"]
    ]
    other = predictions[predictions.model == competitor][keys + ["q50"]]
    pair = tab.merge(
        other, on=keys, suffixes=("_tab", "_other"), validate="one_to_one"
    ).dropna()
    if pair.empty:
        return {"competitor": competitor, "paired_rows": 0}
    pair["difference"] = abs(pair.target_incidence - pair.q50_other) - abs(
        pair.target_incidence - pair.q50_tab
    )
    weekly = pair.groupby("target_start").difference.mean().sort_index()
    calendar = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
    values = weekly.reindex(calendar).to_numpy()
    if len(values) < 8:
        return {
            "competitor": competitor,
            "paired_rows": len(pair),
            "mae_improvement": float(np.nanmean(values)),
            "ci95": None,
        }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, len(values) - 3, size=int(np.ceil(len(values) / 4)))
        sampled = np.concatenate([values[s : s + 4] for s in starts])[: len(values)]
        if np.isfinite(sampled).any():
            estimates.append(float(np.nanmean(sampled)))
    return {
        "competitor": competitor,
        "paired_rows": len(pair),
        "mae_improvement": float(np.nanmean(values)),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "method": "paired moving four-week blocks; all macroregions together",
    }


def probability_readiness(predictions):
    """An explicit empirical release gate, never a personal-risk validation."""
    rows = predictions[
        (predictions.model == "TabPFN-3.5") & predictions.target_incidence.notna()
    ].copy()
    if rows.empty:
        return {"ready": False, "reason": "no_completed_TabPFN_evaluation"}
    y = (rows.target_incidence > rows.seasonal_reference_p80).astype(int)
    p = rows.probability.to_numpy()
    report = metrics(rows)
    # Equal-frequency reliability bins; ties may reduce the number of bins.
    bins = pd.qcut(pd.Series(p).rank(method="first"), min(5, len(p)), labels=False)
    error = 0.0
    for bucket in bins.unique():
        selected = np.asarray(bins == bucket)
        error += selected.mean() * abs(
            float(p[selected].mean()) - float(y.to_numpy()[selected].mean())
        )
    checks = {
        "at_least_20_events": int(y.sum()) >= 20,
        "at_least_20_nonevents": int((1 - y).sum()) >= 20,
        "brier_beats_seasonal_frequency": report["brier"]
        < report.get("brier_climatology", -np.inf),
        "precision_recall_lift": (report["average_precision"] or 0)
        > report["event_rate"],
        "reliability_error_at_most_005": error <= 0.05,
    }
    return {
        "ready": bool(all(checks.values())),
        "checks": checks,
        "reliability_ece": float(error),
        "validation": "locked retrospective community-event evaluation; not individual infection probability",
    }


def tune_tree(name, jobs, columns):
    """Only explicitly supplied development jobs; never tune on locked outcomes."""
    scores = []
    for config in SEARCH_CONFIGS:
        losses = []
        for context, test in jobs:
            if test.target_start.max() >= LOCKED_TARGETS.min():
                raise ValueError("Locked targets cannot enter tuning")
            median = feature_values(context, columns).median().fillna(0)

            def x(data):
                values = feature_values(data, columns)
                return np.column_stack(
                    [values.fillna(median), values.isna().astype(float)]
                )

            y = np.log1p(context.target_incidence) - np.log1p(context.incidence)
            model = make_tree(name, 0.5, config).fit(x(context), y)
            prediction = np.maximum(
                0, np.expm1(model.predict(x(test)) + np.log1p(test.incidence))
            )
            losses.append(mean_absolute_error(test.target_incidence, prediction))
        scores.append(float(np.mean(losses)))
    if not scores or not np.isfinite(scores).all():
        raise ValueError("No valid development tuning rows")
    return {
        "config": SEARCH_CONFIGS[int(np.argmin(scores))],
        "development_mae": scores,
        "transform": "log_change",
        "calibration_method": "raw",
        "calibration_selection": "No TabPFN development calibration experiment; raw retained explicitly",
    }


def model_job(
    name,
    context,
    calibration,
    test,
    reference,
    columns,
    folder,
    config=None,
    calibration_method="raw",
):
    """Worker process entry point. Combines posterior queries, not their labels."""
    import os
    import time

    os.environ.setdefault(
        "SKB_DATA_DIRECTORY", str(Path(folder).resolve() / "skrub_cache")
    )
    os.environ.setdefault("MPLCONFIGDIR", str(Path(folder).resolve() / "mpl_cache"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "predictions.parquet"
    if completed_job(folder):
        return json.loads((folder / "result.json").read_text())
    start = time.monotonic()
    if name in BASELINES:
        predicted = baseline_predictions(name, calibration, test, reference)
        version = {"baseline": name}
    else:
        model_name = name.removesuffix(" full-history")
        path = folder / "model.joblib"
        model = (
            RespiratoryModel.load(path)
            if path.exists()
            else RespiratoryModel(model_name, columns, config).fit(context)
        )
        model.save(path)
        queries = pd.concat([calibration, test], ignore_index=True)
        raw = model.raw(queries, reference.thresholds(queries), folder / "api_cache")
        model.calibrate(
            calibration,
            raw.iloc[: len(calibration)],
            reference.thresholds(calibration),
            calibration_method,
        )
        predicted = model.adjusted(raw.iloc[len(calibration) :]).reset_index(drop=True)
        model.save(path)
        version = model.api_version or {"baseline": model_name}
    result = pd.concat(
        [reference.annotate(test).reset_index(drop=True), predicted], axis=1
    )
    result["climatology_probability"] = seasonal_event_frequency(
        calibration, test, reference
    )
    result["model"] = name
    result.to_parquet(output, index=False)
    report = {
        "model": name,
        "context_rows": len(context),
        "calibration_rows": len(calibration),
        "prediction_rows": len(test),
        "query_rows": len(calibration) + len(test),
        "model_version": version,
        "runtime_seconds": time.monotonic() - start,
        **metrics(result),
    }
    write_json(folder / "result.json", report)
    return report
