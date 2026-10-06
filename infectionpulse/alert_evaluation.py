"""Exploratory alert utility on saved forecasts; no fitting or API calls."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = [
    "TabPFN-3.5",
    "LightGBM full-history",
    "LightGBM",
    "XGBoost full-history",
    "XGBoost",
    "Persistence",
    "Seasonal naive",
    "Recent mean",
    "Recent trend",
]
KEYS = ["target_start", "region_id"]
POLICIES = {
    "high_burden": {
        "alert": "q50 >= reference_p80",
        "event": "target_incidence >= reference_p80",
        "purpose": "App high-community-burden trigger, before activity rules.",
    },
    "seasonal_event": {
        "alert": "probability >= 0.5",
        "event": "target_incidence > seasonal_reference_p80",
        "purpose": "Research-only seasonal exceedance; public probabilities remain gated.",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_protocol(run, path, n_bootstrap=2000):
    """Create once, before evaluation. Existing changed inputs/protocol fail closed."""
    run, path = Path(run), Path(path)
    protocol = {
        "schema_version": 1,
        "run_id": run.name,
        "evidence_status": "exploratory_retrospective",
        "disclosure": (
            "Forecast accuracy on this test was inspected before this alert analysis. "
            "This protocol was frozen before computing alert comparisons, not before "
            "the original test. No confirmatory alert-superiority claim is permitted."
        ),
        "policies": POLICIES,
        "models": MODELS,
        "threshold_selection": "Fixed existing policy; no tuning on test or development.",
        "comparison": "All models on the same complete region-weeks; no missing=negative.",
        "primary_comparator": "LightGBM full-history",
        "bootstrap": {
            "replicates": n_bootstrap,
            "seed": 20260921,
            "block_weeks": 4,
            "method": "moving calendar-week blocks, all regions together",
        },
        "inputs_sha256": {
            name: sha256(run / name)
            for name in [
                "predictions.parquet",
                "coverage_rows.parquet",
                "manifest.json",
            ]
        },
        "scope": "Community high-burden weeks, not individual infection or benefit of WFH.",
    }
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {k: v for k, v in existing.items() if k != "frozen_at"}
        if comparable != protocol:
            raise ValueError(
                "Frozen alert protocol or input hashes changed; use a new output directory."
            )
        return existing
    protocol["frozen_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing a concurrently frozen protocol.
    with path.open("x") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
    return protocol


def matched_predictions(predictions, coverage, models):
    """Validate pairing, available forecasts and shared labels before any scoring."""
    frame = predictions[predictions.model.isin(models)].copy()
    coverage = coverage.copy()
    for data in (frame, coverage):
        data["target_start"] = pd.to_datetime(data.target_start, utc=True)
    if coverage.duplicated(KEYS).any() or frame.duplicated(["model", *KEYS]).any():
        raise ValueError("Duplicate prediction or coverage keys")
    missing = set(models) - set(frame.model)
    if missing:
        raise ValueError(f"Missing required models: {sorted(missing)}")
    planned = pd.MultiIndex.from_frame(coverage[KEYS])
    if not pd.MultiIndex.from_frame(frame[KEYS]).isin(planned).all():
        raise ValueError("Predictions contain unplanned keys")
    numeric = [
        "q50",
        "probability",
        "reference_p80",
        "seasonal_reference_p80",
        "target_incidence",
    ]
    usable = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    usable &= frame.probability.between(0, 1)
    if "eligible" in frame:
        usable &= frame.eligible.eq(True)
    eligible_keys = pd.MultiIndex.from_frame(
        coverage.loc[coverage.eligible.eq(True), KEYS]
    )
    usable &= pd.MultiIndex.from_frame(frame[KEYS]).isin(eligible_keys)
    valid = frame.loc[usable].copy()
    for col in ["target_incidence", "reference_p80", "seasonal_reference_p80"]:
        if (valid.groupby(KEYS)[col].nunique() > 1).any():
            raise ValueError(f"Models disagree on shared {col}")
    size = valid.groupby(KEYS).model.nunique()
    common = size[size == len(models)].index
    matched = valid[pd.MultiIndex.from_frame(valid[KEYS]).isin(common)].copy()
    if matched.empty:
        raise ValueError("No complete matched region-weeks")
    rows = []
    for model in models:
        issued = int((valid.model == model).sum())
        rows.append(
            {
                "model": model,
                "planned_region_weeks": len(coverage),
                "usable_predictions": issued,
                "unavailable_region_weeks": len(coverage) - issued,
                "matched_region_weeks": len(common),
                "unmatched_usable_predictions": issued - len(common),
                "matched_coverage": len(common) / len(coverage),
            }
        )
    return matched.sort_values(["model", *KEYS]), pd.DataFrame(rows)


def labels(frame, policy):
    if policy == "high_burden":
        return (frame.target_incidence >= frame.reference_p80).to_numpy(), (
            frame.q50 >= frame.reference_p80
        ).to_numpy()
    if policy == "seasonal_event":
        return (frame.target_incidence > frame.seasonal_reference_p80).to_numpy(), (
            frame.probability >= 0.5
        ).to_numpy()
    raise ValueError(f"Unknown alert policy: {policy}")


def counts(event, alert):
    event, alert = np.asarray(event, dtype=bool), np.asarray(alert, dtype=bool)
    return np.array(
        [
            np.sum(event & alert),
            np.sum(~event & alert),
            np.sum(event & ~alert),
            np.sum(~event & ~alert),
        ],
        dtype=int,
    )


def scores(values):
    tp, fp, fn, tn = map(int, values)
    total = tp + fp + fn + tn

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "region_weeks": total,
        "events": tp + fn,
        "alerts": tp + fp,
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_events": fn,
        "true_negatives": tn,
        "recall": ratio(tp, tp + fn),
        "precision": ratio(tp, tp + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_discovery_rate": ratio(fp, tp + fp),
        "false_alerts_per_100_region_weeks": ratio(100 * fp, total),
        "alert_rate": ratio(tp + fp, total),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def paired_intervals(frame, policy, calendar, config):
    """Pair calendar blocks, retaining empty source-stale weeks in the time grid."""
    models = ["TabPFN-3.5", "LightGBM full-history"]
    cubes = []
    for model in models:
        data = frame[frame.model == model]
        event, alert = labels(data, policy)
        records = pd.DataFrame(
            {
                "week": data.target_start.to_numpy(),
                "tp": event & alert,
                "fp": ~event & alert,
                "fn": event & ~alert,
                "tn": ~event & ~alert,
            }
        )
        cubes.append(
            records.groupby("week")[["tp", "fp", "fn", "tn"]]
            .sum()
            .reindex(calendar, fill_value=0)
            .to_numpy(dtype=int)
        )
    cube = np.stack(cubes)
    width = min(config["block_weeks"], len(calendar))
    rng = np.random.default_rng(config["seed"])
    metrics = {"recall": 1, "precision": 1, "false_positive_rate": -1, "f1": 1}
    samples = {metric: [] for metric in metrics}
    for _ in range(config["replicates"]):
        starts = rng.integers(
            0, len(calendar) - width + 1, size=int(np.ceil(len(calendar) / width))
        )
        indices = np.concatenate([np.arange(start, start + width) for start in starts])[
            : len(calendar)
        ]
        pair = [scores(values) for values in cube[:, indices].sum(axis=1)]
        for metric, direction in metrics.items():
            if all(item[metric] is not None for item in pair):
                samples[metric].append(direction * (pair[0][metric] - pair[1][metric]))
    actual = [scores(values) for values in cube.sum(axis=1)]
    return [
        {
            "policy": policy,
            "comparator": models[1],
            "metric": metric,
            "positive_means": "TabPFN advantage",
            "difference": direction * (actual[0][metric] - actual[1][metric])
            if all(item[metric] is not None for item in actual)
            else None,
            "ci95": np.quantile(samples[metric], [0.025, 0.975]).tolist()
            if samples[metric]
            else None,
            "valid_replicates": len(samples[metric]),
        }
        for metric, direction in metrics.items()
    ]


def evaluate(run, output, protocol):
    run, output = Path(run), Path(output)
    for name, digest in protocol["inputs_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    frame, coverage = matched_predictions(
        pd.read_parquet(run / "predictions.parquet"),
        pd.read_parquet(run / "coverage_rows.parquet"),
        protocol["models"],
    )
    planned = pd.read_parquet(run / "coverage_rows.parquet")
    calendar = pd.date_range(
        pd.to_datetime(planned.target_start, utc=True).min(),
        pd.to_datetime(planned.target_start, utc=True).max(),
        freq="7D",
    )
    metrics, intervals = [], []
    for policy in protocol["policies"]:
        for model in protocol["models"]:
            data = frame[frame.model == model]
            metrics.append(
                {
                    "policy": policy,
                    "model": model,
                    **scores(counts(*labels(data, policy))),
                }
            )
        intervals.extend(
            paired_intervals(frame, policy, calendar, protocol["bootstrap"])
        )
    summary = pd.DataFrame(metrics)
    summary.to_csv(output / "alert_summary.csv", index=False)
    coverage.to_csv(output / "alert_coverage.csv", index=False)
    result = {
        "protocol": protocol,
        "metrics": metrics,
        "coverage": coverage.to_dict("records"),
        "paired_intervals": intervals,
        "limitations": [
            "Exploratory extension of an already-inspected retrospective test; not a preregistered alert trial.",
            "High burden is a historical surveillance percentile, not a clinically validated danger threshold.",
            "Four macroregions; repeated regional weeks are correlated. Intervals retain calendar blocks and regions together.",
            "No individual exposure outcomes, avoided infections, or causal benefit of working from home were measured.",
            "The seasonal probability diagnostic does not override the failed public probability-readiness gate.",
            "Unavailable forecasts abstain; performance metrics use common available rows and disclose reduced coverage.",
            "Historical source vintages were used for testing; early development/training features were reconstructed.",
        ],
    }
    (output / "alert_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Exploratory warning evaluation",
        "",
        protocol["disclosure"],
        "",
        "Primary alert: forecast median reaches the existing regional historical 80th-percentile burden threshold. "
        "This evaluates the app's community-burden trigger before personal activity rules; it does not validate personal risk or WFH benefit.",
        "",
        f"Matched coverage: {coverage.iloc[0].matched_region_weeks}/{coverage.iloc[0].planned_region_weeks} region-weeks. "
        "One alert means one region-week, not one person or one independent outbreak.",
        "",
    ]
    for policy in protocol["policies"]:
        lines += [
            f"## {policy.replace('_', ' ').title()}",
            "",
            f"Alert: `{POLICIES[policy]['alert']}`; observed event: `{POLICIES[policy]['event']}`.",
            "",
            "| Model | Events detected | Missed | Alerts | False alerts | Precision | Recall | False-positive rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary[summary.policy == policy].to_dict("records"):

            def pct(value):
                return (
                    f"{value:.1%}"
                    if value is not None and pd.notna(value)
                    else "undefined"
                )

            lines.append(
                f"| {row['model']} | {row['true_alerts']}/{row['events']} | {row['missed_events']} | {row['alerts']} | {row['false_alerts']} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['false_positive_rate'])} |"
            )
        lines += [
            "",
            "Precision = correct alerts / all alerts. Recall = alerted high-burden weeks / all high-burden weeks. "
            "False-positive rate = false alerts / weeks without the event; it differs from the share of alerts that were false.",
            "",
            "Paired TabPFN advantage versus full-history LightGBM (percentage points; positive favors TabPFN):",
            "",
        ]
        for row in intervals:
            if row["policy"] == policy and row["ci95"] is not None:
                lines.append(
                    f"- {row['metric']}: {row['difference'] * 100:+.1f}; 95% block interval [{row['ci95'][0] * 100:+.1f}, {row['ci95'][1] * 100:+.1f}]."
                )
        lines += [""]
    lines += [
        "## Limits",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "Reproduce offline: `.venv/bin/python scripts/alert_benchmark.py`. The frozen protocol hashes all source artifacts; changed inputs require a separate output directory.",
    ]
    (output / "alert_report.md").write_text("\n".join(lines) + "\n")
    return result
