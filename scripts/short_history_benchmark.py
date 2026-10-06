#!/usr/bin/env python3
"""Exploratory: the locked ARE test with only 104 training rows (26 weeks x 4 regions).

Reuses the locked run's test, calibration rows and tuned baseline settings, but
fits each model on the most recent 104 region-weeks (26 weeks x 4 regions).
Question: do forecast ranges stay honest with short history? Designed after the
main test was inspected, so it is exploratory, not confirmatory.

Hosted TabPFN calls need --max-api-units (cached results are never re-billed).
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from respiratory_benchmark import isolated, quote_dimensions

from infectionpulse.respiratory_data import FEATURE_COLUMNS
from infectionpulse.respiratory_evaluation import completed_job, metrics, model_job

ROOT = Path(__file__).resolve().parents[1]
ROWS = 104
MODELS = ["TabPFN-3.5", "LightGBM", "XGBoost"]


def paired(predictions, competitor, statistic, n_bootstrap=2000, seed=42):
    """Competitor minus TabPFN; four-week blocks, all regions kept together."""
    keys = ["target_start", "region_id"]
    tab = predictions[predictions.model == "TabPFN-3.5"].sort_values(keys)
    other = predictions[predictions.model == competitor].sort_values(keys)
    if not (tab[keys].to_numpy() == other[keys].to_numpy()).all():
        raise ValueError("Unpaired predictions")
    difference = statistic(other).to_numpy(float) - statistic(tab).to_numpy(float)
    weekly = (
        pd.Series(difference, index=tab.target_start.to_numpy())
        .groupby(level=0)
        .mean()
        .sort_index()
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    estimates = [
        np.concatenate(
            [
                weekly[s : s + 4]
                for s in rng.integers(0, len(weekly) - 3, int(np.ceil(len(weekly) / 4)))
            ]
        )[: len(weekly)].mean()
        for _ in range(n_bootstrap)
    ]
    return float(weekly.mean()), np.percentile(estimates, [2.5, 97.5]).tolist()


def above_range(frame):
    return frame.target_incidence > frame.q90


def summary(predictions, names=MODELS):
    rows = []
    for name in names:
        group = predictions[predictions.model == name]
        if group.empty:
            continue
        m = metrics(group)
        y = group.target_incidence
        event = y >= group.reference_p80
        warn = group.q90 >= group.reference_p80
        rows.append(
            {
                "model": name,
                "rows": m["rows"],
                "mae": m["mae"],
                "pinball": m["pinball"],
                "raw_inside": m["raw_coverage80"],
                "raw_above": float(above_range(group).mean()),
                "raw_width": m["raw_interval_width"],
                "calibrated_inside": m["coverage80"],
                "calibrated_above": float((y > group.interval_high).mean()),
                "calibrated_width": m["interval_width"],
                "q90_warning_missed": int((event & ~warn).sum()),
                "q90_warning_false": int((warn & ~event).sum()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "models/respiratory/runs/65cf524d3927a4b8"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models/respiratory/short_history"
    )
    parser.add_argument("--max-api-units", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    plan = joblib.load(args.run / "plan.joblib")
    jobs = [j for j in plan["jobs"] if j["status"] == "ready"]
    pending = [
        j for j in jobs if not completed_job(args.output / j["block"] / "TabPFN-3.5")
    ]
    if pending:
        quote = isolated(
            quote_dimensions,
            [
                (j["block"], ROWS, len(j["calibration"]) + len(j["test"]))
                for j in pending
            ],
        )
        print(f"TabPFN quote: {quote['estimated_units']} units", flush=True)
        if quote["estimated_units"] > args.max_api_units:
            sys.exit("Quote exceeds --max-api-units; no hosted inference started")
    for job in jobs:
        context = job["pool"].sort_values(["target_start", "region_id"]).tail(ROWS)
        for name in MODELS:
            isolated(
                model_job,
                name,
                context,
                job["calibration"],
                job["test"],
                plan["reference"],
                FEATURE_COLUMNS,
                args.output / job["block"] / name,
                plan["tuning"].get(name, {}).get("config"),
            )
    predictions = pd.concat(
        pd.read_parquet(args.output / j["block"] / name / "predictions.parquet")
        for j in jobs
        for name in MODELS
    )
    locked = pd.read_parquet(args.run / "predictions.parquet")
    reports = ROOT / "reports"
    predictions[
        ["model", "region_id", "target_start", "q10", "q50", "q90", "target_incidence"]
    ].sort_values(["model", "region_id", "target_start"]).to_csv(
        reports / "short_history_forecasts.csv", index=False, float_format="%.1f"
    )
    short = summary(predictions)
    full = summary(locked, [*MODELS, "LightGBM full-history"])
    lines = [
        "# Small training sets: forecast ranges and warnings",
        "",
        f"Locked ARE test (run `{args.run.name}`, {len(predictions) // len(MODELS)} "
        f"region-weeks), but every model is trained on only the last {ROWS} rows "
        "(26 weeks × 4 regions). Exploratory: designed after the main test was "
        "inspected. *Calibrated* ranges add the project's conformal adjustment, "
        "fitted on 208 earlier calibration rows; *raw* ranges are the model's own "
        "10–90% quantiles.",
        "",
        "| Training rows | Model | MAE ↓ | Pinball ↓ | Raw range: inside / above | "
        "Calibrated: inside / above | Calibrated width | q90 warning: missed / false |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, table in [(str(ROWS), short), ("1,000 (main test)", full)]:
        for r in table.itertuples():
            lines.append(
                f"| {label} | {r.model} | {r.mae:.0f} | {r.pinball:.0f} | "
                f"{r.raw_inside:.0%} / {r.raw_above:.0%} | "
                f"{r.calibrated_inside:.0%} / {r.calibrated_above:.0%} | "
                f"{r.calibrated_width:,.0f} | "
                f"{r.q90_warning_missed} / {r.q90_warning_false} |"
            )
    lines += [
        "",
        "Paired differences in raw ranges, competitor minus TabPFN "
        "(four-week block bootstrap, 95% CI):",
        "",
    ]
    for label, frame, competitor in [
        (f"{ROWS} rows", predictions, "LightGBM"),
        (f"{ROWS} rows", predictions, "XGBoost"),
        ("1,000 rows", locked, "LightGBM"),
        ("1,000 rows", locked, "LightGBM full-history"),
    ]:
        above, above_ci = paired(frame, competitor, above_range)
        mae, mae_ci = paired(
            frame, competitor, lambda g: (g.target_incidence - g.q50).abs()
        )
        lines.append(
            f"- {label}, {competitor}: above-range rate {above:+.1%} "
            f"[{above_ci[0]:+.1%}, {above_ci[1]:+.1%}]; "
            f"MAE {mae:+.0f} [{mae_ci[0]:+.0f}, {mae_ci[1]:+.0f}]"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- TabPFN's raw ranges are usable without a calibration step; raw tree "
        "quantiles are overconfident with few rows.",
        "- After calibration, LightGBM's ranges are about as honest (77% inside) with "
        "narrower width; TabPFN keeps fewer outcomes above its range but is wider.",
        "- Warning on q90 ≥ high threshold trades misses for false alarms: TabPFN's "
        "wider ranges miss fewer high weeks and raise more false warnings. This does "
        "not show better warnings overall.",
        "- Point accuracy (MAE) is level with LightGBM at 104 rows.",
        "",
        "Reproduce: `python scripts/short_history_benchmark.py` (cached; new hosted calls "
        "need `--max-api-units`).",
        "",
    ]
    (reports / "small_training_sets.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
