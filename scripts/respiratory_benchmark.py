#!/usr/bin/env python3
"""ARE evidence CLI. Hosted inference requires an explicit quoted-unit ceiling.

Examples:
  python scripts/respiratory_benchmark.py prepare
  python scripts/respiratory_benchmark.py run --baselines-only
  python scripts/respiratory_benchmark.py quote
  python scripts/respiratory_benchmark.py run --max-api-units 100000
  python scripts/respiratory_benchmark.py predict --model Persistence
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from infectionpulse.respiratory_data import (
    FEATURE_COLUMNS,
    SOURCE_URL,
    SnapshotStore,
    build_backtest_rows,
    build_issue_rows,
    build_training_rows,
    issue_for_target,
    utc,
)
from infectionpulse.respiratory_evaluation import (
    BASELINES,
    CONTEXT_LADDER,
    DEVELOPMENT_ORIGINS,
    LOCKED_TARGETS,
    REFERENCE_END,
    SeasonalReference,
    completed_job,
    fingerprint,
    metrics,
    model_job,
    paired_block_bootstrap,
    probability_readiness,
    select_context,
    tune_tree,
    write_json,
)
from infectionpulse.respiratory_model import BATCH_ROWS, atomic_dump

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_AS_OF = "2026-09-17T00:00:00+00:00"
MODELS = BASELINES + [
    "LightGBM",
    "XGBoost",
    "LightGBM full-history",
    "XGBoost full-history",
    "TabPFN-3.5",
]


def isolated(function, *args):
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return executor.submit(function, *args).result()


def preparation_config(args, store):
    manifest = store.manifest_frame()
    manifest = manifest[manifest.available_at <= utc(args.as_of)]
    return {
        "schema_version": 1,
        "context_size": args.context_size,
        "block_weeks": args.block_weeks,
        "seed": 42,
        "target_start": str(LOCKED_TARGETS.min().date()),
        "target_end": str(LOCKED_TARGETS.max().date()),
        "reference_end": REFERENCE_END,
        "snapshots": manifest[["snapshot_id", "sha256"]].to_dict("records"),
        "as_of": args.as_of,
        "code_sha256": hashlib.sha256(
            b"".join(
                p.read_bytes()
                for p in [
                    Path(__file__),
                    ROOT / "infectionpulse/respiratory_model.py",
                    ROOT / "infectionpulse/respiratory_evaluation.py",
                    ROOT / "infectionpulse/respiratory_data.py",
                ]
            )
        ).hexdigest(),
    }


def prepare(args):
    store = SnapshotStore(args.data)
    manifest = store.manifest_frame()
    if manifest.empty:
        raise ValueError("No respiratory snapshots. Run respiratory_refresh.py first.")
    config = preparation_config(args, store)
    run_id = fingerprint(config)
    folder = args.output / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / "plan.joblib"
    if plan_path.exists():
        return folder, joblib.load(plan_path)
    reference = SeasonalReference(store.load_snapshot(manifest.iloc[0].snapshot_id))
    print(
        "Building publication-aware locked rows and calibration histories", flush=True
    )
    test_rows = build_backtest_rows(store, LOCKED_TARGETS, truth_as_of=args.as_of)
    test_rows.to_parquet(folder / "coverage_rows.parquet", index=False)
    jobs = []
    for i in range(0, len(LOCKED_TARGETS), args.block_weeks):
        starts = LOCKED_TARGETS[i : i + args.block_weeks]
        issue = issue_for_target(starts[0])
        block = test_rows[test_rows.target_start.isin(starts)]
        eligible = block[block.eligible].reset_index(drop=True)
        if eligible.empty:
            jobs.append(
                {
                    "block": str(starts[0].date()),
                    "status": "no_eligible_features",
                    "test": eligible,
                }
            )
            continue
        history = build_training_rows(store, issue)
        context, calibration, pool = select_context(history, args.context_size)
        for label, rows in [
            ("context", context),
            ("calibration", calibration),
            ("test", eligible),
        ]:
            keys = [
                k
                for k in [
                    "region_id",
                    "issue_at",
                    "target_start",
                    "snapshot_id",
                    "truth_snapshot_id",
                    "feature_provenance",
                ]
                if k in rows
            ]
            key_path = folder / str(starts[0].date()) / f"{label}_rows.csv"
            key_path.parent.mkdir(parents=True, exist_ok=True)
            rows[keys].to_csv(key_path, index=False)
        jobs.append(
            {
                "block": str(starts[0].date()),
                "status": "ready",
                "context": context,
                "calibration": calibration,
                "pool": pool,
                "test": eligible,
            }
        )
    tuning_jobs, development_specs = [], []
    for target in DEVELOPMENT_ORIGINS:
        history = build_training_rows(
            store, issue_for_target(target), allow_reconstructed_as_of=True
        )
        context, calibration, pool = select_context(history, args.context_size)
        dev = build_backtest_rows(
            store,
            pd.date_range(target, periods=4, freq="W-MON"),
            allow_reconstructed=True,
            truth_as_of=args.as_of,
        )
        dev = dev[dev.eligible & dev.target_incidence.notna()].reset_index(drop=True)
        if not dev.empty:
            tuning_jobs.append((context, dev))
            development_specs.append(
                {
                    "block": target,
                    "context": context,
                    "calibration": calibration,
                    "pool": pool,
                    "test": dev,
                }
            )
    if not tuning_jobs:
        raise ValueError("No development rows; cannot claim tuned GBDT comparison")
    configs = {}
    for name in ["LightGBM", "XGBoost"]:
        configs[name] = isolated(tune_tree, name, tuning_jobs, FEATURE_COLUMNS)
        configs[name + " full-history"] = isolated(
            tune_tree,
            name,
            [(job["pool"], job["test"]) for job in development_specs],
            FEATURE_COLUMNS,
        )
    plan = {
        "config": config,
        "run_id": run_id,
        "jobs": jobs,
        "reference": reference,
        "tuning": configs,
        "coverage": test_rows,
        "development_jobs": tuning_jobs,
        "development_specs": development_specs,
    }
    atomic_dump(plan, plan_path)
    write_json(
        folder / "manifest.json",
        {
            **config,
            "run_id": run_id,
            "status": "prepared",
            "tuning": configs,
            "models": MODELS,
            "planned_rows": len(LOCKED_TARGETS) * 4,
            "planned_weeks": len(LOCKED_TARGETS),
            "limitations": [
                "ARE is survey-reported respiratory illness, not individual infection probability.",
                "Four macroregions are correlated; block bootstrap resamples time, not individual rows.",
                "Prearchive development/training features are reconstructed and explicitly marked.",
                "Prearchive purging uses simulated 42-day label availability; real snapshot timestamps are preserved separately.",
                "Training/calibration use mature labels available at issue, excluding roughly the latest five weeks.",
                "Historical availability uses GitHub commit timestamps as publication proxies.",
                "Raw posterior probability is uncalibrated unless a separately validated calibration is selected.",
            ],
        },
    )
    write_json(
        args.output / "latest_run.json",
        {"run_id": run_id, "path": str(folder), "status": "prepared"},
    )
    return folder, plan


def quote_dimensions(shapes):
    """Runs in an isolated SDK process; metadata/dimensions only, no inference."""
    from infectionpulse.tabpfn_api import require_api_token

    require_api_token()
    from tabpfn_client import estimate_cost
    from tabpfn_client.config import get_access_token

    get_access_token()  # Initialize authentication before reading model limits.
    results = []
    from tabpfn_client.estimator import _limit_for_model_path

    limit = _limit_for_model_path("v3.5_default")
    if limit is None:
        raise ValueError(
            "Cannot verify hosted query limits; refusing an incomplete quote"
        )
    for label, context_rows, query_rows in shapes:
        if (
            min(limit.test_set_max_rows, limit.predict_row_pairs_budget // context_rows)
            < BATCH_ROWS
        ):
            raise ValueError(
                "Configured prediction chunk exceeds active server query limits"
            )
        quotes = []
        for start in range(0, query_rows, BATCH_ROWS):
            count = min(BATCH_ROWS, query_rows - start)
            result = estimate_cost(
                np.empty((context_rows, len(FEATURE_COLUMNS) * 2)),
                np.empty((count, len(FEATURE_COLUMNS) * 2)),
                model_version="v3.5",
                n_estimators=8,
                operation="predict",
            )
            quotes.append(result.model_dump(mode="json"))
        results.append(
            {
                "block": label,
                "context_rows": context_rows,
                "query_rows": query_rows,
                "chunks": quotes,
                "estimated_units": sum(q["estimated_cost"] for q in quotes),
            }
        )
    versions = {q["pricing_version"] for r in results for q in r["chunks"]}
    if len(versions) > 1:
        raise ValueError("Cannot combine incompatible quota versions")
    return {
        "estimated_units": sum(r["estimated_units"] for r in results),
        "pricing_versions": sorted(versions),
        "blocks": results,
        "batch_rows": BATCH_ROWS,
        "warning": "Estimate is not a hard billing cap; reserve quota for retries and live requests.",
    }


def quote_plan(folder, plan, include_development=False, live_shape=None):
    shapes = [
        (j["block"], len(j["context"]), len(j["calibration"]) + len(j["test"]))
        for j in plan["jobs"]
        if j["status"] == "ready"
        and not completed_job(folder / j["block"] / "TabPFN-3.5")
    ]
    if include_development:
        for job in plan["development_specs"]:
            for size in CONTEXT_LADDER:
                if len(job["pool"]) >= size and not completed_job(
                    folder
                    / "development_ladder"
                    / job["block"]
                    / str(size)
                    / "TabPFN-3.5"
                ):
                    shapes.append(
                        (
                            f"development:{job['block']}:N={size}",
                            size,
                            len(job["calibration"]) + len(job["test"]),
                        )
                    )
    if live_shape is not None:
        shapes.append(live_shape)
    quote = (
        isolated(quote_dimensions, shapes)
        if shapes
        else {"estimated_units": 0, "blocks": []}
    )
    write_json(folder / "quote.json", quote)
    return quote


def summarize(folder, plan, expected_models):
    frames = []
    for path in folder.glob("????-??-??/*/predictions.parquet"):
        frames.append(pd.read_parquet(path))
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if predictions.empty:
        return {"status": "incomplete", "errors": ["No model predictions"]}
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    summaries = [
        {"model": name, **metrics(group)}
        for name, group in predictions.groupby("model")
    ]
    pd.DataFrame(summaries).to_csv(folder / "summary.csv", index=False)
    predictions["season"] = np.where(
        predictions.target_start < pd.Timestamp("2025-08-04"), "2024/25", "2025/26"
    )
    pd.DataFrame(
        [
            {"model": name, "season": season, **metrics(group)}
            for (name, season), group in predictions.groupby(["model", "season"])
        ]
    ).to_csv(folder / "season_summary.csv", index=False)
    comparisons = [
        paired_block_bootstrap(predictions, name)
        for name in expected_models
        if name != "TabPFN-3.5"
    ]
    write_json(folder / "paired_comparisons.json", comparisons)
    coverage = plan["coverage"]
    completed = {
        name: int(predictions[predictions.model == name].shape[0])
        for name in expected_models
    }
    expected_issued = int(coverage.eligible.sum())
    status = (
        "complete"
        if all(n == expected_issued for n in completed.values())
        else "incomplete"
    )
    report = {
        "status": status,
        "planned_rows": len(LOCKED_TARGETS) * 4,
        "eligible_rows": expected_issued,
        "truth_available_rows": int(
            (coverage.eligible & coverage.target_incidence.notna()).sum()
        ),
        "abstention_counts": coverage.loc[~coverage.eligible, "status"]
        .value_counts()
        .to_dict(),
        "predictions_per_model": completed,
        "summary": summaries,
        "paired_comparisons": comparisons,
        "probability_readiness": probability_readiness(predictions),
    }
    if status != "complete":
        report["probability_readiness"]["ready"] = False
        report["probability_readiness"]["reason"] = "evaluation_incomplete"
    # Reliability is diagnostic; it does not silently promote raw probabilities.
    bins = []
    for name, group in predictions.dropna(subset=["target_incidence"]).groupby("model"):
        group = group.copy()
        group["event"] = group.target_incidence > group.seasonal_reference_p80
        group["bin"] = pd.cut(
            group.probability, np.linspace(0, 1, 6), include_lowest=True
        )
        for interval, rows in group.groupby("bin", observed=True):
            bins.append(
                {
                    "model": name,
                    "bin": str(interval),
                    "rows": len(rows),
                    "mean_probability": float(rows.probability.mean()),
                    "event_frequency": float(rows.event.mean()),
                }
            )
    pd.DataFrame(bins).to_csv(folder / "reliability.csv", index=False)
    write_json(folder / "report.json", report)
    return report


def run(args):
    folder, plan = prepare(args)
    if args.command == "prepare":
        print(f"Prepared {folder}")
        return 0
    names = [n for n in MODELS if n != "TabPFN-3.5"] if args.baselines_only else MODELS
    if args.command == "quote" or "TabPFN-3.5" in names:
        live_shape = None
        if args.command == "quote" and args.include_live:
            store = SnapshotStore(args.data)
            issued = pd.Timestamp.now(tz="UTC")
            live = build_issue_rows(store, issued)
            if not live.eligible.all():
                raise ValueError(
                    "Live source unavailable; cannot quote a valid live forecast"
                )
            live_context, live_calibration, _ = select_context(
                build_training_rows(store, issued), args.context_size
            )
            live_shape = ("live", len(live_context), len(live_calibration) + len(live))
        quote = quote_plan(
            folder,
            plan,
            args.command == "quote" and args.include_development,
            live_shape,
        )
        print(json.dumps(quote, indent=2))
        if args.command == "quote":
            return 0
        if quote["estimated_units"] > args.max_api_units:
            raise ValueError(
                "Quote exceeds --max-api-units; no hosted inference started"
            )
    errors = []
    for job in plan["jobs"]:
        if job["status"] != "ready":
            continue
        for name in names:
            print(f"ARE {name}: block={job['block']}", flush=True)
            context = job["pool"] if name.endswith("full-history") else job["context"]
            config = plan["tuning"].get(name, {}).get("config")
            try:
                isolated(
                    model_job,
                    name,
                    context,
                    job["calibration"],
                    job["test"],
                    plan["reference"],
                    FEATURE_COLUMNS,
                    folder / job["block"] / name,
                    config,
                )
            except Exception as exc:
                errors.append({"block": job["block"], "model": name, "error": str(exc)})
                write_json(folder / "errors.json", errors)
                if name == "TabPFN-3.5":
                    # Do not spend on every remaining block after a contract or
                    # authentication failure; successful chunks remain cached.
                    report = summarize(folder, plan, MODELS)
                    write_json(
                        args.output / "latest_run.json",
                        {"run_id": plan["run_id"], "path": str(folder), **report},
                    )
                    return 2
    report = summarize(folder, plan, MODELS)
    write_json(
        args.output / "latest_run.json",
        {"run_id": plan["run_id"], "path": str(folder), **report},
    )
    print(pd.DataFrame(report.get("summary", [])).to_string(index=False))
    return 0 if report["status"] == "complete" and not errors else 2


def ladder(args):
    """Optional development-only experiment; never consumes quota implicitly."""
    folder, plan = prepare(args)
    selected = []
    for job in plan["development_specs"]:
        pool = job["pool"]
        order = np.random.default_rng(42).permutation(len(pool))
        for size in CONTEXT_LADDER:
            if len(pool) < size:
                continue
            selected.append(
                {
                    **job,
                    "size": size,
                    "context": pool.iloc[order[:size]].reset_index(drop=True),
                }
            )
    destination = folder / "development_ladder"
    if not args.baselines_only:
        shapes = [
            (
                f"{j['block']}:N={j['size']}",
                j["size"],
                len(j["calibration"]) + len(j["test"]),
            )
            for j in selected
            if not completed_job(
                destination / j["block"] / str(j["size"]) / "TabPFN-3.5"
            )
        ]
        quote = isolated(quote_dimensions, shapes) if shapes else {"estimated_units": 0}
        write_json(destination / "quote.json", quote)
        print(json.dumps(quote, indent=2))
        if args.estimate_only:
            return 0
        if quote["estimated_units"] > args.max_api_units:
            raise ValueError(
                "Development quote exceeds --max-api-units; no inference started"
            )
    results = []
    names = ["LightGBM", "XGBoost"] + ([] if args.baselines_only else ["TabPFN-3.5"])
    for job in selected:
        for name in names:
            result = isolated(
                model_job,
                name,
                job["context"],
                job["calibration"],
                job["test"],
                plan["reference"],
                FEATURE_COLUMNS,
                destination / job["block"] / str(job["size"]) / name,
                plan["tuning"].get(name, {}).get("config"),
            )
            results.append(
                {"origin": job["block"], "context_size": job["size"], **result}
            )
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(destination / "results.csv", index=False)
    write_json(
        destination / "status.json",
        {
            "status": "baselines_only" if args.baselines_only else "complete",
            "limitations": "Two predetermined origins, one context seed; exploratory, not robust sample-efficiency proof.",
        },
    )
    return 2 if args.baselines_only else 0


def predict(args):
    store = SnapshotStore(args.data)
    issued = utc(args.as_of)
    test = build_issue_rows(store, issued)
    if not test.eligible.all():
        write_json(
            args.output / "prediction_status.json",
            {
                "status": "abstained",
                "issued_at": str(issued),
                "rows": test[["region_id", "status"]].to_dict("records"),
            },
        )
        raise ValueError(
            "Fresh complete source observations unavailable; refusing a green fallback"
        )
    test["target_incidence"] = np.nan
    history = build_training_rows(store, issued)
    context, calibration, _ = select_context(history, args.context_size)
    manifest = store.manifest_frame()
    reference = SeasonalReference(store.load_snapshot(manifest.iloc[0].snapshot_id))
    identity = {
        "name": args.model,
        "source": test.snapshot_id.tolist(),
        "target": str(test.target_start.iloc[0]),
        "context": args.context_size,
        "history_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                history[
                    [
                        "region_id",
                        "target_start",
                        "target_incidence",
                        "truth_snapshot_id",
                    ]
                ],
                index=False,
            )
            .to_numpy()
            .tobytes()
        ).hexdigest(),
        "code": preparation_config(args, store)["code_sha256"],
    }
    run_id = fingerprint(identity)
    folder = args.output / "inference" / run_id
    if args.model == "TabPFN-3.5" and not completed_job(folder):
        quote = isolated(
            quote_dimensions, [("live", len(context), len(calibration) + len(test))]
        )
        write_json(folder / "quote.json", quote)
        print(json.dumps(quote, indent=2))
        if args.estimate_only:
            return 0
        if quote["estimated_units"] > args.max_api_units:
            raise ValueError(
                "Live quote exceeds --max-api-units; no hosted inference started"
            )
    elif args.estimate_only:
        print(
            json.dumps(
                {"estimated_units": 0, "reason": "cached inference or local baseline"}
            )
        )
        return 0
    result = isolated(
        model_job,
        args.model,
        context,
        calibration,
        test,
        reference,
        FEATURE_COLUMNS,
        folder,
    )
    forecasts = pd.read_parquet(folder / "predictions.parquet")
    now = pd.Timestamp.now(tz="UTC").isoformat()
    records = []
    readiness = {"ready": False, "reason": "no_matching_completed_evaluation"}
    latest = args.output / "latest_run.json"
    if latest.exists():
        previous = json.loads(latest.read_text())
        run_manifest = Path(previous.get("path", "")) / "manifest.json"
        if previous.get("status") == "complete" and run_manifest.exists():
            metadata = json.loads(run_manifest.read_text())
            if (
                metadata.get("code_sha256") == identity["code"]
                and metadata.get("context_size") == args.context_size
            ):
                readiness = previous.get("probability_readiness", readiness)
    for row in forecasts.to_dict("records"):
        source = manifest[manifest.snapshot_id == row["snapshot_id"]].iloc[0]
        version = result["model_version"]
        probability_ready = bool(
            args.model == "TabPFN-3.5"
            and readiness.get("ready")
            and row["calibration_status"] != "insufficient_events"
            and row.get("probability_threshold_in_grid", False)
        )
        records.append(
            {
                "forecast_id": f"{run_id}:{row['region_id']}",
                "model": args.model,
                "model_version": version.get("resolved", args.model),
                "location_id": row["region_id"],
                "geographic_level": "macroregion",
                "indicator": "are",
                "units": "reported ARE per 100000",
                "observed_through": str(
                    (row["latest_observed_week"] + pd.Timedelta(days=6)).date()
                ),
                "published_at": pd.Timestamp(row["data_available_at"]).isoformat(),
                "fetched_at": str(source.get("fetched_at", now)),
                "issued_at": utc(row["issue_at"]).isoformat(),
                "target_week_start": str(row["target_start"].date()),
                "target_week_end": str(row["target_end"].date()),
                "q10": float(row["q10"]),
                "q50": float(row["q50"]),
                "q90": float(row["q90"]),
                "calibrated_interval_low": float(row["interval_low"]),
                "calibrated_interval_high": float(row["interval_high"]),
                "reference_p50": float(row["reference_p50"]),
                "reference_p80": float(row["reference_p80"]),
                "reference_end": REFERENCE_END,
                "source_url": SOURCE_URL,
                "source_snapshot_id": row["snapshot_id"],
                "calibration_status": row["calibration_status"],
                "seasonal_reference_p80": float(row["seasonal_reference_p80"]),
                "probability_above_seasonal_reference": float(row["probability"])
                if probability_ready
                else None,
                "probability_status": "quantile_grid_approximation_validated_on_locked_backtest"
                if probability_ready
                else "withheld_pending_independent_calibration_validation_or_outside_grid",
                "probability_lower_bound": float(row.get("probability_lower_bound", 0)),
                "probability_upper_bound": float(row.get("probability_upper_bound", 1)),
                "probability_readiness": readiness,
            }
        )
    # Append-only issued artifacts preserve prospective evidence. A repeated
    # identical source/target reuses the original forecast id and issue time.
    ledger = args.service_output.parent / "ledger" / f"{run_id}.json"
    if not ledger.exists():
        write_json(
            ledger, {"schema_version": 1, "generated_at": now, "forecasts": records}
        )
    existing = []
    if args.service_output.exists():
        existing = json.loads(args.service_output.read_text()).get("forecasts", [])

    def key(record):
        return (record["location_id"], record["indicator"], record["target_week_start"])

    # Retain the current and immediately preceding weeks when next week's
    # bundle is issued, so current-week questions do not lose their forecast.
    today = issued.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    retention_start = today - pd.Timedelta(days=today.dayofweek + 7)
    merged = {
        key(r): r
        for r in existing
        if pd.Timestamp(r["target_week_start"]) >= retention_start
    }
    merged.update({key(r): r for r in records})
    bundle = {
        "schema_version": 1,
        "generated_at": now,
        "forecasts": sorted(
            merged.values(),
            key=lambda r: (r["target_week_start"], r["location_id"], r["indicator"]),
        ),
    }
    write_json(args.service_output, bundle)
    write_json(
        args.output / "prediction_status.json",
        {
            "status": "complete",
            "run_id": run_id,
            "model": args.model,
            "rows": len(records),
            "path": str(args.service_output),
        },
    )
    print(f"Saved {len(records)} {args.model} forecasts to {args.service_output}")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "quote", "run", "predict", "ladder"])
    p.add_argument("--data", type=Path, default=ROOT / "data/respiratory")
    p.add_argument("--output", type=Path, default=ROOT / "models/respiratory")
    p.add_argument(
        "--service-output",
        type=Path,
        default=ROOT / "models/service/forecast_bundle.json",
    )
    p.add_argument(
        "--as-of",
        help="UTC timestamp; predict defaults to now, evaluation freezes at 2026-09-17",
    )
    p.add_argument("--context-size", type=int, default=1000)
    p.add_argument("--block-weeks", type=int, choices=[4, 13, 26, 52], default=13)
    p.add_argument("--max-api-units", type=int, default=0)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Quote live prediction or development ladder without inference",
    )
    p.add_argument(
        "--include-development",
        action="store_true",
        help="Include eligible development ladder in quote",
    )
    p.add_argument(
        "--include-live",
        action="store_true",
        help="Include one current forecast in quote",
    )
    p.add_argument(
        "--model",
        choices=BASELINES + ["LightGBM", "XGBoost", "TabPFN-3.5"],
        default="TabPFN-3.5",
    )
    return p


if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=False)
    args = parser().parse_args()
    if args.as_of is None:
        instant = pd.Timestamp.now(tz="UTC")
        args.as_of = (
            instant.isoformat() if args.command == "predict" else EVALUATION_AS_OF
        )
    try:
        raise SystemExit(
            predict(args)
            if args.command == "predict"
            else ladder(args)
            if args.command == "ladder"
            else run(args)
        )
    except (ValueError, LookupError, RuntimeError) as exc:
        print(f"Respiratory benchmark stopped: {exc}")
        raise SystemExit(2)
