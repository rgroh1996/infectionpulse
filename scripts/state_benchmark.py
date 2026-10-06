#!/usr/bin/env python3
"""Frozen, publication-aware 16-state flu/RSV forecasting pilot.

prepare --download pins public historical releases; run --baselines-only is
local. quote sends dimensions only; hosted run requires --max-api-units.
predict writes experimental state forecasts, separate from the ARE service.
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

from infectionpulse.pathogen_data import (
    _read_snapshot,
    _utc,
    _write_json,
    parse_pathogen_snapshot,
)
from infectionpulse.respiratory_model import atomic_dump
from infectionpulse.state_forecasting import (
    DEVELOPMENT,
    FEATURES,
    MODEL_NAMES,
    TARGETS,
    build_job,
    fetch_vintage,
    freeze_protocol,
    issue_for_target,
    run_model,
    summarize,
    tree_tuning,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/pathogens"


def isolated(function, *args, **kwargs):
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return executor.submit(function, *args, **kwargs).result()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def code_hashes():
    files = [
        "scripts/state_benchmark.py",
        "infectionpulse/state_forecasting.py",
        "infectionpulse/respiratory_model.py",
        "infectionpulse/pathogen_data.py",
    ]
    return {
        file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in files
    }


def add_labels(job, labels):
    if job["status"] != "ready":
        return
    query = job["query"].drop(columns="target_incidence")
    query = query.merge(
        labels[["location_id", "week_start", "value"]].rename(
            columns={"week_start": "target_start", "value": "target_incidence"}
        ),
        on=["location_id", "target_start"],
        how="left",
        validate="one_to_one",
    )
    job["query"] = query


def prepare(args):
    frozen = freeze_protocol(args.reports / "state_protocol.json")
    jobs, sources, truths, tuning = {}, {}, [], {}
    for indicator in ("influenza", "rsv"):
        latest = json.loads((DATA / indicator / "latest.json").read_text())
        # This exact label release enters the run identity, never an evolving latest pointer.
        truth = parse_pathogen_snapshot(_read_snapshot(DATA, latest))
        truth = truth.rename(
            columns={"week_start": "target_start", "value": "target_incidence"}
        )
        mature = (
            truth.observed_through + pd.Timedelta(days=35)
            <= _utc(latest["published_at"]).tz_localize(None).normalize()
        )
        truth = truth[mature & truth.target_start.isin(pd.to_datetime(TARGETS))].copy()
        truth["indicator"] = indicator
        truths.append(truth)
        sources[f"{indicator}/truth"] = latest
        dev_truth, dev_provenance = fetch_vintage(
            DATA, indicator, issue_for_target(TARGETS[0]), download=args.download
        )
        if dev_truth is None:
            raise ValueError("No pretest vintage for development labels")
        sources[f"{indicator}/development_truth"] = dev_provenance
        development = []
        for target in DEVELOPMENT + TARGETS:
            issue = issue_for_target(target)
            observations, provenance = fetch_vintage(
                DATA, indicator, issue, download=args.download
            )
            print(
                f"Prepare {indicator} target={target}: {provenance.get('source_snapshot_id', 'unavailable')}",
                flush=True,
            )
            job = build_job(observations, provenance, target)
            job["indicator"] = indicator
            sources[f"{indicator}/{target}"] = provenance
            if target in DEVELOPMENT:
                add_labels(job, dev_truth)
                development.append(job)
            else:
                jobs[f"{indicator}/{target}"] = job
        tuning[indicator] = {
            "LightGBM": isolated(tree_tuning, development),
            "LightGBM full-history": isolated(
                tree_tuning, development, full_history=True
            ),
        }
    identity = {
        "protocol": frozen["protocol"],
        "sources": {
            k: {
                field: v.get(field)
                for field in ("sha256", "source_snapshot_id", "published_at")
            }
            for k, v in sources.items()
        },
        "code_hashes": code_hashes(),
    }
    run_id = digest(identity)
    folder = args.output / "runs" / run_id
    plan = {
        "run_id": run_id,
        "identity": identity,
        "jobs": jobs,
        "tuning": tuning,
        "sources": sources,
    }
    atomic_dump(plan, folder / "plan.joblib")
    pd.concat(truths).to_parquet(folder / "truth.parquet", index=False)
    _write_json(
        folder / "manifest.json",
        {
            "run_id": run_id,
            **identity,
            "tuning": tuning,
            "job_status": {key: job["status"] for key, job in jobs.items()},
            "planned_rows_per_model": len(TARGETS) * 16 * 2,
        },
    )
    _write_json(args.output / "latest_run.json", {"run_id": run_id})
    print(f"Prepared state run {run_id}")
    return folder, plan


def load_plan(args):
    latest = json.loads((args.output / "latest_run.json").read_text())
    folder = args.output / "runs" / latest["run_id"]
    plan = joblib.load(folder / "plan.joblib")
    if plan["identity"]["code_hashes"] != code_hashes():
        raise ValueError("State/model source changed since prepare; prepare a new run")
    return folder, plan


def cached(folder, job, model):
    path = folder / "predictions.parquet"
    if not path.exists():
        return False
    rows = pd.read_parquet(path)
    query = job["query"]
    return (
        len(rows) == len(query)
        and rows.model.eq(model).all()
        and rows.location_id.tolist() == query.location_id.tolist()
        and rows.target_start.tolist() == query.target_start.tolist()
        and np.isfinite(rows.q50).all()
        and (
            model in ("Persistence", "Seasonal naive")
            or (
                np.isfinite(rows[["q10", "q90"]]).all().all()
                and (rows.q10 <= rows.q50).all()
                and (rows.q50 <= rows.q90).all()
            )
        )
    )


def quote_shapes(shapes):
    from infectionpulse.tabpfn_api import require_api_token

    require_api_token()
    from tabpfn_client import estimate_cost
    from tabpfn_client.config import get_access_token

    get_access_token()

    items = []
    for name, context, query in shapes:
        quote = estimate_cost(
            np.empty((context, len(FEATURES) * 2)),
            np.empty((query, len(FEATURES) * 2)),
            model_version="v3.5",
            n_estimators=8,
            operation="predict",
        )
        items.append(
            {
                "job": name,
                "context_rows": context,
                "query_rows": query,
                **quote.model_dump(mode="json"),
            }
        )
    return {
        "estimated_units": sum(item["estimated_cost"] for item in items),
        "jobs": items,
        "note": "Estimate, not a hard billing cap; retries may consume additional units.",
    }


def run(args, folder, plan):
    names = MODEL_NAMES[:-1] if args.baselines_only else MODEL_NAMES
    shapes = [
        (key, len(job["context"]), len(job["query"]))
        for key, job in plan["jobs"].items()
        if job["status"] == "ready"
        and not cached(folder / key / "TabPFN-3.5", job, "TabPFN-3.5")
    ]
    if args.command == "quote" or "TabPFN-3.5" in names:
        quote = (
            isolated(quote_shapes, shapes)
            if shapes
            else {"estimated_units": 0, "jobs": []}
        )
        _write_json(folder / "quote.json", quote)
        print(json.dumps(quote, indent=2), flush=True)
        if args.command == "quote":
            return
        if quote["estimated_units"] > args.max_api_units:
            raise SystemExit(
                "Quote exceeds --max-api-units; no hosted inference started"
            )
    for key, job in plan["jobs"].items():
        if job["status"] != "ready":
            continue
        for name in names:
            destination = folder / key / name
            if cached(destination, job, name):
                continue
            config = plan["tuning"][job["indicator"]].get(name, {}).get("selected")
            print(f"Fit {key} {name}, query={len(job['query'])}", flush=True)
            isolated(run_model, job, name, config, destination)
    report(args, folder, plan)


def report(args, folder, plan):
    rows, coverage = [], []
    for key, job in plan["jobs"].items():
        for name in MODEL_NAMES:
            count = 0
            if job["status"] == "ready" and cached(folder / key / name, job, name):
                frame = pd.read_parquet(folder / key / name / "predictions.parquet")
                frame["indicator"] = job["indicator"]
                rows.append(frame)
                count = len(frame)
            coverage.append(
                {
                    "indicator": job["indicator"],
                    "target": job["target"],
                    "model": name,
                    "planned": 16,
                    "predicted": count,
                    "status": job["status"]
                    if job["status"] != "ready"
                    else "complete"
                    if count == 16
                    else "partial_or_not_run",
                }
            )
    if not rows:
        raise ValueError("No completed state predictions")
    predictions = pd.concat(rows, ignore_index=True)
    ready_jobs = sum(job["status"] == "ready" for job in plan["jobs"].values())
    # Never label a partially-run model as a full benchmark competitor.
    complete = [
        name
        for name in MODEL_NAMES
        if sum(
            cached(folder / key / name, job, name)
            for key, job in plan["jobs"].items()
            if job["status"] == "ready"
        )
        == ready_jobs
    ]
    selected = predictions[predictions.model.isin(complete)]
    summary, scored = summarize(
        selected, pd.read_parquet(folder / "truth.parquet"), complete
    )
    args.reports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.reports / "state_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.reports / "state_coverage.csv", index=False)
    scored.to_parquet(folder / "scored_predictions.parquet", index=False)
    per_origin = (
        scored.assign(absolute_error=lambda f: (f.q50 - f.target_incidence).abs())
        .groupby(["indicator", "target_start", "model"], as_index=False)
        .agg(mae=("absolute_error", "mean"), rows=("absolute_error", "size"))
    )
    per_origin.to_csv(args.reports / "state_per_origin.csv", index=False)
    records = json.loads(summary.to_json(orient="records"))
    result = {
        "run_id": plan["run_id"],
        "status": "complete" if set(complete) == set(MODEL_NAMES) else "partial_models",
        "complete_models": complete,
        "planned_rows_per_model": len(TARGETS) * 32,
        "matched_rows_per_model": len(
            scored.drop_duplicates(["indicator", "location_id", "target_start"])
        ),
        "summary": records,
        "limitations": plan["identity"]["protocol"]["limitations"],
        "truth_snapshots": {
            k: v for k, v in plan["identity"]["sources"].items() if k.endswith("/truth")
        },
    }
    _write_json(args.reports / "state_results.json", result)
    lines = [
        "# Experimental state forecast pilot",
        "",
        f"Run `{plan['run_id']}`. Status: **{result['status']}**. Matched coverage: {result['matched_rows_per_model']}/{result['planned_rows_per_model']} planned state-weeks per model. Actual issue-time public releases; 12 monthly origins, 16 states, separately evaluated diseases. These results do not activate personal risk scores or validated probabilities.",
        "",
        "| Disease / regime | Model | Rows | MAE | RMSE | Raw 80% coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in records:
        coverage_text = "—" if r["coverage_80"] is None else f"{r['coverage_80']:.1%}"
        lines.append(
            f"| {r['indicator']} / {r['regime']} | {r['model']} | {r['rows']} | {r['mae']:.3f} | {r['rmse']:.3f} | {coverage_text} |"
        )
    lines += [
        "",
        "Errors are notification cases per 100,000. Point-only baselines have no interval. See state_coverage.csv for all planned/abstained rows and state_per_origin.csv for seasonal variability. Twelve dates do not establish reliable statistical superiority.",
        "",
        "RSV is a secondary stress test: releases before 23 February 2026 were affected by RKI's documented query error. Pre/post correction results must not be conflated.",
        "",
    ]
    lines.extend(f"- {note}" for note in result["limitations"])
    lines += [
        "",
        "Sources: [RKI influenza](https://github.com/robert-koch-institut/Influenzafaelle_in_Deutschland), [RKI RSV](https://github.com/robert-koch-institut/Respiratorische_Synzytialvirusfaelle_in_Deutschland). Full release checksums and fixed protocol are in the run manifest.",
        "",
    ]
    (args.reports / "state_report.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))


def predict(args, plan):
    now = pd.Timestamp.now(tz="UTC")
    day = now.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    target = day + pd.Timedelta(days=7 - day.weekday())
    jobs, statuses = {}, {}
    for indicator in ("influenza", "rsv"):
        metadata = json.loads((DATA / indicator / "latest.json").read_text())
        checked = _utc(metadata.get("latest_checked_at", metadata["fetched_at"]))
        if not pd.Timedelta(0) <= now - checked <= pd.Timedelta(hours=72):
            statuses[indicator] = "source_check_stale_or_future"
            continue
        observations = parse_pathogen_snapshot(_read_snapshot(DATA, metadata))
        job = build_job(observations, metadata, target, issue=now)
        statuses[indicator] = job["status"]
        if job["status"] == "ready":
            jobs[indicator] = job
    key = digest(
        {
            "training_run": plan["run_id"],
            "target": target,
            "training_maturity_day": day,
            "sources": {k: v["provenance"]["sha256"] for k, v in jobs.items()},
        }
    )
    folder = args.output / "live" / key
    shapes = [
        (k, len(j["context"]), len(j["query"]))
        for k, j in jobs.items()
        if not cached(folder / k, j, "TabPFN-3.5")
    ]
    quote = isolated(quote_shapes, shapes) if shapes else {"estimated_units": 0}
    print(json.dumps(quote, indent=2), flush=True)
    if args.estimate_only:
        print(json.dumps({"target": str(target.date()), "status": statuses}, indent=2))
        return
    if quote["estimated_units"] > args.max_api_units:
        raise SystemExit("Quote exceeds --max-api-units; no hosted inference started")
    forecasts = []
    for indicator, job in jobs.items():
        if not cached(folder / indicator, job, "TabPFN-3.5"):
            isolated(run_model, job, "TabPFN-3.5", None, folder / indicator)
        result = pd.read_parquet(folder / indicator / "predictions.parquet")
        model = json.loads((folder / indicator / "model_metadata.json").read_text())
        for row in result.to_dict(orient="records"):
            forecasts.append(
                {
                    **row,
                    "indicator": indicator,
                    "geographic_level": "state",
                    "kind": "experimental_forecast",
                    "forecast_id": f"state-{key}-{indicator}-{row['location_id']}",
                    "target_week_start": str(target.date()),
                    "target_week_end": str((target + pd.Timedelta(days=6)).date()),
                    "observed_through": str(
                        (row["latest_observed_week"] + pd.Timedelta(days=6)).date()
                    ),
                    "published_at": job["provenance"]["published_at"],
                    "source_url": job["provenance"]["source_url"],
                    "latest_checked_at": job["provenance"].get(
                        "latest_checked_at", job["provenance"]["fetched_at"]
                    ),
                    "generated_at": model["inference_generated_at"],
                    "units": "weekly reported cases per 100000",
                    "interval_status": "raw_uncalibrated",
                    "model_version": model["api_version"],
                    "personal_infection_probability": None,
                }
            )
    # JSON only public data; timestamps and missing seasonal values encoded strictly.
    clean = (
        json.loads(pd.DataFrame(forecasts).to_json(orient="records", date_format="iso"))
        if forecasts
        else []
    )
    artifact = {
        "schema_version": 1,
        "experimental": True,
        "generated_at": now.isoformat(),
        "target_week_start": str(target.date()),
        "source_status": statuses,
        "forecasts": clean,
        "drives_precaution_score": False,
    }
    _write_json(args.output / "forecast_bundle.json", artifact)
    _write_json(folder / "forecast_bundle.json", artifact)
    print(f"Saved {len(clean)} experimental state forecasts: {statuses}")


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "quote", "run", "report", "predict"]
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--max-api-units", type=int, default=0)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "models/state_pathogens")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    folder, plan = load_plan(args)
    if args.command == "predict":
        predict(args, plan)
    elif args.command == "report":
        report(args, folder, plan)
    else:
        run(args, folder, plan)


if __name__ == "__main__":
    main()
