"""Compare unchanged saved forecasts with currently available public observations.

This is a preliminary latest-release check, separate from scripts/score_live.py's
35-day mature-label evaluation. No downloads, fitting or hosted inference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from infectionpulse.pathogen_data import _read_snapshot, parse_pathogen_snapshot
from infectionpulse.respiratory_data import SnapshotStore, utc

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "south": "South",
    "east": "East",
    "north_west": "North-West",
    "central_west": "Central-West",
}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess_record(
    forecast, bundle_generated, observations, source, as_of, persistence=None
):
    """Require pre-target generation and exact region/week labels; never fill zeros."""
    now = utc(as_of)
    target = pd.Timestamp(forecast["target_week_start"])
    end = pd.Timestamp(forecast["target_week_end"])
    if target.weekday() != 0 or end != target + pd.Timedelta(days=6):
        raise ValueError("Forecast must cover a complete Monday-Sunday week")
    start_instant = target.tz_localize("Europe/Berlin").tz_convert("UTC")
    # Calendar completion must respect the German timezone across DST changes.
    completed = (
        (target + pd.Timedelta(days=7)).tz_localize("Europe/Berlin").tz_convert("UTC")
    )
    issued_value = forecast.get("issued_at") or forecast.get("issue_at")
    generated_value = forecast.get("generated_at") or bundle_generated
    row = {
        "forecast_id": forecast["forecast_id"],
        "indicator": forecast["indicator"],
        "location_id": forecast["location_id"],
        "name": NAMES.get(
            forecast["location_id"], forecast.get("state_name", forecast["location_id"])
        ),
        "issued_at": issued_value,
        "generated_at": generated_value,
        "target_week_start": str(target.date()),
        "target_week_end": str(end.date()),
        "q10": float(forecast["q10"]),
        "q50": float(forecast["q50"]),
        "q90": float(forecast["q90"]),
        "status": "pending_publication",
        "observed": None,
        "absolute_error": None,
        "squared_error": None,
        "interval_covered": None,
        "persistence": persistence,
        "persistence_absolute_error": None,
        "predicted_high": None,
        "observed_high": None,
        "outcome_snapshot_id": source["source_snapshot_id"],
        "outcome_published_at": source["published_at"],
        "outcome_source_url": source["source_url"],
        "mature_label_earliest_at": (target + pd.Timedelta(days=42))
        .tz_localize("Europe/Berlin")
        .isoformat(),
    }
    q = np.array([row["q10"], row["q50"], row["q90"]])
    if not np.isfinite(q).all() or (q < 0).any() or (np.diff(q) < 0).any():
        raise ValueError("Invalid saved forecast quantiles")
    if any(
        value is None or pd.Timestamp(value).tzinfo is None
        for value in (issued_value, generated_value)
    ):
        row["status"] = "generation_or_issue_time_unknown"
        return row
    if utc(issued_value) >= start_instant or utc(generated_value) >= start_instant:
        row["status"] = "forecast_created_after_target_started"
        return row
    if utc(generated_value) < utc(issued_value):
        raise ValueError("Generation precedes declared issue")
    if utc(generated_value) > now or utc(issued_value) > now:
        row["status"] = "forecast_not_yet_created_at_cutoff"
        return row
    if utc(source["published_at"]) > now:
        raise ValueError("Outcome release is after the evaluation cutoff")
    if now < completed:
        row["status"] = "target_week_not_complete"
        return row
    if utc(source["published_at"]) < completed:
        return row
    matching = observations[
        observations.location_id.eq(forecast["location_id"])
        & observations.week_start.eq(target)
    ]
    if len(matching) > 1:
        raise ValueError("Ambiguous duplicate outcome")
    if matching.empty or pd.isna(matching.iloc[0].value):
        if observations.week_start.max() >= target:
            row["status"] = "missing_region_outcome"
        return row
    value = float(matching.iloc[0].value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Invalid observed incidence")
    row.update(
        status="preliminary_latest_release",
        observed=value,
        absolute_error=abs(row["q50"] - value),
        squared_error=(row["q50"] - value) ** 2,
        interval_covered=bool(row["q10"] <= value <= row["q90"]),
    )
    if persistence is not None:
        if not np.isfinite(persistence) or persistence < 0:
            raise ValueError("Invalid persistence baseline")
        row["persistence_absolute_error"] = abs(persistence - value)
    if "reference_p80" in forecast:
        row["predicted_high"] = bool(row["q50"] >= forecast["reference_p80"])
        row["observed_high"] = bool(value >= forecast["reference_p80"])
    return row


def summarize(rows):
    results = []
    for indicator in sorted({row["indicator"] for row in rows}):
        selected = [row for row in rows if row["indicator"] == indicator]
        valid = [r for r in selected if r["status"] == "preliminary_latest_release"]
        paired = [r for r in valid if r["persistence_absolute_error"] is not None]

        def mean(key, group):
            return float(np.mean([r[key] for r in group])) if group else None

        mae = mean("absolute_error", valid)
        persistence_mae = mean("persistence_absolute_error", paired)
        paired_mae = mean("absolute_error", paired)
        results.append(
            {
                "indicator": indicator,
                "saved_forecasts": len(selected),
                "observed_forecasts": len(valid),
                "pending_or_excluded": len(selected) - len(valid),
                "target_weeks": len({r["target_week_start"] for r in valid}),
                "mae": mae,
                "rmse": float(np.sqrt(mean("squared_error", valid))) if valid else None,
                "covered": sum(r["interval_covered"] for r in valid),
                "raw_coverage80": mean("interval_covered", valid),
                "persistence_paired_rows": len(paired),
                "persistence_mae": persistence_mae,
                "paired_tabpfn_mae": paired_mae,
                "mae_reduction_vs_persistence": (1 - paired_mae / persistence_mae)
                if persistence_mae
                else None,
                "high_burden_detected": sum(
                    r["predicted_high"] and r["observed_high"]
                    for r in valid
                    if r["observed_high"] is not None
                ),
                "high_burden_observed": sum(
                    r["observed_high"] for r in valid if r["observed_high"] is not None
                ),
            }
        )
    return results


def evaluate(root, as_of):
    root = Path(root)
    paths = sorted((root / "models/service/ledger").glob("*.json")) + sorted(
        (root / "models/state_pathogens/live").glob("*/forecast_bundle.json")
    )
    if not paths:
        raise ValueError("No saved live forecasts")
    hashes = {str(p.relative_to(root)): checksum(p) for p in paths}
    store = SnapshotStore(root / "data/respiratory")
    manifest = store.manifest_frame()
    available = manifest[manifest.available_at <= utc(as_of)]
    if available.empty:
        raise ValueError("No outcome release available by evaluation cutoff")
    last = available.iloc[-1]
    observations = {
        "are": store.load_snapshot(last.snapshot_id).rename(
            columns={"region_id": "location_id", "incidence": "value"}
        )
    }
    sources = {
        "are": {
            "source_snapshot_id": last.snapshot_id,
            "published_at": last.available_at.isoformat(),
            "sha256": last.sha256,
            "source_url": last.source_url,
        }
    }
    forecasts, rows, seen = [], [], set()
    for path in paths:
        bundle = json.loads(path.read_text())
        for forecast in bundle["forecasts"]:
            if forecast["forecast_id"] in seen:
                raise ValueError("Duplicate saved forecast ID")
            seen.add(forecast["forecast_id"])
            indicator = forecast["indicator"]
            if indicator not in observations:
                metadata = json.loads(
                    (root / "data/pathogens" / indicator / "latest.json").read_text()
                )
                observations[indicator] = parse_pathogen_snapshot(
                    _read_snapshot(root / "data/pathogens", metadata)
                )
                sources[indicator] = metadata
            persistence = None
            if indicator == "are":
                original = manifest[
                    manifest.snapshot_id.eq(forecast["source_snapshot_id"])
                ]
                if len(original) != 1 or original.iloc[0].available_at > utc(
                    forecast["issued_at"]
                ):
                    raise ValueError(
                        "Persistence baseline requires the actual issue-time source"
                    )
                original_data = store.load_snapshot(forecast["source_snapshot_id"])
                history = original_data[
                    original_data.region_id.eq(forecast["location_id"])
                    & original_data.week_start.eq(
                        pd.Timestamp(forecast["observed_through"])
                        - pd.Timedelta(days=6)
                    )
                ]
                if len(history) != 1:
                    raise ValueError("Missing original last observation")
                persistence = float(history.iloc[0].incidence)
            rows.append(
                assess_record(
                    forecast,
                    bundle.get("generated_at"),
                    observations[indicator],
                    sources[indicator],
                    as_of,
                    persistence,
                )
            )
            forecasts.append(
                {
                    "ledger_file": str(path.relative_to(root)),
                    "bundle_generated_at": bundle.get("generated_at"),
                    "forecast": forecast,
                }
            )
    if hashes != {str(p.relative_to(root)): checksum(p) for p in paths}:
        raise ValueError("Saved forecasts changed during validation")
    return {
        "evaluated_at": utc(as_of).isoformat(),
        "kind": "preliminary_latest_release_check",
        "forecast_files_sha256": hashes,
        "outcome_sources": sources,
        "saved_forecasts": forecasts,
        "rows": rows,
        "summary": summarize(rows),
        "limitations": [
            "Latest reported outcomes remain provisional; scripts/score_live.py's original 35-day mature-label gate is unchanged.",
            "Four ARE regions in one week are correlated, not four independent trials or evidence of general superiority.",
            "Persistence is reconstructed from the exact original input release, not a newly trained competing model.",
            "Missing or unpublished outcomes are pending, never treated as zero.",
            "This evaluates community illness forecasts, not individual infection risk or the benefit of staying home.",
        ],
    }


def render_plot(rows, output, evaluated_at):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        r
        for r in rows
        if r["indicator"] == "are" and r["status"] == "preliminary_latest_release"
    ]
    if not selected:
        return
    selected.sort(key=lambda r: (r["target_week_start"], r["name"]))
    fig, ax = plt.subplots(figsize=(10.4, 4.8), facecolor="#f5f8fa")
    ax.set_facecolor("#f5f8fa")
    y = np.arange(len(selected))
    ax.hlines(
        y,
        [r["q10"] for r in selected],
        [r["q90"] for r in selected],
        color="#91bac5",
        linewidth=8,
        alpha=0.6,
        label="Saved 10th–90th percentile interval",
    )
    ax.scatter(
        [r["q50"] for r in selected],
        y,
        color="#126b80",
        s=70,
        label="Saved TabPFN median",
        zorder=3,
    )
    ax.scatter(
        [r["observed"] for r in selected],
        y,
        color="#b64f2b",
        marker="D",
        s=60,
        label="Reported outcome (preliminary)",
        zorder=4,
    )
    labels = [
        r["name"]
        if len({r["target_week_start"] for r in selected}) == 1
        else f"{r['name']} · {r['target_week_start']}"
        for r in selected
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_ylim(len(selected) - 0.5, -1.05)
    ax.set_xlim(0, max(max(r["q90"], r["observed"]) for r in selected) * 1.12)
    ax.set_xlabel("Weekly ARE illnesses per 100,000")
    ax.grid(axis="x", color="#dfe7ec")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="upper left", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0, -0.18)
    )
    fig.suptitle(
        "Saved forecasts versus the following reported week",
        x=0.055,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#17394c",
    )
    starts = sorted({r["target_week_start"] for r in selected})
    ends = sorted({r["target_week_end"] for r in selected})
    generated_dates = sorted({str(r["generated_at"])[:10] for r in selected})
    fig.text(
        0.055,
        0.885,
        f"Target: {starts[0]}–{ends[-1]} · Saved {generated_dates[0]} · Checked {str(evaluated_at)[:10]}",
        fontsize=10,
        color="#4b6476",
    )
    fig.text(
        0.055,
        0.018,
        f"Preliminary observations, {len(starts)} target week(s). Separate mature-label rules remain unchanged.",
        fontsize=9,
        color="#4b6476",
    )
    fig.subplots_adjust(left=0.16, right=0.97, top=0.82, bottom=0.27)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep a dated audit copy as well as the current report.
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    dated = (
        output
        / "live_checks"
        / f"{result['evaluated_at'][:10]}-{hashlib.sha256(encoded.encode()).hexdigest()[:12]}.json"
    )
    dated.parent.mkdir(exist_ok=True)
    dated.write_text(encoded)
    (output / "live_forecast_check.json").write_text(encoded)
    pd.DataFrame(result["rows"]).to_csv(output / "live_forecast_check.csv", index=False)
    lines = [
        "# First check of forecasts saved before their target week",
        "",
        f"Checked **{result['evaluated_at']}** against the latest available RKI releases. No model was rerun, and the saved forecasts were unchanged (file checksums in the JSON report). This is a **preliminary latest-release check**, separate from the mature-label benchmark.",
        "",
        "![Saved forecasts and preliminary reported outcomes](live_forecast_check.png)",
        "",
        "## Regional respiratory illness (ARE)",
        "",
        "| Region | Saved median | Saved 10th–90th percentiles | Reported outcome | Absolute error | Inside interval? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        if row["indicator"] == "are" and row["observed"] is not None:
            lines.append(
                f"| {row['name']} | {row['q50']:,.0f} | {row['q10']:,.0f}–{row['q90']:,.0f} | {row['observed']:,.0f} | {row['absolute_error']:,.0f} | {'Yes' if row['interval_covered'] else 'No'} |"
            )
    for item in result["summary"]:
        lines += [
            "",
            f"**{item['indicator'].upper()}: {item['observed_forecasts']}/{item['saved_forecasts']} outcomes available**, {item['pending_or_excluded']} pending or excluded.",
        ]
        if item["mae"] is not None:
            lines += [
                f"MAE: **{item['mae']:,.1f}**; RMSE: **{item['rmse']:,.1f}**; outcomes within saved 10th–90th intervals: **{item['covered']}/{item['observed_forecasts']}**."
            ]
        if item["persistence_mae"] is not None:
            lines += [
                f"On {item['persistence_paired_rows']} matched outcomes, carrying forward the last originally available observation gives MAE **{item['persistence_mae']:,.1f}**, versus TabPFN **{item['paired_tabpfn_mae']:,.1f}** ({item['mae_reduction_vs_persistence']:.1%} lower). This is a simple persistence reference, not a new LightGBM/XGBoost comparison.",
                f"The saved median flagged {item['high_burden_detected']}/{item['high_burden_observed']} regions whose reported outcome exceeded the fixed historical high-burden threshold.",
            ]
    available_are = [
        r
        for r in result["rows"]
        if r["indicator"] == "are" and r["observed"] is not None
    ]
    if available_are:
        worst = max(available_are, key=lambda r: r["absolute_error"])
        lines += [
            "",
            f"The largest miss was **{worst['name']}**: predicted {worst['q50']:,.0f}, reported {worst['observed']:,.0f} illnesses per 100,000. {'Its outcome was outside the saved interval.' if not worst['interval_covered'] else 'Its outcome was within the saved interval.'} All {len(available_are)} observed rows are shown; no region was removed because of its error.",
        ]
    lines += ["", "## Outcome provenance"]
    for key, source in result["outcome_sources"].items():
        lines += [
            "",
            f"- {key.upper()}: published {source['published_at']}; [exact public release]({source['source_url']}); SHA256 `{source['sha256']}`.",
        ]
    pending = [r for r in result["rows"] if r["status"] == "pending_publication"]
    if pending:
        pending_labels = sorted(
            {
                "{indicator} {target_week_start}–{target_week_end}".format(**r)
                for r in pending
            }
        )
        lines += [
            "",
            f"Still awaiting publication: {len(pending)} forecasts, for "
            + ", ".join(pending_labels)
            + ". The available releases do not yet contain these outcomes. No earlier week is substituted.",
        ]
    earliest = sorted({r["mature_label_earliest_at"] for r in result["rows"]})
    lines += [
        "",
        "Mature-label eligibility begins at "
        + ", ".join(earliest)
        + "; scoring still requires a public release on or after the respective cutoff.",
        "",
        "## What this supports",
        "",
        "This is evidence that forecasts were saved before a later reported outcome and can be checked transparently. It is one preliminary forecast week, not proof of general superiority, calibrated personal risk or prevented infections.",
        "",
    ]
    lines.extend(f"- {note}" for note in result["limitations"])
    lines += [
        "",
        "Reproduce after public-data refresh: `python scripts/check_live_forecasts.py`. The original mature-label scorer remains `python scripts/score_live.py`.",
        "",
    ]
    (output / "live_forecast_check.md").write_text("\n".join(lines))
    render_plot(
        result["rows"], output / "live_forecast_check.png", result["evaluated_at"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=pd.Timestamp.now(tz="UTC").isoformat())
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if utc(args.as_of) > pd.Timestamp.now(tz="UTC"):
        parser.error("Cannot evaluate future outcomes")
    result = evaluate(ROOT, args.as_of)
    write_report(result, args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
