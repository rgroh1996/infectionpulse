"""Download and audit immutable public RKI respiratory source vintages."""

import argparse
import json
from pathlib import Path

import pandas as pd

from infectionpulse.respiratory_data import (
    SnapshotStore,
    build_backtest_rows,
    refresh_covid_context,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh", "audit", "build", "covid"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/respiratory"))
    parser.add_argument("--since", default="2023-09-01")
    parser.add_argument("--until")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--start", default="2024-08-05")
    parser.add_argument("--end", default="2026-07-27")
    args = parser.parse_args()
    if args.command == "covid":
        result = refresh_covid_context(root=args.data_dir, offline=args.offline)
        print(
            json.dumps(
                {
                    "observations": len(result["observations"]),
                    "observed_through": result["observations"][0]["observed_through"],
                }
            )
        )
        return
    store = SnapshotStore(args.data_dir)
    manifest = (
        store.ensure_history(until=args.until, since=args.since, offline=args.offline)
        if args.command == "refresh"
        else store.manifest_frame()
    )
    if manifest.empty:
        raise SystemExit("No snapshots; run scripts/respiratory_data.py refresh first")
    summary = {
        "snapshot_count": len(manifest),
        "first_available_at": str(manifest.available_at.min()),
        "last_available_at": str(manifest.available_at.max()),
        "max_release_gap_days": float(
            manifest.available_at.diff().dt.total_seconds().max() / 86400
        ),
        "source": "RKI GrippeWeb",
        "availability_basis": "GitHub commit timestamp publication proxy",
    }
    if args.command == "build":
        rows = build_backtest_rows(
            store, pd.date_range(args.start, args.end, freq="W-MON")
        )
        path = args.data_dir / "evaluation_rows.parquet"
        rows.to_parquet(path, index=False)
        summary.update(
            {
                "evaluation_rows": len(rows),
                "target_weeks": int(rows.target_start.nunique()),
                "eligible_rows": int(rows.eligible.sum()),
                "mature_truth_rows": int(rows.target_incidence.notna().sum()),
                "status_counts": rows.status.value_counts().to_dict(),
                "output": str(path),
            }
        )
    report_name = (
        "quality_report.json" if args.command == "build" else "source_report.json"
    )
    (args.data_dir / report_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
