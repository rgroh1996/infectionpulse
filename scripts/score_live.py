"""Score immutable live forecasts only when their public labels have matured."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from infectionpulse.respiratory_data import SnapshotStore, truth_for_target
from infectionpulse.service import ForecastRecord

ROOT = Path(__file__).resolve().parents[1]


def score(ledger, store, as_of):
    truths, rows = {}, []
    for path in sorted(Path(ledger).glob("*.json")):
        bundle = json.loads(path.read_text())
        generated = (
            datetime.fromisoformat(bundle["generated_at"])
            if bundle.get("generated_at")
            else None
        )
        for item in bundle["forecasts"]:
            forecast = ForecastRecord.model_validate(item)
            if forecast.indicator != "are":
                continue
            target = forecast.target_week_start.isoformat()
            if target not in truths:
                truths[target] = truth_for_target(store, target, as_of=as_of)
            truth = truths[target]
            match = truth[truth.region_id == forecast.location_id]
            issued_date = forecast.issued_at.astimezone(
                ZoneInfo("Europe/Berlin")
            ).date()
            status = (
                "issued_after_target_started"
                if issued_date >= forecast.target_week_start
                else "pending_mature_label"
            )
            if status == "pending_mature_label":
                if generated is None or generated.tzinfo is None:
                    status = "generation_time_unknown"
                elif (
                    generated.astimezone(ZoneInfo("Europe/Berlin")).date()
                    >= forecast.target_week_start
                ):
                    status = "generated_after_target_started"
            row = {
                "forecast_id": forecast.forecast_id,
                "region_id": forecast.location_id,
                "model": forecast.model,
                "issued_at": forecast.issued_at.isoformat(),
                "target_week_start": target,
                "status": status,
            }
            if not match.empty and status == "pending_mature_label":
                outcome = match.iloc[0]
                y = float(outcome.target_incidence)
                row.update(
                    status="scored",
                    target_incidence=y,
                    absolute_error=abs(y - forecast.q50),
                    squared_error=(y - forecast.q50) ** 2,
                    raw_interval_covered=forecast.q10 <= y <= forecast.q90,
                    truth_snapshot_id=outcome.truth_snapshot_id,
                    truth_available_at=outcome.truth_available_at.isoformat(),
                )
            rows.append(row)
    if rows:
        frame = pd.DataFrame(rows)
        if frame.forecast_id.duplicated().any():
            raise ValueError("Duplicate immutable forecast IDs in ledger")
        scored = frame[frame.status == "scored"]
    else:
        frame = pd.DataFrame(
            columns=["forecast_id", "region_id", "target_week_start", "status"]
        )
        scored = frame
    report = {
        "as_of": str(as_of),
        "issued_forecasts": len(frame),
        "scored_forecasts": len(scored),
        "unique_region_weeks": int(
            frame[["region_id", "target_week_start"]].drop_duplicates().shape[0]
        ),
        "statuses": frame.status.value_counts().to_dict(),
        "mae": float(scored.absolute_error.mean()) if len(scored) else None,
        "rmse": float(np.sqrt(scored.squared_error.mean())) if len(scored) else None,
        "raw_coverage80": float(scored.raw_interval_covered.mean())
        if len(scored)
        else None,
        "limitations": "Repeated issues for one region/week are correlated. Only pre-target forecasts are scored. Labels are first public releases at least 35 days after target Sunday.",
    }
    return frame, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=datetime.now(UTC).isoformat())
    args = parser.parse_args()
    frame, report = score(
        ROOT / "models/service/ledger",
        SnapshotStore(ROOT / "data/respiratory"),
        args.as_of,
    )
    destination = ROOT / "models/service"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "prospective_scores.csv", index=False)
    temporary = destination / "prospective_scores.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination / "prospective_scores.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
