"""Record public service artifacts or replay a clearly labeled historical example.

No network, LLM calls, or model inference. Replay time is explicit and never used
by the live API or dashboard.
"""

import argparse
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

from infectionpulse.activity_policy import Activity
from infectionpulse.service import AssessmentRequest, ForecastService

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/replay"


def record():
    files = {
        "forecast_bundle.json": ROOT / "models/service/forecast_bundle.json",
        "source_status.json": ROOT / "data/respiratory/source_status.json",
        "covid_observations.json": ROOT / "data/service/covid_observations.json",
    }
    if not all(p.exists() for p in files.values()):
        raise SystemExit(
            "Generate a real forecast and refresh both sources before recording."
        )
    DEMO.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        shutil.copyfile(source, DEMO / name)
    (DEMO / "metadata.json").write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "mode": "historical_replay",
                "notice": "Real public forecasts replayed at their recorded time; not current advice.",
            },
            indent=2,
        )
        + "\n"
    )


def replay(brief=False):
    if not (DEMO / "metadata.json").exists():
        raise SystemExit(
            "No recorded example. Run scripts/demo.py --record after generating a real forecast."
        )
    metadata = json.loads((DEMO / "metadata.json").read_text())
    recorded = datetime.fromisoformat(metadata["recorded_at"])
    service = ForecastService(
        bundle_path=DEMO / "forecast_bundle.json",
        locations_path=ROOT / "examples/locations.json",
        now=lambda: recorded,
        source_status_path=DEMO / "source_status.json",
        observations_path=DEMO / "covid_observations.json",
    )
    available = service.get_forecast("12054")["forecasts"]
    if not available:
        raise SystemExit("Replay contains no supported future forecast.")
    day = min(date.fromisoformat(f["target_week_start"]) for f in available)
    work = service.assess_activity(
        AssessmentRequest(
            home_location_id="11000",
            destination_location_id="12054",
            activity_date=day,
            activity=Activity(
                category="work",
                setting="indoor",
                duration_minutes=360,
                crowding="crowded",
                ventilation="unknown",
                commute_mode="public_transit",
                commute_minutes=35,
                commute_crowding="crowded",
            ),
        )
    )
    outdoor = service.assess_activity(
        AssessmentRequest(
            home_location_id="11000",
            destination_location_id="12054",
            activity_date=day,
            activity=Activity(
                category="social",
                setting="outdoor",
                duration_minutes=60,
                crowding="uncrowded",
                ventilation="good",
                commute_mode="bike",
                commute_minutes=20,
            ),
        )
    )
    result = {
        **metadata,
        "indoor_work_and_transit": work,
        "outdoor_alternative": outdoor,
    }
    if brief:
        print(
            f"Historical replay recorded {metadata['recorded_at']} — not current advice."
        )
        print(f"Target date: {day.isoformat()}")
        for label, response in [
            ("Office and commute", work),
            ("Outdoor alternative", outdoor),
        ]:
            assessment = response["assessment"]
            print(
                f"{label} [{assessment['traffic_light']}]: {assessment['recommendation']}"
            )
    else:
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()
    if args.record:
        record()
    replay(args.brief)
