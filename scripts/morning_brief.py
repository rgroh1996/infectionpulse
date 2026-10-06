"""Write a local evidence-bearing morning brief. No messages are sent."""

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from infectionpulse.morning_brief import MorningProfile, build_brief, save_brief
from infectionpulse.service import ForecastService

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "examples/morning_profile.json"
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Activity date; default tomorrow in Berlin (replay: first saved target week)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output stem; default .runtime/morning-brief/latest or replay",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full machine-readable brief instead of prose",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Explicit historical replay using recorded fixtures; never current advice",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Run public-data refresh before reading saved forecasts",
    )
    parser.add_argument(
        "--max-api-units",
        type=int,
        default=0,
        help="Explicit refresh forecast budget; default zero prevents spending",
    )
    args = parser.parse_args(argv)
    if args.max_api_units < 0:
        parser.error("--max-api-units must be nonnegative")
    if args.replay and args.refresh:
        parser.error("Historical replay cannot refresh current data")
    if args.max_api_units and not args.refresh:
        parser.error("--max-api-units requires --refresh")
    try:
        profile = MorningProfile.model_validate_json(args.profile.read_text())
    except (OSError, ValueError):
        parser.error(
            "Profile missing or invalid; use examples/morning_profile.json as the schema"
        )
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True, mode=0o700)
    output = args.output or runtime / "morning-brief" / (
        "replay" if args.replay else "latest"
    )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Morning brief already running; skipped.", file=sys.stderr)
            return 3
        refresh_status = "not_requested"
        if args.refresh:
            refreshed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/refresh.py"),
                    "--max-api-units",
                    str(args.max_api_units),
                ],
                cwd=ROOT,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            refresh_status = "ok" if refreshed.returncode == 0 else "failed"
        generated = datetime.now(UTC)
        day = args.date or (
            generated.astimezone(ZoneInfo("Europe/Berlin")).date() + timedelta(days=1)
        )
        if args.replay:
            folder = ROOT / "examples/replay"
            metadata = json.loads((folder / "metadata.json").read_text())
            recorded = datetime.fromisoformat(metadata["recorded_at"])
            service = ForecastService(
                bundle_path=folder / "forecast_bundle.json",
                locations_path=ROOT / "examples/locations.json",
                now=lambda: recorded,
                source_status_path=folder / "source_status.json",
                observations_path=folder / "covid_observations.json",
                pathogens_path=folder / "pathogen_observations.json",
            )
            if args.date is None:
                bundle = json.loads((folder / "forecast_bundle.json").read_text())
                day = min(
                    date.fromisoformat(f["target_week_start"])
                    for f in bundle["forecasts"]
                )
        else:
            service = ForecastService(now=lambda: generated)
        brief = build_brief(
            service,
            profile,
            day,
            generated,
            mode="historical_replay" if args.replay else "live",
            refresh_status=refresh_status,
        )
        save_brief(brief, output)
        print(
            json.dumps(brief, indent=2, allow_nan=False)
            if args.json
            else brief["text"],
            end="\n" if args.json else "",
        )
        return (
            1
            if refresh_status == "failed"
            else (0 if brief["status"] == "available" else 2)
        )


if __name__ == "__main__":
    raise SystemExit(main())
