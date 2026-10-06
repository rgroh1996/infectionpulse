"""Daily public-data refresh, with a bounded Friday forecast job.

This command never installs a scheduler. Run from cron/systemd in Europe/Berlin.
"""

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def forecast_due(now):
    local = now.astimezone(ZoneInfo("Europe/Berlin"))
    return local.weekday() == 4 and local.hour >= 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-now",
        action="store_true",
        help="Explicit manual bootstrap outside the Friday schedule",
    )
    parser.add_argument(
        "--max-api-units",
        type=int,
        default=0,
        help="Maximum quoted units for this forecast; zero prevents spending",
    )
    args = parser.parse_args()
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    with (runtime / "refresh.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Refresh already running; skipped.")
            return 0
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        report = {
            "started_at": now.isoformat(),
            "source_refresh": "pending",
            "forecast": "not_due",
        }
        refresh = subprocess.run(
            [sys.executable, str(ROOT / "scripts/respiratory_data.py"), "refresh"],
            cwd=ROOT,
        )
        report["source_refresh"] = "ok" if refresh.returncode == 0 else "failed"
        covid = subprocess.run(
            [sys.executable, str(ROOT / "scripts/respiratory_data.py"), "covid"],
            cwd=ROOT,
        )
        report["covid_refresh"] = "ok" if covid.returncode == 0 else "failed"
        pathogens = subprocess.run(
            [sys.executable, str(ROOT / "scripts/pathogen_data.py"), "refresh"],
            cwd=ROOT,
        )
        report["pathogen_refresh"] = "ok" if pathogens.returncode == 0 else "failed"
        code = refresh.returncode or covid.returncode or pathogens.returncode
        if refresh.returncode == 0 and (args.forecast_now or forecast_due(now)):
            inference = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/respiratory_benchmark.py"),
                    "predict",
                    "--as-of",
                    datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
                    "--max-api-units",
                    str(args.max_api_units),
                ],
                cwd=ROOT,
            )
            code = code or inference.returncode
            report["forecast"] = (
                "ok" if inference.returncode == 0 else "failed_or_budget_blocked"
            )
        if refresh.returncode == 0:
            scoring = subprocess.run(
                [sys.executable, str(ROOT / "scripts/score_live.py")], cwd=ROOT
            )
            report["prospective_scoring"] = (
                "ok" if scoring.returncode == 0 else "failed"
            )
            code = code or scoring.returncode
        report["completed_at"] = datetime.now(ZoneInfo("Europe/Berlin")).isoformat()
        destination = runtime / "refresh_status.json"
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(destination)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
