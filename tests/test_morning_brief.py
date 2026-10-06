import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from infectionpulse.morning_brief import MorningProfile, build_brief, save_brief
from infectionpulse.service import ForecastService

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "examples/replay"


def profile():
    return MorningProfile.model_validate_json(
        (ROOT / "examples/morning_profile.json").read_text()
    )


def service(now=None, **paths):
    recorded = datetime.fromisoformat(
        json.loads((REPLAY / "metadata.json").read_text())["recorded_at"]
    )
    return ForecastService(
        bundle_path=paths.get("bundle_path", REPLAY / "forecast_bundle.json"),
        locations_path=ROOT / "examples/locations.json",
        now=lambda: now or recorded,
        source_status_path=REPLAY / "source_status.json",
        observations_path=REPLAY / "covid_observations.json",
        pathogens_path=REPLAY / "pathogen_observations.json",
    )


def test_real_recorded_forecast_drives_evidence_and_policy():
    saved = service()
    result = build_brief(
        saved, profile(), date(2026, 9, 21), datetime.now(UTC), mode="historical_replay"
    )
    assert result["traffic_light"] == "red"
    assert result["precaution_score"]["value"] == 85
    assert "7,881" in result["text"]
    assert "2026-09-13" in result["text"]
    assert "HISTORICAL REPLAY — not current advice" in result["text"]
    assert result["community_probability"]["status"] == "withheld"
    assert result["personal_infection_probability"] is None
    assert result["llm_used"] is False
    assert result["disease_observations"] == []  # Never mix live pathogens into replay.
    assert "not your infection probability" in result["text"]


def test_activity_change_recomputes_recommendation():
    saved = service()
    person = profile()
    person.activity.shared_space = False
    person.activity.commute_mode = "none"
    person.activity.commute_minutes = 0
    result = build_brief(
        saved, person, date(2026, 9, 21), datetime.now(UTC), mode="historical_replay"
    )
    assert result["traffic_light"] == "green"
    assert result["precaution_score"]["value"] == 30
    assert "home office" not in result["recommendation"]


@pytest.mark.parametrize("case", ["stale", "missing", "past", "unsupported_date"])
def test_unavailable_evidence_never_produces_safe_advice_or_score(tmp_path, case):
    now = datetime(2026, 9, 21, tzinfo=UTC) if case == "stale" else None
    paths = {"bundle_path": tmp_path / "absent.json"} if case == "missing" else {}
    day = date(2026, 9, 21)
    if case == "past":
        day = date(2026, 9, 1)
    if case == "unsupported_date":
        day = date(2027, 1, 1)
    result = build_brief(service(now, **paths), profile(), day, datetime.now(UTC))
    assert result["status"] == "unavailable"
    assert result["traffic_light"] == "grey"
    assert result["precaution_score"]["value"] is None
    assert "85/100" not in result["text"]
    assert "not enough current forecast evidence" in result["text"]
    assert "TabPFN median" not in result["text"]


def test_atomic_artifacts_private_and_authoritative(tmp_path):
    result = build_brief(
        service(),
        profile(),
        date(2026, 9, 21),
        datetime.now(UTC),
        mode="historical_replay",
    )
    path = tmp_path / "brief"
    save_brief(result, path)
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["text"] == path.with_suffix(".txt").read_text()
    assert os.stat(path.with_suffix(".json")).st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".brief-*"))


def test_cli_replay_is_valid_json_and_separate_output(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/morning_brief.py"),
            "--replay",
            "--json",
            "--output",
            str(tmp_path / "replay"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["mode"] == "historical_replay"
    assert payload["activity_date"] == "2026-09-21"
    assert (tmp_path / "replay.json").exists()


def test_cli_rejects_refresh_in_replay():
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/morning_brief.py"),
            "--replay",
            "--refresh",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    assert "Historical replay cannot refresh" in run.stderr


def test_cli_lock_prevents_same_output_writers(tmp_path):
    output = tmp_path / "busy"
    with output.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/morning_brief.py"),
                "--replay",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
    assert run.returncode == 3
    assert not output.with_suffix(".json").exists()


def test_failed_refresh_disclosed_even_if_cached_evidence_usable():
    result = build_brief(
        service(),
        profile(),
        date(2026, 9, 21),
        datetime.now(UTC),
        refresh_status="failed",
    )
    assert "Today's refresh failed" in result["text"]
    assert result["refresh_status"] == "failed"
