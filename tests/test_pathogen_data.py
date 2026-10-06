import gzip
import hashlib
import json

import pandas as pd
import pytest
import requests

from infectionpulse import pathogen_data as pathogens

NOW = "2026-09-18T08:00:00Z"


def source(rows=None):
    if rows is None:
        rows = [
            ["2026-W36", "Berlin", "11", "00+", 5, 1.5],
            ["2026-W37", "Berlin", "11", "00+", 10, 3.0],
            ["2026-W37", "Berlin", "11", "00-14", 9, 9.0],
            ["2026-W37", "Deutschland", "00", "00+", 100, 10.0],
            ["2026-W38", "Berlin", "11", "00+", 1, 0.3],
            ["2026-W36", "Bremen", "04", "00+", 0, 0.0],
        ]
    columns = [
        "Meldewoche",
        "Region",
        "Region_Id",
        "Altersgruppe",
        "Fallzahl",
        "Inzidenz",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .to_csv(sep="\t", index=False)
        .encode("utf-8-sig")
    )


def provenance():
    return {
        "published_at": "2026-09-17T07:00:00Z",
        "fetched_at": NOW,
        "source_url": "https://example.test/rki",
        "source_snapshot_id": "a" * 40,
        "sha256": hashlib.sha256(source()).hexdigest(),
    }


def test_all_age_state_schema_and_bom():
    parsed = pathogens.parse_pathogen_snapshot(source())
    assert len(parsed) == 4
    assert set(parsed.location_id) == {"11", "04"}
    assert parsed.loc[parsed.week_start.eq("2026-09-07"), "value"].iloc[0] == 3
    with pytest.raises(ValueError, match="schema"):
        pathogens.parse_pathogen_snapshot(source().replace(b"Meldewoche", b"week"))
    with pytest.raises(ValueError, match="Duplicate"):
        pathogens.parse_pathogen_snapshot(source() + source().split(b"\n", 1)[1])
    with pytest.raises(ValueError, match="week"):
        pathogens.parse_pathogen_snapshot(source().replace(b"2026-W37", b"2026-W54"))


def test_complete_weeks_exact_prior_and_missing_not_zero():
    output = pathogens.build_pathogen_observations(
        source(), "influenza", provenance(), NOW
    )
    records = output["observations"]
    berlin = next(
        r for r in records if r["location_id"] == "11" and r["is_latest_week"]
    )
    assert berlin["week_start"] == "2026-09-07"
    assert berlin["observed_through"] == "2026-09-13"
    assert berlin["value"] == 3.0 and berlin["previous_value"] == 1.5
    assert berlin["trend"] == "rising" and berlin["percentage_change"] == 100
    assert berlin["freshness"] == "fresh"
    assert not any(r["location_id"] == "04" and r["is_latest_week"] for r in records)
    assert any(
        r["location_id"] == "04" and r["is_latest_week"] for r in output["missing"]
    )
    assert next(r for r in records if r["location_id"] == "04")["value"] == 0


def test_sunday_is_not_complete_until_monday():
    output = pathogens.build_pathogen_observations(
        source(), "rsv", provenance(), "2026-09-13T20:00Z"
    )
    assert output["latest_source_week"] == "2026-08-31"


def test_gap_and_zero_previous_do_not_invent_growth():
    raw = source(
        [
            ["2026-W35", "Berlin", "11", "00+", 1, 0.1],
            ["2026-W37", "Berlin", "11", "00+", 5, 0.5],
        ]
    )
    row = pathogens.build_pathogen_observations(raw, "rsv", provenance(), NOW)[
        "observations"
    ][0]
    assert row["trend"] == "unknown" and row["previous_value"] is None
    raw = source(
        [
            ["2026-W36", "Berlin", "11", "00+", 0, 0],
            ["2026-W37", "Berlin", "11", "00+", 5, 0.5],
        ]
    )
    row = pathogens.build_pathogen_observations(raw, "rsv", provenance(), NOW)[
        "observations"
    ][-1]
    assert row["trend"] == "rising" and row["percentage_change"] is None


def test_read_rechecks_freshness_and_never_previous_fallback(tmp_path):
    payload = pathogens.build_pathogen_observations(
        source(), "influenza", provenance(), NOW
    )
    payload["schema_version"] = 1
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(payload))
    berlin = pathogens.read_pathogen_context(path, "11", NOW)
    assert berlin[0]["freshness"] == "fresh" and berlin[1]["freshness"] == "missing"
    assert pathogens.read_pathogen_context(path, "04", NOW)[0]["freshness"] == "missing"
    assert (
        pathogens.read_pathogen_context(path, "11", "2026-10-01T10:00Z")[0]["freshness"]
        == "stale"
    )
    assert (
        pathogens.read_pathogen_context(path, "11", "2026-09-16T10:00Z")[0]["freshness"]
        == "invalid"
    )
    with pytest.raises(ValueError, match="state"):
        pathogens.read_pathogen_context(path, "11000", NOW)


class Response:
    def __init__(self, content=None, payload=None):
        self.content, self.payload = content, payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_immutable_refresh_offline_and_failed_poll_provenance(tmp_path, monkeypatch):
    class Session:
        def get(self, url, **kwargs):
            if "api.github.com" in url:
                return Response(
                    payload=[
                        {
                            "sha": "a" * 40,
                            "commit": {"committer": {"date": "2026-09-17T07:00:00Z"}},
                        }
                    ]
                )
            return Response(content=source())

    monkeypatch.setattr(pathogens, "_session", Session)
    output_path = tmp_path / "output.json"
    first = pathogens.refresh_pathogen_context(
        tmp_path / "cache", output_path, as_of=NOW
    )
    assert first["status"] == "complete" and len(first["observations"]) == 6
    later = "2026-09-19T08:00:00Z"
    offline = pathogens.refresh_pathogen_context(
        tmp_path / "cache", output_path, offline=True, as_of=later
    )
    assert (
        offline["sources"]["rsv"]["latest_checked_at"] == pd.Timestamp(NOW).isoformat()
    )

    class Failed:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    monkeypatch.setattr(pathogens, "_session", Failed)
    failed = pathogens.refresh_pathogen_context(
        tmp_path / "cache", output_path, as_of=later
    )
    assert failed["status"] == "partial"
    assert (
        failed["sources"]["rsv"]["latest_checked_at"] == pd.Timestamp(NOW).isoformat()
    )
    raw = tmp_path / "cache/rsv/snapshots" / ("a" * 40 + ".tsv.gz")
    raw.write_bytes(gzip.compress(b"tampered"))
    corrupted = pathogens.refresh_pathogen_context(
        tmp_path / "cache", output_path, offline=True, as_of=later
    )
    assert corrupted["sources"]["rsv"]["status"] == "unavailable"
    assert all(r["indicator"] == "influenza" for r in corrupted["observations"])


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"schema_version": 1, "observations": [None]},
        {"schema_version": 1, "observations": ["bad"]},
        {"schema_version": 1, "observations": [], "sources": []},
        {"schema_version": 1, "observations": [], "sources": {"rsv": None}},
    ],
)
def test_malformed_artifact_is_safe_json(tmp_path, payload):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(payload))
    rows = pathogens.read_pathogen_context(path, "11", NOW)
    assert len(rows) == 2 and all(row["freshness"] == "invalid" for row in rows)
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize(
    "field",
    [
        "value",
        "previous_value",
        "absolute_change",
        "percentage_change",
        "unexpected_extension",
    ],
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_observation_and_trend_never_reach_api_json(
    tmp_path, field, bad_value
):
    payload = pathogens.build_pathogen_observations(
        source(), "influenza", provenance(), NOW
    )
    payload["schema_version"] = 1
    row = next(
        row
        for row in payload["observations"]
        if row["location_id"] == "11" and row["is_latest_week"]
    )
    row[field] = bad_value
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(payload))
    rows = pathogens.read_pathogen_context(path, "11", NOW)
    assert rows[0]["freshness"] == "invalid"
    assert all(
        rows[0][name] is None
        for name in ("value", "previous_value", "absolute_change", "percentage_change")
    )
    json.dumps(rows, allow_nan=False)
