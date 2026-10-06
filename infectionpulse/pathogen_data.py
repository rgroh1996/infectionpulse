"""Versioned RKI influenza/RSV state observations, never infection forecasts.

Both sources publish notification-week incidence, normally on Thursdays with a
Wednesday data cutoff. Recent reports remain provisional and may be revised.
An absent state/week is missing, not an inferred zero.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SOURCES = {
    "influenza": {
        "repository": "robert-koch-institut/Influenzafaelle_in_Deutschland",
        "filename": "IfSG_Influenzafaelle.tsv",
        "label": "Laborbestätigte Influenza",
    },
    "rsv": {
        "repository": "robert-koch-institut/Respiratorische_Synzytialvirusfaelle_in_Deutschland",
        "filename": "IfSG_RSVfaelle.tsv",
        "label": "RSV",
    },
}
STATE_CODES = tuple(f"{number:02d}" for number in range(1, 17))
REQUIRED_COLUMNS = {
    "Meldewoche",
    "Region",
    "Region_Id",
    "Altersgruppe",
    "Fallzahl",
    "Inzidenz",
}
UNITS = "reported cases per 100000 per calendar week"


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _session():
    session = requests.Session()
    session.headers["User-Agent"] = "InfectionPulse/1.0 (RKI open-data observations)"
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    return session


def parse_pathogen_snapshot(content: bytes) -> pd.DataFrame:
    """Preserve published all-age state incidence without summing age strata."""
    data = pd.read_csv(
        io.BytesIO(content),
        sep="\t",
        encoding="utf-8-sig",
        dtype={"Region_Id": str, "Altersgruppe": str, "Meldewoche": str},
    )
    if not REQUIRED_COLUMNS.issubset(data.columns):
        raise ValueError(
            f"RKI notification schema missing {sorted(REQUIRED_COLUMNS - set(data.columns))}"
        )
    totals = data[data.Altersgruppe.eq("00+")].copy()
    codes = totals.Region_Id.dropna()
    if (
        not codes.str.fullmatch(r"\d{2}").all()
        or not codes.isin((*STATE_CODES, "00")).all()
    ):
        raise ValueError("Unexpected RKI state identifiers")
    states = totals[totals.Region_Id.isin(STATE_CODES)].copy()
    if states.empty:
        raise ValueError("No all-age state records in RKI notification source")
    if states.duplicated(["Region_Id", "Meldewoche"]).any():
        raise ValueError("Duplicate all-age state/week observations")
    if not states.Meldewoche.str.fullmatch(r"\d{4}-W\d{2}").all():
        raise ValueError("Invalid ISO notification week")
    try:
        states["week_start"] = pd.to_datetime(
            states.Meldewoche + "-1", format="%G-W%V-%u", errors="raise"
        )
    except ValueError as exc:
        raise ValueError("Invalid ISO notification week") from exc
    if not states.week_start.dt.strftime("%G-W%V").eq(states.Meldewoche).all():
        raise ValueError("Invalid ISO notification week")
    states["observed_through"] = states.week_start + pd.Timedelta(days=6)
    states["value"] = pd.to_numeric(states.Inzidenz, errors="raise")
    states["cases"] = pd.to_numeric(states.Fallzahl, errors="raise")
    if not (
        states.value.isna()
        | (np.isfinite(states.value) & states.value.between(0, 100_000))
    ).all():
        raise ValueError("Invalid published incidence")
    if not (
        np.isfinite(states.cases) & states.cases.ge(0) & states.cases.mod(1).eq(0)
    ).all():
        raise ValueError("Invalid case counts")
    return (
        states.rename(columns={"Region_Id": "location_id", "Region": "state_name"})[
            [
                "location_id",
                "state_name",
                "week_start",
                "observed_through",
                "value",
                "cases",
            ]
        ]
        .sort_values(["week_start", "location_id"])
        .reset_index(drop=True)
    )


def build_pathogen_observations(
    content: bytes, indicator: str, provenance: dict, as_of=None
) -> dict:
    if indicator not in SOURCES:
        raise ValueError("Unsupported pathogen")
    now = _utc(as_of or pd.Timestamp.now(tz="UTC"))
    local_day = now.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    frame = parse_pathogen_snapshot(content)
    # Sunday observations are complete only once that Sunday has ended.
    complete = frame[frame.observed_through < local_day].copy()
    if complete.empty:
        raise ValueError("No completed notification weeks in source")
    latest = complete.week_start.max()
    previous = latest - pd.Timedelta(weeks=1)
    published = _utc(provenance["published_at"])
    fetched = _utc(provenance["fetched_at"])
    checked = _utc(provenance.get("latest_checked_at", provenance["fetched_at"]))
    source_valid = (
        0 <= (now - published).total_seconds() <= 10 * 86400
        and 0 <= (now - checked).total_seconds() <= 72 * 3600
    )
    selected = complete[complete.week_start.isin([previous, latest])]
    records, missing = [], []
    for week in (previous, latest):
        for code in STATE_CODES:
            matching = selected[
                (selected.location_id == code) & (selected.week_start == week)
            ]
            if matching.empty or pd.isna(matching.iloc[0].value):
                missing.append(
                    {
                        "indicator": indicator,
                        "location_id": code,
                        "geographic_level": "state",
                        "week_start": week.date().isoformat(),
                        "is_latest_week": week == latest,
                        "reason": "no_published_all_age_incidence",
                    }
                )
                continue
            row = matching.iloc[0]
            earlier = complete[
                (complete.location_id == code)
                & (complete.week_start == week - pd.Timedelta(weeks=1))
            ]
            prior = (
                None
                if earlier.empty or pd.isna(earlier.iloc[0].value)
                else float(earlier.iloc[0].value)
            )
            change = None if prior is None else float(row.value) - prior
            direction = (
                "unknown"
                if change is None
                else "rising"
                if change > 0
                else "falling"
                if change < 0
                else "stable"
            )
            observed_age = (local_day - row.observed_through).days
            records.append(
                {
                    "indicator": indicator,
                    "geographic_level": "state",
                    "kind": "observation",
                    "location_id": code,
                    "state_name": row.state_name,
                    "week_start": week.date().isoformat(),
                    "observed_through": row.observed_through.date().isoformat(),
                    "value": float(row.value),
                    "cases": int(row.cases),
                    "units": UNITS,
                    "is_latest_week": week == latest,
                    "latest_source_week": latest.date().isoformat(),
                    "previous_week_start": (week - pd.Timedelta(weeks=1))
                    .date()
                    .isoformat(),
                    "previous_value": prior,
                    "absolute_change": change,
                    "trend": direction,
                    "percentage_change": (100 * change / prior)
                    if prior is not None and prior > 0
                    else None,
                    "source_url": provenance["source_url"],
                    "source_snapshot_id": provenance["source_snapshot_id"],
                    "sha256": provenance["sha256"],
                    "published_at": published.isoformat(),
                    "fetched_at": fetched.isoformat(),
                    "latest_checked_at": checked.isoformat(),
                    "publication_basis": "GitHub data-file commit timestamp (publication proxy)",
                    "freshness": "fresh"
                    if source_valid and observed_age <= 14
                    else "stale",
                    "observation_age_days": observed_age,
                    "provisional": True,
                    "reporting_note": "Weekly reported cases; recent weeks may be revised. Not all infections and not a forecast.",
                }
            )
    return {
        "observations": records,
        "missing": missing,
        "latest_source_week": latest.date().isoformat(),
    }


def _read_snapshot(root: Path, metadata: dict) -> bytes:
    content = gzip.decompress((root / metadata["raw_file"]).read_bytes())
    if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
        raise ValueError("Pathogen snapshot checksum mismatch")
    return content


def refresh_pathogen_context(
    root="data/pathogens",
    output="data/service/pathogen_observations.json",
    offline=False,
    as_of=None,
    indicators=None,
):
    """Refresh both independently; failed sources retain explicitly dated cache.

    The output's errors/status expose partial failure. No cache fallback invents
    a new fetch/check time. Raw gzip bytes and per-snapshot metadata are immutable.
    """
    root = Path(root)
    now = _utc(as_of or pd.Timestamp.now(tz="UTC"))
    observations, missing, errors, sources = [], [], [], {}
    selected_indicators = tuple(indicators or SOURCES)
    for indicator in selected_indicators:
        if indicator not in SOURCES:
            raise ValueError("Unsupported pathogen")
        specification = SOURCES[indicator]
        latest_path = root / indicator / "latest.json"
        metadata = json.loads(latest_path.read_text()) if latest_path.exists() else None
        content = None
        source_error = None
        try:
            if offline:
                if metadata is None:
                    raise RuntimeError("No cached pathogen snapshot")
                content = _read_snapshot(root, metadata)
            else:
                session = _session()
                response = session.get(
                    f"https://api.github.com/repos/{specification['repository']}/commits",
                    params={"path": specification["filename"], "per_page": 1},
                    timeout=30,
                )
                response.raise_for_status()
                commits = response.json()
                if not isinstance(commits, list) or not commits:
                    raise ValueError("RKI commit inventory is empty")
                commit = commits[0]
                snapshot_id = commit["sha"]
                if not re.fullmatch(r"[0-9a-f]{40}", snapshot_id):
                    raise ValueError("Invalid upstream commit identifier")
                if (
                    metadata is not None
                    and metadata["source_snapshot_id"] == snapshot_id
                ):
                    content = _read_snapshot(root, metadata)
                else:
                    source_url = f"https://raw.githubusercontent.com/{specification['repository']}/{snapshot_id}/{specification['filename']}"
                    raw = session.get(source_url, timeout=60)
                    raw.raise_for_status()
                    content = raw.content
                    parse_pathogen_snapshot(content)
                    digest = hashlib.sha256(content).hexdigest()
                    relative_path = (
                        Path(indicator) / "snapshots" / f"{snapshot_id}.tsv.gz"
                    )
                    raw_path = root / relative_path
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    if (
                        raw_path.exists()
                        and hashlib.sha256(
                            gzip.decompress(raw_path.read_bytes())
                        ).hexdigest()
                        != digest
                    ):
                        raise ValueError("Immutable upstream snapshot changed")
                    if not raw_path.exists():
                        raw_path.write_bytes(gzip.compress(content, mtime=0))
                    metadata = {
                        "indicator": indicator,
                        "source_snapshot_id": snapshot_id,
                        "sha256": digest,
                        "raw_file": str(relative_path),
                        "source_url": source_url,
                        "repository_url": f"https://github.com/{specification['repository']}",
                        "published_at": _utc(
                            commit["commit"]["committer"]["date"]
                        ).isoformat(),
                        "fetched_at": now.isoformat(),
                        "license": "CC-BY-4.0",
                        "attribution": "Robert Koch-Institut",
                    }
                    snapshot_metadata = raw_path.with_suffix(".json")
                    if not snapshot_metadata.exists():
                        _write_json(snapshot_metadata, metadata)
                metadata["latest_checked_at"] = now.isoformat()
                metadata["last_check_status"] = "ok"
                _write_json(latest_path, metadata)
        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
            OSError,
            KeyError,
        ) as exc:
            source_error = str(exc)
            errors.append({"indicator": indicator, "error": source_error})
            # Re-read last successfully accepted provenance after any failed update.
            metadata = (
                json.loads(latest_path.read_text()) if latest_path.exists() else None
            )
            try:
                content = _read_snapshot(root, metadata) if metadata else None
            except (ValueError, OSError) as cache_error:
                errors.append({"indicator": indicator, "error": str(cache_error)})
                content = None
        if content is None:
            sources[indicator] = {
                "status": "unavailable",
                "error": source_error or "missing source",
            }
            continue
        parsed = build_pathogen_observations(content, indicator, metadata, now)
        observations.extend(parsed["observations"])
        missing.extend(parsed["missing"])
        sources[indicator] = {
            "status": "cached_after_error"
            if source_error
            else "cached_offline"
            if offline
            else "ok",
            "latest_source_week": parsed["latest_source_week"],
            "source_snapshot_id": metadata["source_snapshot_id"],
            "published_at": metadata["published_at"],
            "fetched_at": metadata["fetched_at"],
            "latest_checked_at": metadata.get(
                "latest_checked_at", metadata["fetched_at"]
            ),
        }
    result = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "status": "partial"
        if errors and observations
        else "unavailable"
        if errors
        else "complete",
        "observations": observations,
        "missing": missing,
        "sources": sources,
        "errors": errors,
        "notes": [
            "State-level notification incidence; no district-level or personal infection probability.",
            "Weekly publication, usually Thursday; a completed week can still have reporting delays.",
            "Omitted or null records are missing, never inferred as zero.",
            "RSV historical reports for March 2025–February 2026 were retrospectively corrected by RKI on 23 February 2026.",
        ],
    }
    _write_json(Path(output), result)
    return result


def read_pathogen_context(path, state_id: str, now=None) -> list[dict]:
    """Read two latest state indicators without network access or old-week fallback."""
    if str(state_id) not in STATE_CODES:
        raise ValueError("Expected a two-digit German state AGS")
    instant = _utc(now or pd.Timestamp.now(tz="UTC"))
    local_day = instant.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    artifact_invalid = False
    try:
        payload = json.loads(Path(path).read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("observations"), list)
            or not all(isinstance(row, dict) for row in payload["observations"])
            or not isinstance(payload.get("sources", {}), dict)
            or not all(
                isinstance(value, dict) for value in payload.get("sources", {}).values()
            )
        ):
            raise ValueError("Unsupported pathogen observation artifact")
    except OSError:
        payload = {"observations": [], "sources": {}}
    except (ValueError, TypeError):
        payload = {"observations": [], "sources": {}}
        artifact_invalid = True
    result = []
    for indicator, source in SOURCES.items():
        matches = [
            row
            for row in payload["observations"]
            if row.get("indicator") == indicator
            and row.get("location_id") == state_id
            and row.get("is_latest_week") is True
        ]
        base = {
            "indicator": indicator,
            "label": source["label"],
            "geographic_level": "state",
            "kind": "observation",
            "location_id": state_id,
            "value": None,
            "previous_value": None,
            "absolute_change": None,
            "percentage_change": None,
            "units": UNITS,
            "trend": "unknown",
            "freshness": "missing",
            "source_url": f"https://github.com/{source['repository']}",
            "reason": "no_published_latest_state_incidence",
        }
        if artifact_invalid:
            base.update(freshness="invalid", reason="invalid_observation_artifact")
        if len(matches) != 1:
            if len(matches) > 1:
                base.update(
                    freshness="invalid", reason="ambiguous_latest_state_incidence"
                )
            result.append(base)
            continue
        row = dict(matches[0])
        row["label"] = source["label"]
        row["source_status"] = (
            payload.get("sources", {}).get(indicator, {}).get("status", "unknown")
        )
        try:
            published, fetched = _utc(row["published_at"]), _utc(row["fetched_at"])
            checked = _utc(row.get("latest_checked_at", row["fetched_at"]))
            observed = pd.Timestamp(row["observed_through"])
            started = pd.Timestamp(row["week_start"])
            if (
                not np.isfinite(float(row["value"]))
                or float(row["value"]) < 0
                or any(
                    pd.isna(stamp)
                    for stamp in (published, fetched, checked, observed, started)
                )
                or any(stamp > instant for stamp in (published, fetched, checked))
                or started.weekday() != 0
                or observed != started + pd.Timedelta(days=6)
                or observed >= local_day
                or row.get("latest_source_week") != row["week_start"]
                or row.get("geographic_level") != "state"
            ):
                raise ValueError("Inconsistent or future-dated pathogen observation")
            for field in ("previous_value", "absolute_change", "percentage_change"):
                value = row.get(field)
                if value is not None and not np.isfinite(float(value)):
                    raise ValueError("Nonfinite pathogen trend")
            # Reject any other nonfinite extension fields as well. Invalid
            # records are replaced below, never forwarded to strict API JSON.
            json.dumps(row, allow_nan=False)
            fresh = (
                instant - published <= pd.Timedelta(days=10)
                and instant - checked <= pd.Timedelta(hours=72)
                and (local_day - observed).days <= 14
            )
            row["freshness"] = "fresh" if fresh else "stale"
        except (ValueError, KeyError, TypeError, OverflowError):
            row = {
                **base,
                "freshness": "invalid",
                "reason": "invalid_observation_or_provenance",
            }
        result.append(row)
    return result
