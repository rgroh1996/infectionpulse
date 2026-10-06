"""Versioned RKI GrippeWeb observations and issue-time-correct forecasting rows.

Git commit timestamps are historical publication proxies, not clinical event dates.
Pre-repository training features are explicitly marked reconstructed; evaluation
never reconstructs unavailable snapshots. No individual infection labels exist.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPOSITORY = "robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts"
DATA_FILE = "GrippeWeb_Daten_des_Wochenberichts.tsv"
SOURCE_URL = f"https://github.com/{REPOSITORY}"
REGIONS = ("Sueden", "Osten", "Norden (West)", "Mitte (West)")
REGION_IDS = dict(zip(REGIONS, ("south", "east", "north_west", "central_west")))
STATE_REGIONS = {
    "01": "Norden (West)",
    "02": "Norden (West)",
    "03": "Norden (West)",
    "04": "Norden (West)",
    "05": "Mitte (West)",
    "06": "Mitte (West)",
    "07": "Mitte (West)",
    "08": "Sueden",
    "09": "Sueden",
    "10": "Mitte (West)",
    "11": "Osten",
    "12": "Osten",
    "13": "Osten",
    "14": "Osten",
    "15": "Osten",
    "16": "Osten",
}
FEATURE_COLUMNS = [
    "incidence",
    *[f"inc_lag_{i}" for i in range(1, 5)],
    "latest_change",
    "rolling_mean3",
    "sin_week",
    "cos_week",
    "region_code",
    "respondents",
    "inc_lag_52",
    "seasonal_naive_incidence",
    "horizon_weeks",
]
SCHEMA_COLUMNS = {
    "Meldungen",
    "Saison",
    "Erkrankung",
    "Altersgruppe",
    "Region",
    "Kalenderwoche",
    "Inzidenz",
}


def utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def monday(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Europe/Berlin").tz_localize(None)
    return timestamp.normalize() - pd.Timedelta(days=timestamp.weekday())


def issue_for_target(target_start) -> pd.Timestamp:
    target = monday(target_start)
    if pd.Timestamp(target_start).tz_localize(None).normalize() != target:
        raise ValueError("Target must start on Monday")
    return (
        (target - pd.Timedelta(days=3) + pd.Timedelta(hours=10))
        .tz_localize("Europe/Berlin")
        .tz_convert("UTC")
    )


def next_target(issue_at) -> pd.Timestamp:
    return monday(utc(issue_at)) + pd.Timedelta(days=7)


def region_for_district(ags: str) -> str:
    ags = str(ags)
    if len(ags) != 5 or not ags.isdigit() or ags[:2] not in STATE_REGIONS:
        raise ValueError("Expected a five-digit German district AGS")
    return REGION_IDS[STATE_REGIONS[ags[:2]]]


def parse_snapshot(content: bytes) -> pd.DataFrame:
    """Validate official schema and retain all-age regional/national ARE only."""
    data = pd.read_csv(io.BytesIO(content), sep="\t", dtype={"Altersgruppe": str})
    if set(data.columns) != SCHEMA_COLUMNS:
        raise ValueError(f"Unexpected GrippeWeb schema: {list(data.columns)}")
    selected = data[(data.Erkrankung == "ARE") & (data.Altersgruppe == "00+")].copy()
    expected = set(REGIONS) | {"Bundesweit"}
    if set(selected.Region) != expected:
        raise ValueError("Missing or unexpected GrippeWeb all-age ARE regions")
    if selected.duplicated(["Region", "Kalenderwoche"]).any():
        raise ValueError("Duplicate region-week observations")
    selected["week_start"] = pd.to_datetime(
        selected.Kalenderwoche + "-1", format="%G-W%V-%u", errors="raise"
    )
    if not selected.week_start.dt.strftime("%G-W%V").eq(selected.Kalenderwoche).all():
        raise ValueError("Invalid ISO week")
    selected["incidence"] = pd.to_numeric(selected.Inzidenz, errors="raise")
    selected["respondents"] = pd.to_numeric(selected.Meldungen, errors="raise")
    if not selected.incidence.between(0, 100_000).all():
        raise ValueError("Invalid ARE incidence")
    if not (selected.respondents.ge(1) & selected.respondents.mod(1).eq(0)).all():
        raise ValueError("Invalid respondent count")
    selected["region_id"] = selected.Region.map(REGION_IDS).fillna("national")
    return (
        selected[["region_id", "Region", "week_start", "incidence", "respondents"]]
        .sort_values(["week_start", "region_id"])
        .reset_index(drop=True)
    )


def _session():
    session = requests.Session()
    session.headers["User-Agent"] = "InfectionPulse/1.0 (RKI open-data research)"
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    return session


class SnapshotStore:
    def __init__(self, root: str | Path = "data/respiratory"):
        self.root = Path(root)
        self._manifest_stamp = None
        self._manifest_cache = None

    def manifest_frame(self) -> pd.DataFrame:
        path = self.root / "manifest.json"
        if not path.exists():
            return pd.DataFrame(
                columns=[
                    "snapshot_id",
                    "available_at",
                    "observed_through",
                    "sha256",
                    "raw_file",
                ]
            )
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if stamp == self._manifest_stamp:
            return self._manifest_cache.copy()
        result = pd.DataFrame(json.loads(path.read_text())["snapshots"])
        if not result.empty:
            result["available_at"] = pd.to_datetime(
                result.available_at, utc=True, format="mixed"
            )
            result["observed_through"] = pd.to_datetime(
                result.observed_through, format="mixed"
            )
            result = result.sort_values(["available_at", "snapshot_id"]).reset_index(
                drop=True
            )
        self._manifest_stamp, self._manifest_cache = stamp, result
        return result

    def _save_manifest(self, records):
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "source": SOURCE_URL,
            "license": "CC-BY-4.0",
            "availability_basis": "GitHub data-file commit timestamp (publication proxy)",
            "snapshots": sorted(
                records, key=lambda x: (str(x["available_at"]), x["snapshot_id"])
            ),
        }
        temp = self.root / "manifest.json.tmp"
        temp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        temp.replace(self.root / "manifest.json")

    def ingest(self, snapshot_id: str, available_at, content: bytes, fetched_at=None):
        """Ingest immutable bytes; useful for both live downloads and test fixtures."""
        frame = parse_snapshot(content)
        digest = hashlib.sha256(content).hexdigest()
        current = self.manifest_frame()
        existing = current[current.snapshot_id == snapshot_id]
        if not existing.empty:
            if existing.iloc[0].sha256 != digest:
                raise ValueError("Immutable snapshot changed")
            return existing.iloc[0].to_dict()
        raw = self.root / "snapshots" / f"{snapshot_id}.tsv.gz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        if (
            raw.exists()
            and hashlib.sha256(gzip.decompress(raw.read_bytes())).hexdigest() != digest
        ):
            raise ValueError("Immutable cached snapshot changed")
        if not raw.exists():
            raw.write_bytes(gzip.compress(content, mtime=0))
        record = {
            "snapshot_id": snapshot_id,
            "available_at": utc(available_at).isoformat(),
            "fetched_at": utc(fetched_at or pd.Timestamp.now(tz="UTC")).isoformat(),
            "observed_through": frame.week_start.max().date().isoformat(),
            "sha256": digest,
            "raw_file": str(raw.relative_to(self.root)),
            "source_url": f"https://raw.githubusercontent.com/{REPOSITORY}/{snapshot_id}/{DATA_FILE}",
        }
        records = current.to_dict("records") + [record]
        self._save_manifest(records)
        return record

    @lru_cache(maxsize=256)
    def load_snapshot(self, snapshot_id: str) -> pd.DataFrame:
        manifest = self.manifest_frame()
        selected = manifest[manifest.snapshot_id == snapshot_id]
        if selected.empty:
            raise KeyError(snapshot_id)
        record = selected.iloc[0]
        content = gzip.decompress((self.root / record.raw_file).read_bytes())
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ValueError("Cached snapshot checksum mismatch")
        return parse_snapshot(content)

    def ensure_history(self, until=None, since="2023-09-01", offline=False):
        if offline:
            manifest = self.manifest_frame()
            if manifest.empty:
                raise RuntimeError("No cached respiratory source snapshots")
            return manifest
        session = _session()
        commits = []
        for page in range(1, 101):
            params = {
                "path": DATA_FILE,
                "since": utc(since).isoformat(),
                "per_page": 100,
                "page": page,
            }
            if until is not None:
                params["until"] = utc(until).isoformat()
            response = session.get(
                f"https://api.github.com/repos/{REPOSITORY}/commits",
                params=params,
                timeout=45,
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise ValueError("Unexpected GitHub commit response")
            commits.extend(batch)
            if len(batch) < 100:
                break
        known = set(self.manifest_frame().snapshot_id)
        missing = [item for item in commits if item["sha"] not in known]

        def fetch(item):
            response = _session().get(
                f"https://raw.githubusercontent.com/{REPOSITORY}/{item['sha']}/{DATA_FILE}",
                timeout=60,
            )
            response.raise_for_status()
            parse_snapshot(response.content)
            return item, response.content

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for item, content in executor.map(fetch, missing):
                self.ingest(item["sha"], item["commit"]["committer"]["date"], content)
        manifest = self.manifest_frame()
        if manifest.empty:
            raise RuntimeError("No source snapshots retrieved")
        status = {
            "latest_checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "last_check_status": "ok",
            "latest_snapshot_id": manifest.iloc[-1].snapshot_id,
            "latest_available_at": manifest.iloc[-1].available_at.isoformat(),
        }
        status_path = self.root / "source_status.json.tmp"
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        status_path.replace(self.root / "source_status.json")
        return manifest

    def snapshot_at(self, issue_at):
        manifest = self.manifest_frame()
        available = manifest[manifest.available_at <= utc(issue_at)]
        if available.empty:
            raise LookupError("No source snapshot was available at issue time")
        return available.iloc[-1].to_dict()


def freshness(available_at, observed_week, issue_at) -> dict:
    issued = utc(issue_at)
    release_age = (issued - utc(available_at)).total_seconds() / 86400
    observed_end = (pd.Timestamp(observed_week) + pd.Timedelta(days=6)).tz_localize(
        "Europe/Berlin"
    )
    observation_age = (issued - observed_end.tz_convert("UTC")).total_seconds() / 86400
    return {
        "source_age_days": release_age,
        "observation_age_days": observation_age,
        "fresh": 0 <= release_age <= 10 and 0 <= observation_age <= 14,
    }


def build_issue_rows(
    store: SnapshotStore, issue_at, target_start=None, allow_reconstructed=False
) -> pd.DataFrame:
    issued = utc(issue_at)
    target = next_target(issued) if target_start is None else monday(target_start)
    if target <= monday(issued):
        raise ValueError("Only next complete/future ISO weeks are supported")
    reconstructed = False
    try:
        snapshot = store.snapshot_at(issued)
    except LookupError:
        if not allow_reconstructed or store.manifest_frame().empty:
            raise
        snapshot = store.manifest_frame().iloc[0].to_dict()
        reconstructed = True
    data = store.load_snapshot(snapshot["snapshot_id"])
    latest_allowed = monday(issued) - pd.Timedelta(weeks=1)
    rows = []
    for index, (region, region_id) in enumerate(REGION_IDS.items()):
        history = (
            data[(data.region_id == region_id) & (data.week_start <= latest_allowed)]
            .set_index("week_start")
            .sort_index()
        )
        if history.empty:
            continue
        latest = history.index.max()
        current = float(history.loc[latest, "incidence"])
        values = {
            f"inc_lag_{lag}": float(
                history.incidence.get(latest - pd.Timedelta(weeks=lag), np.nan)
            )
            for lag in (1, 2, 3, 4, 52)
        }
        recent = history.incidence.reindex(
            pd.date_range(latest - pd.Timedelta(weeks=2), latest, freq="W-MON")
        )
        fresh = freshness(snapshot["available_at"], latest, issued)
        feature_ok = np.isfinite(
            [current, *values.values(), recent.mean(skipna=False)]
        ).all()
        status = (
            "eligible"
            if feature_ok and (fresh["fresh"] or reconstructed)
            else "missing_history"
            if not feature_ok
            else "source_stale"
        )
        week = target.isocalendar().week
        rows.append(
            {
                "region_id": region_id,
                "region_name": region,
                "region_code": index,
                "issue_at": issued,
                "target_start": target,
                "target_end": target + pd.Timedelta(days=6),
                "snapshot_id": snapshot["snapshot_id"],
                "data_available_at": utc(snapshot["available_at"]),
                "latest_observed_week": latest,
                "feature_provenance": "reconstructed_pre_archive"
                if reconstructed
                else "historical_vintage",
                "status": status,
                "eligible": status == "eligible",
                **fresh,
                "incidence": current,
                **values,
                "respondents": float(history.loc[latest, "respondents"]),
                "seasonal_naive_incidence": float(
                    history.incidence.get(target - pd.Timedelta(weeks=52), np.nan)
                ),
                "horizon_weeks": (target - latest).days / 7,
                "latest_change": current - values["inc_lag_1"],
                "rolling_mean3": recent.mean(skipna=False),
                "sin_week": np.sin(2 * np.pi * week / 52.1775),
                "cos_week": np.cos(2 * np.pi * week / 52.1775),
            }
        )
    return pd.DataFrame(rows)


def truth_for_target(store: SnapshotStore, target_start, as_of=None) -> pd.DataFrame:
    target = monday(target_start)
    # At least 35 full days after the target week's Sunday has ended.
    maturity = (
        (target + pd.Timedelta(days=7 + 35))
        .tz_localize("Europe/Berlin")
        .tz_convert("UTC")
    )
    manifest = store.manifest_frame()
    candidates = manifest[manifest.available_at >= maturity]
    if as_of is not None:
        candidates = candidates[candidates.available_at <= utc(as_of)]
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "region_id",
                "target_incidence",
                "truth_available_at",
                "truth_snapshot_id",
            ]
        )
    record = candidates.iloc[0]
    data = store.load_snapshot(record.snapshot_id)
    selected = data[
        (data.week_start == target) & data.region_id.isin(REGION_IDS.values())
    ].copy()
    selected = selected.rename(columns={"incidence": "target_incidence"})
    selected["truth_available_at"] = record.available_at
    selected["truth_snapshot_id"] = record.snapshot_id
    return selected[
        ["region_id", "target_incidence", "truth_available_at", "truth_snapshot_id"]
    ]


def build_backtest_rows(
    store: SnapshotStore, target_starts, allow_reconstructed=False, truth_as_of=None
) -> pd.DataFrame:
    parts = []
    for target in target_starts:
        try:
            features = build_issue_rows(
                store, issue_for_target(target), target, allow_reconstructed
            )
        except LookupError:
            if allow_reconstructed:
                raise
            # Missing vintages must remain visible, not silently become retrospective rows.
            features = pd.DataFrame(
                {
                    "region_id": list(REGION_IDS.values()),
                    "issue_at": issue_for_target(target),
                    "target_start": monday(target),
                    "target_end": monday(target) + pd.Timedelta(days=6),
                    "status": "missing_snapshot",
                    "eligible": False,
                    "feature_provenance": "unavailable",
                }
            )
        if features.empty:
            continue
        truth = truth_for_target(store, target, as_of=truth_as_of)
        joined = features.merge(
            truth, on="region_id", how="left", validate="one_to_one"
        )
        joined["target_incidence"] = pd.to_numeric(
            joined.target_incidence, errors="raise"
        ).astype(float)
        joined["truth_available_at"] = pd.to_datetime(
            joined.truth_available_at, utc=True
        )
        parts.append(joined)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def seasonal_reference_frame(store: SnapshotStore) -> pd.DataFrame:
    """Fixed pre-evaluation reference, explicitly using earliest archive revision."""
    data = store.load_snapshot(store.manifest_frame().iloc[0].snapshot_id)
    return data[
        (data.week_start >= "2017-09-04")
        & (data.week_start <= "2022-08-28")
        & data.region_id.isin(REGION_IDS.values())
    ].copy()


def build_training_rows(
    store: SnapshotStore, as_of, years=12, allow_reconstructed_as_of=False
) -> pd.DataFrame:
    end = monday(utc(as_of)) - pd.Timedelta(weeks=1)
    start = monday(end - pd.DateOffset(years=years))
    pre_archive = utc(as_of) < store.manifest_frame().available_at.min()
    retrospective = pre_archive and allow_reconstructed_as_of
    rows = build_backtest_rows(
        store,
        pd.date_range(start, end, freq="W-MON"),
        allow_reconstructed=True,
        truth_as_of=None if retrospective else as_of,
    )
    if rows.empty:
        return rows
    if retrospective:
        # This is a DEVELOPMENT reconstruction, never a historical availability claim.
        # Keep actual future publication timestamps; only the observation/maturity
        # cutoff is simulated. Callers must disclose this provenance explicitly.
        maturity = (
            (rows.target_start + pd.Timedelta(days=42))
            .dt.tz_localize("Europe/Berlin")
            .dt.tz_convert("UTC")
        )
        rows = rows[
            rows.eligible & rows.target_incidence.notna() & (maturity <= utc(as_of))
        ].copy()
        rows["training_provenance"] = "retrospective_development_reconstruction"
    else:
        rows = rows[
            rows.eligible
            & rows.target_incidence.notna()
            & (rows.truth_available_at <= utc(as_of))
            & (rows.data_available_at <= utc(as_of))
        ].copy()
        rows["training_provenance"] = "issue_available_labels"
    return rows.reset_index(drop=True)


COVID_URL = "https://raw.githubusercontent.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/main/COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv"


def covid_observations(content: bytes, as_of, fetched_at, snapshot_id: str) -> dict:
    """Observed district COVID context, never a respiratory forecast.

    Select the latest actual seven-day endpoint at least three calendar days old.
    Consolidate Berlin boroughs by summing population and seven-day cases.
    """
    data = pd.read_csv(io.BytesIO(content), dtype={"Landkreis_id": str})
    required = {
        "Meldedatum",
        "Landkreis_id",
        "Bevoelkerung",
        "Faelle_7-Tage",
        "Inzidenz_7-Tage",
    }
    if not required.issubset(data):
        raise ValueError("Unexpected RKI COVID seven-day schema")
    data["Meldedatum"] = pd.to_datetime(data.Meldedatum, errors="raise")
    codes = data.Landkreis_id
    if not codes.str.fullmatch(r"\d{5}").all():
        raise ValueError("Invalid district AGS in COVID source")
    cutoff = utc(as_of).tz_convert("Europe/Berlin").tz_localize(
        None
    ).normalize() - pd.Timedelta(days=3)
    eligible = data[data.Meldedatum <= cutoff].copy()
    if eligible.empty:
        raise ValueError("No completed COVID observation endpoint available")
    # Use one shared source date: mixing Berlin borough dates would be invalid.
    observed = eligible.Meldedatum.max()
    latest = eligible[eligible.Meldedatum == observed].copy()
    if latest.duplicated("Landkreis_id").any():
        raise ValueError("Duplicate COVID district/date")
    for name in ("Bevoelkerung", "Faelle_7-Tage", "Inzidenz_7-Tage"):
        latest[name] = pd.to_numeric(latest[name], errors="raise")
    if not (
        latest.Bevoelkerung.gt(0)
        & latest["Faelle_7-Tage"].ge(0)
        & latest["Inzidenz_7-Tage"].ge(0)
    ).all():
        raise ValueError("Invalid COVID cases/population")
    latest.loc[latest.Landkreis_id.str.startswith("11"), "Landkreis_id"] = "11000"
    records = []
    for code, district in latest.groupby("Landkreis_id"):
        # Retain exact published incidence except the explicitly consolidated Berlin.
        value = (
            district["Faelle_7-Tage"].sum() / district.Bevoelkerung.sum() * 100_000
            if len(district) > 1
            else float(district["Inzidenz_7-Tage"].iloc[0])
        )
        records.append(
            {
                "location_id": code,
                "indicator": "covid",
                "observed_through": observed.date().isoformat(),
                "value": float(value),
                "units": "reported cases per 100000 over 7 days",
                "source_url": COVID_URL,
                "source_snapshot_id": snapshot_id,
                "fetched_at": utc(fetched_at).isoformat(),
            }
        )
    return {
        "schema_version": 1,
        "observations": records,
        "reporting_lag_buffer_days": 3,
        "notes": "Observed reported COVID notifications only; no inference of individual or all-virus infection risk.",
    }


def refresh_covid_context(
    root="data/respiratory",
    output="data/service/covid_observations.json",
    as_of=None,
    offline=False,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "covid_source.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if offline:
        if not metadata:
            raise RuntimeError(
                "No cached versioned COVID source; run covid refresh first"
            )
        content = gzip.decompress((root / metadata["raw_file"]).read_bytes())
    else:
        headers = {"If-None-Match": metadata["etag"]} if metadata.get("etag") else {}
        response = _session().get(COVID_URL, headers=headers, timeout=90)
        if response.status_code == 304:
            content = gzip.decompress((root / metadata["raw_file"]).read_bytes())
        else:
            response.raise_for_status()
            content = response.content
            digest = hashlib.sha256(content).hexdigest()
            # Validate before accepting a new upstream snapshot.
            covid_observations(
                content,
                as_of or pd.Timestamp.now(tz="UTC"),
                pd.Timestamp.now(tz="UTC"),
                digest,
            )
            raw = root / "covid_snapshots" / f"{digest}.csv.gz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not raw.exists():
                raw.write_bytes(gzip.compress(content, mtime=0))
            metadata = {
                "raw_file": str(raw.relative_to(root)),
                "source_snapshot_id": digest,
                "etag": response.headers.get("ETag"),
                "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "source_url": COVID_URL,
            }
        metadata["latest_checked_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if hashlib.sha256(content).hexdigest() != metadata["source_snapshot_id"]:
        raise ValueError("Cached COVID checksum mismatch")
    result = covid_observations(
        content,
        as_of or pd.Timestamp.now(tz="UTC"),
        metadata.get("latest_checked_at", metadata["fetched_at"]),
        metadata["source_snapshot_id"],
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    return result
