import gzip
import io

import pandas as pd
import pytest

from infectionpulse.respiratory_data import (
    FEATURE_COLUMNS,
    REGIONS,
    SnapshotStore,
    build_backtest_rows,
    build_issue_rows,
    build_training_rows,
    covid_observations,
    freshness,
    issue_for_target,
    next_target,
    parse_snapshot,
    region_for_district,
    truth_for_target,
)


def source(end="2024-03-25", offset=0):
    rows = []
    for region in (*REGIONS, "Bundesweit"):
        for i, week in enumerate(pd.date_range("2022-01-03", end, freq="W-MON")):
            rows.append(
                {
                    "Meldungen": 1000,
                    "Saison": "2023/24",
                    "Erkrankung": "ARE",
                    "Altersgruppe": "00+",
                    "Region": region,
                    "Kalenderwoche": week.strftime("%G-W%V"),
                    "Inzidenz": 1000 + i + offset,
                }
            )
    return pd.DataFrame(rows).to_csv(sep="\t", index=False).encode()


def test_schema_regions_and_agss():
    assert len(parse_snapshot(source()).region_id.unique()) == 5
    assert region_for_district("11000") == "east"
    assert region_for_district("01001") == "north_west"
    assert region_for_district("09162") == "south"
    assert region_for_district("05315") == "central_west"
    for invalid in ["1001", "00000", "099999", "ABCDE"]:
        with pytest.raises(ValueError):
            region_for_district(invalid)
    with pytest.raises(ValueError, match="schema"):
        parse_snapshot(source().replace(b"Meldungen", b"count"))
    with pytest.raises(ValueError, match="Duplicate"):
        parse_snapshot(source() + source().split(b"\n", 1)[1])


def test_calendar_dst_and_target():
    assert issue_for_target("2024-03-25") == pd.Timestamp("2024-03-22T09:00Z")
    assert issue_for_target("2024-04-01") == pd.Timestamp("2024-03-29T09:00Z")
    assert issue_for_target("2024-04-08") == pd.Timestamp("2024-04-05T08:00Z")
    assert next_target("2024-12-27T09:00Z") == pd.Timestamp("2024-12-30")
    with pytest.raises(ValueError):
        issue_for_target("2024-04-02")


def test_immutable_checksum(tmp_path):
    store = SnapshotStore(tmp_path)
    store.ingest("first", "2024-04-04T07:00Z", source())
    store.ingest("first", "2024-04-04T07:00Z", source())
    assert len(store.manifest_frame()) == 1
    with pytest.raises(ValueError, match="Immutable"):
        store.ingest("first", "2024-04-04T07:00Z", source(offset=20))
    path = tmp_path / "snapshots/first.tsv.gz"
    path.write_bytes(gzip.compress(source(offset=20)))
    with pytest.raises(ValueError, match="checksum"):
        store.load_snapshot("first")


def test_features_use_issue_snapshot_never_later_revisions(tmp_path):
    store = SnapshotStore(tmp_path)
    store.ingest("first", "2024-04-04T07:00Z", source())
    earlier = build_issue_rows(store, "2024-04-05T08:00Z")
    store.ingest("second", "2024-04-11T07:00Z", source("2024-04-01", offset=200))
    repeated = build_issue_rows(store, "2024-04-05T08:00Z")
    pd.testing.assert_frame_equal(earlier, repeated)
    assert earlier.eligible.all()
    assert earlier[FEATURE_COLUMNS].notna().all().all()
    assert (earlier.latest_observed_week == pd.Timestamp("2024-03-25")).all()
    assert (earlier.target_start == pd.Timestamp("2024-04-08")).all()
    assert (earlier.incidence - earlier.inc_lag_1 == 1).all()


def test_missing_week_is_not_compressed(tmp_path):
    data = pd.read_csv(io.BytesIO(source()), sep="\t", dtype={"Altersgruppe": str})
    data = data[data.Kalenderwoche != "2024-W12"]
    store = SnapshotStore(tmp_path)
    store.ingest(
        "missing", "2024-04-04T07:00Z", data.to_csv(sep="\t", index=False).encode()
    )
    result = build_issue_rows(store, "2024-04-05T08:00Z")
    assert not result.eligible.any()
    assert result.inc_lag_1.isna().all()


def test_stale_source_and_future_rejection(tmp_path):
    assert not freshness("2024-04-04T07:00Z", "2024-03-25", "2024-04-19T08:00Z")[
        "fresh"
    ]
    assert not freshness("2024-04-04T07:00Z", "2024-03-25", "2024-04-03T08:00Z")[
        "fresh"
    ]
    store = SnapshotStore(tmp_path)
    store.ingest("first", "2024-04-04T07:00Z", source())
    assert (build_issue_rows(store, "2024-04-19T08:00Z").status == "source_stale").all()
    with pytest.raises(ValueError, match="future"):
        build_issue_rows(store, "2024-04-05T08:00Z", "2024-04-01")


def test_truth_maturity_and_first_vintage(tmp_path):
    store = SnapshotStore(tmp_path)
    # Target ends March 3; mature only after April 7 has ended.
    store.ingest("early", "2024-04-04T07:00Z", source())
    store.ingest("mature", "2024-04-11T07:00Z", source(offset=30))
    store.ingest("later", "2024-04-18T07:00Z", source(offset=60))
    truth = truth_for_target(store, "2024-02-26")
    assert (truth.truth_snapshot_id == "mature").all()
    assert truth_for_target(store, "2024-02-26", as_of="2024-04-05").empty


def test_reconstruction_is_labeled_and_never_used_in_evaluation(tmp_path):
    store = SnapshotStore(tmp_path)
    store.ingest("first", "2024-04-04T07:00Z", source())
    result = build_backtest_rows(store, ["2024-02-05"])
    assert (result.status == "missing_snapshot").all()
    reconstructed = build_backtest_rows(store, ["2024-02-05"], allow_reconstructed=True)
    assert (reconstructed.feature_provenance == "reconstructed_pre_archive").all()
    assert not build_training_rows(store, "2024-04-05", years=1).empty
    assert build_training_rows(store, "2024-04-03", years=1).empty
    development = build_training_rows(
        store, "2024-04-03", years=1, allow_reconstructed_as_of=True
    )
    assert not development.empty
    assert (
        development.training_provenance == "retrospective_development_reconstruction"
    ).all()
    assert (development.truth_available_at > pd.Timestamp("2024-04-03", tz="UTC")).all()


def test_covid_context_uses_dated_endpoint_and_population_weighted_berlin():
    data = pd.DataFrame(
        [
            ["2026-09-14", "11001", 100000, 10, 10.0],
            ["2026-09-14", "11002", 300000, 60, 20.0],
            ["2026-09-14", "01001", 100000, 5, 5.0],
            ["2026-09-16", "01001", 100000, 99, 99.0],
        ],
        columns=[
            "Meldedatum",
            "Landkreis_id",
            "Bevoelkerung",
            "Faelle_7-Tage",
            "Inzidenz_7-Tage",
        ],
    )
    result = covid_observations(
        data.to_csv(index=False).encode(),
        "2026-09-17T08:00Z",
        "2026-09-17T08:00Z",
        "digest",
    )
    by_district = {r["location_id"]: r for r in result["observations"]}
    assert by_district["11000"]["value"] == 17.5
    assert by_district["01001"]["value"] == 5
    assert all(r["observed_through"] == "2026-09-14" for r in result["observations"])
