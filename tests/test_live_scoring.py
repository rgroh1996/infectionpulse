import json

import pandas as pd
from score_live import score


class Store:
    def manifest_frame(self):
        return pd.DataFrame(
            {
                "snapshot_id": ["too_early", "mature"],
                "available_at": pd.to_datetime(
                    ["2026-09-05T10:00Z", "2026-09-10T10:00Z"]
                ),
            }
        )

    def load_snapshot(self, snapshot_id):
        assert snapshot_id == "mature"
        return pd.DataFrame(
            {
                "region_id": ["east"],
                "week_start": pd.to_datetime(["2026-07-27"]),
                "incidence": [12.0],
            }
        )


def test_live_labels_wait_for_maturity_and_exclude_late_issues(tmp_path):
    record = {
        "forecast_id": "test:east",
        "model": "TabPFN-3.5",
        "model_version": "v3.5",
        "location_id": "east",
        "geographic_level": "macroregion",
        "indicator": "are",
        "units": "ARE per 100000",
        "observed_through": "2026-07-19",
        "published_at": "2026-07-23T08:00:00Z",
        "fetched_at": "2026-07-24T08:00:00Z",
        "issued_at": "2026-07-24T08:00:00Z",
        "target_week_start": "2026-07-27",
        "target_week_end": "2026-08-02",
        "q10": 8,
        "q50": 10,
        "q90": 15,
        "reference_p50": 5,
        "reference_p80": 18,
        "reference_end": "2022-08-28",
        "source_url": "https://github.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts",
    }
    path = tmp_path / "issue.json"
    path.write_text(
        json.dumps({"generated_at": "2026-07-24T08:01:00Z", "forecasts": [record]})
    )
    _, pending = score(tmp_path, Store(), "2026-09-08T00:00Z")
    assert pending["scored_forecasts"] == 0
    assert pending["mae"] is None
    frame, measured = score(tmp_path, Store(), "2026-09-12T00:00Z")
    assert measured["mae"] == 2
    assert measured["raw_coverage80"] == 1
    assert frame.truth_snapshot_id.iloc[0] == "mature"
    record["issued_at"] = "2026-07-28T08:00:00Z"
    path.write_text(
        json.dumps({"generated_at": "2026-07-24T08:01:00Z", "forecasts": [record]})
    )
    _, late = score(tmp_path, Store(), "2026-09-12T00:00Z")
    assert late["scored_forecasts"] == 0
    assert late["statuses"] == {"issued_after_target_started": 1}
    record["issued_at"] = "2026-07-24T08:00:00Z"
    path.write_text(
        json.dumps({"generated_at": "2026-09-12T08:00:00Z", "forecasts": [record]})
    )
    _, backdated = score(tmp_path, Store(), "2026-09-12T10:00Z")
    assert backdated["scored_forecasts"] == 0
    assert backdated["statuses"] == {"generated_after_target_started": 1}
