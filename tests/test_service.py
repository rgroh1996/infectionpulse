import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from infectionpulse.activity_policy import Activity
from infectionpulse.service import (
    AssessmentRequest,
    ForecastBundle,
    ForecastService,
    create_app,
)

NOW = datetime(2026, 9, 17, 12, tzinfo=UTC)


@pytest.fixture
def artifacts(tmp_path):
    row = {
        "forecast_id": "are-east-20260921",
        "model": "TabPFN-3.5",
        "model_version": "v3.5",
        "location_id": "east",
        "geographic_level": "macroregion",
        "indicator": "are",
        "units": "ARE per 100000",
        "observed_through": "2026-09-13",
        "published_at": "2026-09-17T08:00:00Z",
        "fetched_at": "2026-09-17T09:00:00Z",
        "issued_at": "2026-09-17T10:00:00Z",
        "target_week_start": "2026-09-21",
        "target_week_end": "2026-09-27",
        "q10": 10,
        "q50": 90,
        "q90": 100,
        "reference_p50": 50,
        "reference_p80": 80,
        "reference_end": "2026-08-31",
        "source_url": "https://github.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts",
    }
    bundle = tmp_path / "forecast_bundle.json"
    bundle.write_text(
        json.dumps(
            {"schema_version": 1, "generated_at": NOW.isoformat(), "forecasts": [row]}
        )
    )
    locations = tmp_path / "locations.json"
    locations.write_text(
        json.dumps(
            [
                {"location_id": "11000", "name": "Berlin", "region_id": "east"},
                {"location_id": "12054", "name": "Potsdam", "region_id": "east"},
                {
                    "location_id": "12053",
                    "name": "Frankfurt (Oder)",
                    "region_id": "east",
                },
                {
                    "location_id": "06412",
                    "name": "Frankfurt am Main",
                    "region_id": "central_west",
                },
            ]
        )
    )
    return bundle, locations


@pytest.fixture
def service(artifacts):
    return ForecastService(*artifacts, now=lambda: NOW)


def request(day="2026-09-22"):
    return AssessmentRequest(
        home_location_id="11000",
        destination_location_id="12054",
        activity_date=day,
        activity=Activity(
            category="work",
            setting="indoor",
            duration_minutes=120,
            commute_mode="public_transit",
            commute_minutes=35,
        ),
    )


def test_location_ambiguity_is_preserved(service):
    assert service.resolve_location("Frankfurt")["status"] == "ambiguous"
    assert service.resolve_location("11000")["candidates"][0]["name"] == "Berlin"
    assert service.resolve_location("Not a real district")["status"] == "not_found"


def test_explanation_numbers_and_counterfactuals_follow_actual_policy(service):
    result = service.assess_activity(request())
    explanation = result["explanation"]
    assert explanation["method"] == "exact_policy_trace"
    assert result["precaution_score"]["value"] == 85
    assert "90 expected ARE illnesses per 100,000" in explanation["summary"]
    assert "12.5% above" in explanation["summary"]
    assert "120 minutes indoor" in explanation["summary"]
    assert (
        "high community activity with high exposure to give red"
        in explanation["summary"]
    )
    alternatives = {a["title"]: a for a in explanation["alternatives"]}
    # Removing the train must not erase the shared-office exposure.
    assert alternatives["Travel alone instead"]["traffic_light"] == "red"
    assert alternatives["Work from home alone"]["traffic_light"] == "green"
    assert alternatives["Work from home alone"]["score"] == 30
    assert explanation["probability"]["value"] is None
    assert explanation["probability"]["personal_infection_probability"] is None


def test_unknown_evidence_does_not_generate_reassuring_explanation(service):
    result = service.assess_activity(request("2026-09-18"))
    explanation = result["explanation"]
    assert explanation["component_details"] == []
    assert result["precaution_score"]["value"] is None
    assert explanation["alternatives"] == []
    assert explanation["probability"]["status"] == "unavailable"
    assert "No risk level or probability is inferred" in explanation["summary"]


def test_upper_interval_explanation_preserves_yellow_floor(service, artifacts):
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"][0].update(q50=20, q90=100)
    artifacts[0].write_text(json.dumps(payload))
    changed = request().model_copy(
        update={
            "activity": Activity(
                category="social",
                setting="indoor",
                shared_space=True,
                duration_minutes=15,
                crowding="uncrowded",
                ventilation="good",
                commute_mode="none",
            )
        }
    )
    result = service.assess_activity(changed)
    assert result["assessment"]["traffic_light"] == "yellow"
    assert (
        "upper forecast crossing the high threshold" in result["explanation"]["summary"]
    )


def test_shared_macroregion_does_not_invent_district_forecasts(service):
    result = service.assess_activity(request())
    assert result["assessment"]["traffic_light"] == "red"
    assert len(result["forecasts"]) == 1
    assert result["forecasts"][0]["location_id"] == "east"
    assert result["forecasts"][0]["geographic_level"] == "macroregion"
    assert result["forecasts"][0]["freshness"] == "fresh"


def test_date_without_supported_forecast_is_grey(service):
    result = service.assess_activity(request("2026-09-18"))
    assert result["assessment"]["traffic_light"] == "grey"
    result = service.assess_activity(request("2026-09-27"))
    assert result["assessment"]["traffic_light"] == "red"
    assert (
        service.assess_activity(request("2026-09-28"))["assessment"]["traffic_light"]
        == "grey"
    )


def test_stale_checks_do_not_greenlight(service):
    service.now = lambda: datetime(2026, 9, 21, 12, tzinfo=UTC)
    result = service.assess_activity(request())
    assert result["assessment"]["traffic_light"] == "grey"
    assert "source_check_stale" in result["assessment"]["reason_codes"]


def test_missing_and_invalid_artifacts_fail_closed(service, artifacts):
    artifacts[0].write_text("not json")
    assert service.get_source_status()["status"] == "forecast_artifact_invalid"
    assert service.assess_activity(request())["assessment"]["traffic_light"] == "grey"
    artifacts[0].unlink()
    assert service.get_forecast("11000")["status"] == "forecast_artifact_missing"


def test_quantiles_and_reference_cannot_be_invalid(artifacts):
    bundle = json.loads(artifacts[0].read_text())
    bundle["forecasts"][0]["q10"] = 1000
    with pytest.raises(ValidationError):
        ForecastBundle.model_validate(bundle)


def test_rest_validation_auth_and_parity(service):
    from fastapi.testclient import TestClient

    with TestClient(
        create_app(service, access_token="test-service-token", include_mcp=False)
    ) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/status").status_code == 401
        headers = {"Authorization": "Bearer test-service-token"}
        result = client.post(
            "/v1/assessments", headers=headers, json=request().model_dump(mode="json")
        )
        assert result.status_code == 200
        assert result.json() == service.assess_activity(request())
        assert (
            client.post("/v1/assessments", headers=headers, json={}).status_code == 422
        )
        assert client.get("/openapi.json").status_code == 200


def test_real_stdio_mcp_discovers_and_calls(artifacts):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        env = {
            **os.environ,
            "INFECTIONPULSE_BUNDLE": str(artifacts[0]),
            "INFECTIONPULSE_LOCATIONS": str(artifacts[1]),
        }
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "infectionpulse.mcp_server"],
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tools = await client.list_tools()
                assert {t.name for t in tools.tools} == {
                    "resolve_location",
                    "get_forecast",
                    "assess_activity",
                    "get_source_status",
                }
                result = await client.call_tool("resolve_location", {"query": "Berlin"})
                assert not result.isError
                payload = result.structuredContent
                assert payload["status"] == "resolved"
                assert payload["candidates"][0]["location_id"] == "11000"
                status = await client.call_tool("get_source_status", {})
                assert status.structuredContent["inference_on_request"] is False

    asyncio.run(run())


def test_real_http_mcp_parity(service):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def run():
        app = create_app(service, access_token="test-service-token")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                headers={"Authorization": "Bearer test-service-token"},
                follow_redirects=True,
            ) as http:
                async with streamable_http_client(
                    "http://localhost:8000/mcp/", http_client=http
                ) as (reader, writer, _):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        result = await client.call_tool(
                            "assess_activity", request().model_dump(mode="json")
                        )
                        assert not result.isError
                        assert result.structuredContent == service.assess_activity(
                            request()
                        )

    asyncio.run(run())


def test_activity_window_uses_german_local_date(service):
    service.now = lambda: datetime(2026, 9, 17, 23, tzinfo=UTC)
    result = service.assess_activity(request("2026-09-17"))
    assert result["assessment"]["reason_codes"] == ["activity_date_in_past"]


def test_source_release_and_observation_age_independent_of_download(service, artifacts):
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"][0]["published_at"] = "2026-09-06T08:00:00Z"
    artifacts[0].write_text(json.dumps(payload))
    assert service.assess_activity(request())["assessment"]["reason_codes"] == [
        "source_release_stale"
    ]
    payload["forecasts"][0]["published_at"] = "2026-09-17T08:00:00Z"
    payload["forecasts"][0]["observed_through"] = "2026-08-30"
    artifacts[0].write_text(json.dumps(payload))
    assert service.assess_activity(request())["assessment"]["reason_codes"] == [
        "observations_stale"
    ]


def test_unchanged_poll_refreshes_check_not_issue_or_observation(
    service, artifacts, tmp_path
):
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"][0]["source_snapshot_id"] = "snapshot-1"
    artifacts[0].write_text(json.dumps(payload))
    service.now = lambda: datetime(2026, 9, 21, 12, tzinfo=UTC)
    status = tmp_path / "status.json"
    service.source_status_path = status
    status.write_text(
        json.dumps(
            {
                "latest_checked_at": "2026-09-21T09:00:00Z",
                "last_check_status": "unchanged",
                "latest_snapshot_id": "snapshot-1",
            }
        )
    )
    result = service.assess_activity(request())
    assert result["assessment"]["traffic_light"] == "red"
    assert result["forecasts"][0]["issued_at"] == "2026-09-17T10:00:00Z"
    assert result["forecasts"][0]["observed_through"] == "2026-09-13"
    status.write_text(
        json.dumps(
            {
                "latest_checked_at": "2026-09-21T09:00:00Z",
                "last_check_status": "updated",
                "latest_snapshot_id": "snapshot-2",
            }
        )
    )
    result = service.assess_activity(request())
    assert result["assessment"]["traffic_light"] == "red"
    assert result["forecasts"][0]["newer_source_available"] is True
    assert result["forecasts"][0]["issued_at"] == "2026-09-17T10:00:00Z"


def test_observed_covid_is_separate_and_does_not_change_are_policy(service, tmp_path):
    before = service.assess_activity(request())
    service.observations_path = tmp_path / "covid.json"
    service.observations_path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "location_id": "11000",
                        "indicator": "covid",
                        "observed_through": "2026-09-16",
                        "value": 1.5,
                        "units": "reported COVID cases per 100000 over 7 days",
                        "source_url": "https://github.com/robert-koch-institut",
                        "source_snapshot_id": "covid-1",
                        "fetched_at": "2026-09-17T08:00:00Z",
                    }
                ]
            }
        )
    )
    after = service.assess_activity(request())
    assert before["assessment"] == after["assessment"]
    assert after["local_context"][0]["kind"] == "observation"
    assert after["local_context"][0]["value"] == 1.5
    assert "target_week_start" not in after["local_context"][0]


def test_mcp_endpoint_requires_service_token(service):
    from fastapi.testclient import TestClient

    with TestClient(create_app(service, access_token="service-secret")) as client:
        assert client.post("/mcp/", json={}).status_code == 401


def test_new_thursday_release_preserves_current_week_and_next_week(
    service, artifacts, tmp_path
):
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"][0]["source_snapshot_id"] = "snapshot-1"
    newer = dict(payload["forecasts"][0])
    newer.update(
        forecast_id="are-east-20260928",
        source_snapshot_id="snapshot-2",
        observed_through="2026-09-20",
        published_at="2026-09-24T08:00:00Z",
        fetched_at="2026-09-24T09:00:00Z",
        issued_at="2026-09-24T10:00:00Z",
        target_week_start="2026-09-28",
        target_week_end="2026-10-04",
    )
    payload["forecasts"].append(newer)
    artifacts[0].write_text(json.dumps(payload))
    service.now = lambda: datetime(2026, 9, 24, 12, tzinfo=UTC)
    service.source_status_path = tmp_path / "source_status.json"
    service.source_status_path.write_text(
        json.dumps(
            {
                "latest_checked_at": "2026-09-24T09:00:00Z",
                "last_check_status": "ok",
                "latest_snapshot_id": "snapshot-2",
            }
        )
    )
    current = service.assess_activity(request("2026-09-25"))
    upcoming = service.assess_activity(request("2026-09-30"))
    assert current["assessment"]["traffic_light"] == "red"
    assert current["forecasts"][0]["newer_source_available"] is True
    assert upcoming["assessment"]["traffic_light"] == "red"
    assert upcoming["forecasts"][0]["newer_source_available"] is False
    assert (
        current["forecasts"][0]["forecast_id"]
        != upcoming["forecasts"][0]["forecast_id"]
    )


def test_target_sunday_remains_supported_at_ten_day_release_boundary(
    service, artifacts, tmp_path
):
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"][0]["source_snapshot_id"] = "snapshot-1"
    artifacts[0].write_text(json.dumps(payload))
    service.now = lambda: datetime(2026, 9, 27, 18, tzinfo=UTC)
    service.source_status_path = tmp_path / "source_status.json"
    service.source_status_path.write_text(
        json.dumps(
            {
                "latest_checked_at": "2026-09-27T12:00:00Z",
                "last_check_status": "ok",
                "latest_snapshot_id": "snapshot-2",
            }
        )
    )
    result = service.assess_activity(request("2026-09-27"))
    assert result["assessment"]["traffic_light"] == "red"
    assert result["forecasts"][0]["freshness"] == "fresh"


def test_packaged_catalog_resolves_locations_without_downloaded_data(
    tmp_path, monkeypatch
):
    import shutil

    import infectionpulse.service as service_module

    source = Path(__file__).resolve().parents[1] / "examples/locations.json"
    (tmp_path / "examples").mkdir()
    shutil.copyfile(source, tmp_path / "examples/locations.json")
    monkeypatch.setattr(service_module, "ROOT", tmp_path)
    monkeypatch.delenv("INFECTIONPULSE_LOCATIONS", raising=False)
    standalone = ForecastService()
    assert not (tmp_path / "data").exists()
    assert standalone.get_source_status()["location_count"] == 400
    berlin = standalone.resolve_location("Berlin")
    assert berlin["status"] == "resolved"
    assert berlin["candidates"][0]["location_id"] == "11000"
    assert berlin["candidates"][0]["region_id"] == "east"
    explicit_missing = ForecastService(locations_path=tmp_path / "missing.json")
    assert (
        explicit_missing.resolve_location("Berlin")["status"]
        == "location_catalog_unavailable"
    )


def add_low_destination(artifacts):
    payload = json.loads(artifacts[0].read_text())
    destination = dict(payload["forecasts"][0])
    destination.update(
        forecast_id="are-central-20260921",
        location_id="central_west",
        q10=5,
        q50=10,
        q90=20,
    )
    payload["forecasts"].append(destination)
    artifacts[0].write_text(json.dumps(payload))


def cross_region_request(mode="none", minutes=0):
    return AssessmentRequest(
        home_location_id="11000",
        destination_location_id="06412",
        activity_date="2026-09-22",
        activity=Activity(
            category="work",
            setting="indoor",
            duration_minutes=120,
            crowding="crowded",
            commute_mode=mode,
            commute_minutes=minutes,
        ),
    )


def test_no_commute_does_not_apply_home_burden_to_destination(service, artifacts):
    add_low_destination(artifacts)
    result = service.assess_activity(cross_region_request())
    assert result["assessment"]["traffic_light"] == "yellow"
    assert result["assessment"]["burden"] == "low"
    assert result["dominant_component"] == "destination"
    assert len(result["components"]) == 1
    assert {f["location_id"] for f in result["forecasts"]} == {"central_west"}
    assert "Central-western Germany: 10 expected" in result["explanation"]["summary"]
    assert "Eastern Germany" not in result["explanation"]["summary"]


def test_map_preserves_geography_date_and_missingness(service):
    from datetime import date

    from infectionpulse.risk_scores import germany_overview

    rows = germany_overview(service, date(2026, 9, 22))
    assert len(rows) == 4
    assert next(r for r in rows if r["region_id"] == "east")["level"] == "high"
    assert sum(r["level"] == "unknown" for r in rows) == 3
    assert all(
        r["level"] == "unknown" for r in germany_overview(service, date(2026, 9, 29))
    )
    service.now = lambda: datetime(2026, 9, 21, 12, tzinfo=UTC)
    stale = germany_overview(service, date(2026, 9, 22))
    assert all(r["level"] == "unknown" and r["q50"] is None for r in stale)


def test_same_colour_uses_higher_score_and_zero_minute_commute_is_ignored(
    service, artifacts
):
    add_low_destination(artifacts)
    solo = service.assess_activity(cross_region_request("public_transit", 20))
    assert solo["assessment"]["traffic_light"] == "yellow"
    assert solo["precaution_score"]["value"] == 65
    assert solo["dominant_component"] == "commute"
    no_travel = service.assess_activity(cross_region_request("car_alone", 0))
    assert no_travel["precaution_score"]["value"] == 45
    assert no_travel["dominant_component"] == "destination"
    assert len(no_travel["components"]) == 1


def test_shared_commute_evaluated_separately_from_destination(service, artifacts):
    add_low_destination(artifacts)
    result = service.assess_activity(cross_region_request("public_transit", 45))
    components = {c["component"]: c for c in result["components"]}
    assert components["destination"]["assessment"]["traffic_light"] == "yellow"
    assert components["destination"]["assessment"]["burden"] == "low"
    assert components["commute"]["assessment"]["traffic_light"] == "red"
    assert result["dominant_component"] == "commute"
    assert "commute" in result["assessment"]["recommendation"]
    assert "home office" in result["assessment"]["recommendation"]
    assert "Eastern Germany: 90 expected" in result["explanation"]["summary"]
    assert "45 minutes of public transit" in result["explanation"]["summary"]
    assert result["explanation"]["alternatives"][0]["traffic_light"] == "yellow"
    shorter = service.assess_activity(cross_region_request("public_transit", 15))
    assert shorter["assessment"]["traffic_light"] == "yellow"
    solo = service.assess_activity(cross_region_request("car_alone", 45))
    assert solo["assessment"]["traffic_light"] == "yellow"
    assert (
        next(c for c in solo["components"] if c["component"] == "commute")[
            "assessment"
        ]["exposure"]
        == "low"
    )


def test_home_forecast_required_only_with_commute_but_home_id_always_validated(
    service, artifacts
):
    add_low_destination(artifacts)
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"] = [
        f for f in payload["forecasts"] if f["location_id"] == "central_west"
    ]
    artifacts[0].write_text(json.dumps(payload))
    assert (
        service.assess_activity(cross_region_request())["assessment"]["traffic_light"]
        == "yellow"
    )
    with_commute = service.assess_activity(cross_region_request("public_transit", 45))
    assert with_commute["assessment"]["traffic_light"] == "grey"
    assert with_commute["dominant_component"] == "commute"
    invalid_home = cross_region_request().model_copy(
        update={"home_location_id": "99999"}
    )
    assert service.assess_activity(invalid_home)["assessment"]["reason_codes"] == [
        "location_not_found"
    ]


def test_cached_service_reads_do_not_use_network_or_spawn_inference(
    service, monkeypatch
):
    import builtins
    import socket
    import subprocess

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(
            (
                "tabpfn",
                "infectionpulse.tabpfn_api",
                "infectionpulse.respiratory_model",
            )
        ), "Model imports forbidden in service requests"
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Service request attempted network or subprocess inference"
        )

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert service.resolve_location("Berlin")["status"] == "resolved"
    assert service.get_forecast("11000")["status"] == "available"
    assert service.assess_activity(request())["assessment"]["traffic_light"] == "red"
    assert service.get_source_status()["inference_on_request"] is False


def test_city_and_county_with_same_name_remain_ambiguous():
    service = ForecastService(
        locations_path=Path(__file__).resolve().parents[1] / "examples/locations.json"
    )
    result = service.resolve_location("München")
    assert result["status"] == "ambiguous"
    assert {row["location_id"] for row in result["candidates"]} == {"09162", "09184"}
    assert service.resolve_location("09162")["status"] == "resolved"


@pytest.mark.parametrize(
    "ready,point,expected", [(False, 0.7, None), (True, None, None), (True, 0.7, 0.6)]
)
def test_public_probability_bounds_respect_readiness(
    service, artifacts, ready, point, expected
):
    payload = json.loads(artifacts[0].read_text())
    payload["forecasts"][0].update(
        probability_above_seasonal_reference=point,
        probability_lower_bound=0.6,
        probability_upper_bound=0.8,
        probability_readiness={"ready": ready},
    )
    artifacts[0].write_text(json.dumps(payload))
    forecast = service.get_forecast("11000")["forecasts"][0]
    assert forecast["probability_lower_bound"] == expected
    assert forecast["probability_upper_bound"] == (
        0.8 if expected is not None else None
    )
    assert forecast["probability_above_seasonal_reference"] == (
        point if expected is not None else None
    )
    assessment = service.assess_activity(request())
    assert assessment["forecasts"][0]["probability_lower_bound"] == expected
    assert (
        json.loads(artifacts[0].read_text())["forecasts"][0]["probability_lower_bound"]
        == 0.6
    )
