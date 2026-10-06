"""Validate a running cached service through real TCP HTTP and MCP, without inference."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]


async def validate(base_url):
    headers = {}
    if token := os.getenv("INFECTIONPULSE_ACCESS_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30) as http:
        health = await http.get(f"{base_url}/healthz")
        health.raise_for_status()
        async with streamable_http_client(f"{base_url}/mcp/", http_client=http) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                inventory = await session.list_tools()
                tool_names = sorted(t.name for t in inventory.tools)
                assert tool_names == sorted(
                    [
                        "resolve_location",
                        "get_forecast",
                        "assess_activity",
                        "get_source_status",
                    ]
                )
                resolution = await session.call_tool(
                    "resolve_location", {"query": "Berlin"}
                )
                assert not resolution.isError
                assert resolution.structuredContent["status"] == "resolved"
                forecast = await session.call_tool(
                    "get_forecast", {"location_id": "11000"}
                )
                assert not forecast.isError
                payload = forecast.structuredContent
                assert payload["status"] == "available", (
                    f"Forecast unavailable: {payload['status']}"
                )
                are = sorted(
                    [f for f in payload["forecasts"] if f["indicator"] == "are"],
                    key=lambda f: f["target_week_start"],
                )[-1]
                assert are["model"] == "TabPFN-3.5", (
                    "This validation requires a real TabPFN artifact"
                )
                assert are["freshness"] == "fresh", (
                    f"Forecast freshness: {are['freshness']}"
                )
                request = {
                    "home_location_id": "11000",
                    "destination_location_id": "12054",
                    "activity_date": are["target_week_start"],
                    "activity": {
                        "category": "work",
                        "setting": "indoor",
                        "duration_minutes": 480,
                        "shared_space": True,
                        "crowding": "crowded",
                        "ventilation": "unknown",
                        "commute_mode": "public_transit",
                        "commute_minutes": 45,
                        "commute_crowding": "crowded",
                    },
                }
                assessment = await session.call_tool("assess_activity", request)
                assert not assessment.isError
                result = assessment.structuredContent
                assert result["assessment"]["traffic_light"] != "grey"
                assert result["assessment"]["policy_version"] == "1.1"
                assert result["explanation"]["method"] == "exact_policy_trace"
                assert 0 <= result["precaution_score"]["value"] <= 100
                assert result["precaution_score"]["kind"] == "ordinal_policy_index"
                assert result["explanation"]["component_details"]
                assert (
                    result["explanation"]["probability"][
                        "personal_infection_probability"
                    ]
                    is None
                )
                assert {c["component"] for c in result["components"]} == {
                    "destination",
                    "commute",
                }
                rest = await http.post(f"{base_url}/v1/assessments", json=request)
                rest.raise_for_status()
                assert rest.json() == result, "REST and MCP assessment mismatch"
                status = await session.call_tool("get_source_status", {})
                assert not status.isError
                assert status.structuredContent["inference_on_request"] is False
                return {
                    "schema_version": 1,
                    "checked_at": datetime.now(UTC).isoformat(),
                    "service_url": base_url,
                    "status": "passed",
                    "transports": [
                        "HTTP REST over TCP",
                        "MCP Streamable HTTP over TCP",
                    ],
                    "packages": {
                        name: version(name) for name in ["mcp", "fastapi", "httpx"]
                    },
                    "tools": tool_names,
                    "location_resolution": resolution.structuredContent,
                    "forecast": payload,
                    "assessment_request": request,
                    "assessment": result,
                    "rest_mcp_parity": True,
                    "inference_on_request": False,
                    "limitations": [
                        "No Hermes or OpenClaw end-user session was tested.",
                        "Docker execution was not tested.",
                        "These are protocol and integration checks, not prospective accuracy validation.",
                    ],
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/local-validation.json"
    )
    args = parser.parse_args()
    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_service.py::test_cached_service_reads_do_not_use_network_or_spawn_inference",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if check.returncode:
        raise RuntimeError(check.stdout + check.stderr)
    result = asyncio.run(validate(args.url.rstrip("/")))
    result["cached_service_no_network_or_inference_guard_test"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(args.output),
                "model": result["forecast"]["forecasts"][0]["model"],
                "policy_version": result["assessment"]["assessment"]["policy_version"],
            }
        )
    )


if __name__ == "__main__":
    main()
