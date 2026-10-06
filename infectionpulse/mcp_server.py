"""Portable read-only MCP tools over the same cached service as REST."""

import os
from datetime import date
from typing import Any

from infectionpulse.activity_policy import Activity
from infectionpulse.service import AssessmentRequest, ForecastService


def create_mcp(service: ForecastService | None = None):
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    service = service or ForecastService()
    extra_hosts = [
        host.strip()
        for host in os.getenv("INFECTIONPULSE_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    server = FastMCP(
        "InfectionPulse",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "localhost",
                "localhost:*",
                "127.0.0.1",
                "127.0.0.1:*",
                "[::1]",
                "[::1]:*",
                *extra_hosts,
            ],
            allowed_origins=[
                "http://localhost",
                "http://localhost:*",
                "http://127.0.0.1",
                "http://127.0.0.1:*",
                *[
                    origin.strip()
                    for origin in os.getenv("INFECTIONPULSE_ALLOWED_ORIGINS", "").split(
                        ","
                    )
                    if origin.strip()
                ],
            ],
        ),
        instructions="Use published, dated respiratory forecasts. Personal activity recommendations are ordinal precautions, not infection probabilities. Resolve ambiguous locations; preserve grey/unknown results and actual macroregional geography. Calls read cached artifacts and never invoke paid inference.",
    )

    @server.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def resolve_location(query: str) -> dict[str, Any]:
        """Resolve a German district name or five-digit AGS; clarify ambiguous results."""
        return service.resolve_location(query)

    @server.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def get_forecast(
        location_id: str, target_week_start: str | None = None
    ) -> dict[str, Any]:
        """Read saved ARE macroregion and separate district COVID forecasts; dates and freshness are explicit."""
        return service.get_forecast(location_id, target_week_start)

    @server.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def assess_activity(
        home_location_id: str,
        destination_location_id: str,
        activity_date: date,
        activity: Activity,
    ) -> dict[str, Any]:
        """Assess an activity within a saved forecast's dated target week. Unsupported dates return grey; never a personal infection percentage."""
        request = AssessmentRequest(
            home_location_id=home_location_id,
            destination_location_id=destination_location_id,
            activity_date=activity_date,
            activity=activity,
        )
        return service.assess_activity(request)

    @server.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def get_source_status() -> dict[str, Any]:
        """Read data freshness, supported target weeks and model identity; no network refresh."""
        return service.get_source_status()

    return server


if __name__ == "__main__":
    create_mcp().run(transport="stdio")
