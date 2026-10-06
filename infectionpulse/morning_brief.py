"""Grounded local morning briefs; no LLM, delivery, downloads or inference."""

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from infectionpulse.activity_policy import Activity
from infectionpulse.explanations import REGION_NAMES, exposure_description
from infectionpulse.service import AssessmentRequest, ForecastService


class MorningProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    home_location_id: str = Field(pattern=r"^\d{5}$")
    destination_location_id: str = Field(pattern=r"^\d{5}$")
    activity: Activity


def build_brief(
    service: ForecastService,
    profile: MorningProfile,
    activity_date: date,
    generated_at: datetime,
    *,
    mode: str = "live",
    refresh_status: str = "not_requested",
) -> dict:
    """Use the public policy/probability gate unchanged and retain traceable evidence."""
    if mode not in ("live", "historical_replay"):
        raise ValueError("Unknown brief mode")
    if generated_at.tzinfo is None:
        raise ValueError("generated_at requires timezone")
    request = AssessmentRequest(
        **profile.model_dump(exclude={"schema_version"}), activity_date=activity_date
    )
    response = service.assess_activity(request)
    policy = response["assessment"]
    available = policy["traffic_light"] != "grey"
    evidence = []
    for forecast in response["forecasts"]:
        evidence.append(
            {
                key: forecast.get(key)
                for key in (
                    "forecast_id",
                    "model",
                    "model_version",
                    "location_id",
                    "geographic_level",
                    "indicator",
                    "units",
                    "q10",
                    "q50",
                    "q90",
                    "reference_p80",
                    "target_week_start",
                    "target_week_end",
                    "observed_through",
                    "published_at",
                    "fetched_at",
                    "issued_at",
                    "source_url",
                    "freshness",
                    "newer_source_available",
                )
            }
        )
    brief = {
        "schema_version": 1,
        "mode": mode,
        "generated_at": generated_at.isoformat(),
        "evidence_evaluated_at": service.now().isoformat(),
        "activity_date": activity_date.isoformat(),
        "status": "available" if available else "unavailable",
        "refresh_status": refresh_status,
        "renderer": "deterministic_policy_trace_v1",
        "llm_used": False,
        "profile": profile.model_dump(mode="json"),
        "traffic_light": policy["traffic_light"],
        "recommendation": policy["recommendation"],
        "reason_codes": policy["reason_codes"],
        "precaution_score": response.get("precaution_score"),
        "community_probability": (response.get("explanation") or {}).get("probability"),
        "personal_infection_probability": None,
        "forecasts": evidence,
        "disease_observations": response.get("disease_context", []),
        "explanation": response.get("explanation"),
        "limitations": response["limitations"],
    }
    brief["text"] = render_brief(brief)
    return brief


def render_brief(brief: dict) -> str:
    """Compact evidence-bearing prose with explicit source age and forecast scope."""
    lines = []
    if brief["mode"] == "historical_replay":
        lines.append(
            "HISTORICAL REPLAY — not current advice. "
            f"Evidence evaluated at {brief['evidence_evaluated_at']}."
        )
    light = brief["traffic_light"]
    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "grey": "⚪"}[light]
    score = (brief.get("precaution_score") or {}).get("value")
    score_text = f" · precaution score {score}/100" if score is not None else ""
    lines.append(
        f"{icon} {brief['activity_date']}{score_text}: {brief['recommendation']}"
    )
    if brief["refresh_status"] == "failed":
        lines.append(
            "Today's refresh failed; this brief uses the saved evidence and its freshness checks."
        )
    if brief["status"] == "unavailable":
        lines.append(
            "Evidence unavailable: "
            + ", ".join(brief["reason_codes"]).replace("_", " ")
            + "."
        )
        for forecast in brief["forecasts"]:
            lines.append(
                f"Saved forecast (not used for advice): observed through {forecast['observed_through']}; "
                f"source published {forecast['published_at']}; forecast issued {forecast['issued_at']}; "
                f"freshness: {forecast['freshness']}."
            )
    else:
        for forecast in brief["forecasts"]:
            region = REGION_NAMES.get(forecast["location_id"], forecast["location_id"])
            lines.append(
                f"{region}: TabPFN median {forecast['q50']:,.0f} ARE illnesses per 100,000 "
                f"for {forecast['target_week_start']}–{forecast['target_week_end']} "
                f"(10th–90th percentiles {forecast['q10']:,.0f}–{forecast['q90']:,.0f}; "
                f"historical high threshold {forecast['reference_p80']:,.0f}). "
                f"Observed through {forecast['observed_through']}; source published "
                f"{forecast['published_at']}; forecast issued {forecast['issued_at']}."
            )
        activity = brief["profile"]["activity"]
        inputs = exposure_description(activity, "destination")
        if activity["commute_mode"] != "none" and activity["commute_minutes"] > 0:
            inputs += "; " + exposure_description(activity, "commute")
        lines.append("Your plan: " + inputs + ".")
    lines.append(
        "The score is a rule-based precaution index, not your infection probability. Forecasts describe a broad region, not your street or train."
    )
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, text: str):
    """Keep personal profiles and briefs owner-readable and replace complete files."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".brief-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_brief(brief: dict, destination: Path):
    # JSON is authoritative and embeds the matching text, even during a reader race.
    atomic_write(destination.with_suffix(".txt"), brief["text"])
    atomic_write(
        destination.with_suffix(".json"),
        json.dumps(brief, indent=2, allow_nan=False) + "\n",
    )
