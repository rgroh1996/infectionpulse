"""Versioned ordinal precautions; these rules do not estimate personal infection odds."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

POLICY_VERSION = "1.2"


class Activity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["work", "social", "travel", "other"] = "other"
    setting: Literal["indoor", "outdoor"]
    duration_minutes: int = Field(ge=0, le=1440)
    shared_space: bool = True
    crowding: Literal["uncrowded", "crowded", "unknown"] = "unknown"
    ventilation: Literal["good", "poor", "unknown"] = "unknown"
    commute_mode: Literal[
        "none", "car_alone", "bike", "walk", "public_transit", "shared_car"
    ] = "none"
    commute_minutes: int = Field(default=0, ge=0, le=720)
    commute_crowding: Literal["uncrowded", "crowded", "unknown"] = "unknown"


class PolicyResult(BaseModel):
    policy_version: str = POLICY_VERSION
    traffic_light: Literal["green", "yellow", "red", "grey"]
    burden: Literal["low", "moderate", "high", "unknown"]
    exposure: Literal["low", "moderate", "high"]
    reason_codes: list[str]
    recommendation: str
    interpretation: str = "Precaution guidance from community surveillance and activity assumptions; not a personal infection probability."


def exposure_level(activity: Activity) -> str:
    shared_indoor = activity.setting == "indoor" and activity.shared_space
    high_activity = (
        shared_indoor
        and activity.duration_minutes >= 30
        and (activity.crowding == "crowded" or activity.ventilation != "good")
    )
    shared_commute = (
        activity.commute_mode in ("public_transit", "shared_car")
        and activity.commute_minutes > 0
    )
    high_commute = (
        shared_commute
        and activity.commute_minutes >= 30
        and activity.commute_crowding != "uncrowded"
    )
    if high_activity or high_commute:
        return "high"
    low_activity = not activity.shared_space or (
        activity.setting == "outdoor" and activity.crowding == "uncrowded"
    )
    return "low" if low_activity and not shared_commute else "moderate"


def assess_policy(
    activity: Activity, forecasts: list[dict], unavailable_reason: str | None = None
) -> PolicyResult:
    exposure = exposure_level(activity)
    if unavailable_reason or not forecasts:
        return PolicyResult(
            traffic_light="grey",
            burden="unknown",
            exposure=exposure,
            reason_codes=[unavailable_reason or "forecast_unavailable"],
            recommendation="There is not enough current forecast evidence for this activity date; follow routine respiratory precautions.",
        )
    levels = [
        "high"
        if f["q50"] >= f["reference_p80"]
        else "moderate"
        if f["q50"] >= f["reference_p50"]
        else "low"
        for f in forecasts
    ]
    burden = max(levels, key=["low", "moderate", "high"].index)
    matrix = {
        "low": ["green", "green", "yellow"],
        "moderate": ["green", "yellow", "yellow"],
        "high": ["green", "yellow", "red"],
    }
    light = matrix[burden][["low", "moderate", "high"].index(exposure)]
    reasons = [f"community_burden_{burden}", f"activity_exposure_{exposure}"]
    shared_indoor = activity.setting == "indoor" and activity.shared_space
    shared_commute = (
        activity.commute_mode in ("public_transit", "shared_car")
        and activity.commute_minutes > 0
    )
    if shared_commute:
        reasons.append("shared_transport")
    if shared_indoor and activity.duration_minutes >= 30:
        reasons.append("prolonged_shared_indoor_activity")
    if shared_indoor and activity.ventilation != "good":
        reasons.append(f"ventilation_{activity.ventilation}")
    if (shared_indoor or shared_commute) and any(
        f["q90"] >= f["reference_p80"] for f in forecasts
    ):
        if light == "green":
            light = "yellow"
        reasons.append("upper_forecast_crosses_high_reference")
    if light == "green":
        text = "No extra precaution is indicated by this forecast for your planned activity; keep your usual respiratory precautions."
    elif light == "yellow":
        text = "Take extra precautions for shared indoor time: consider a well-fitting mask, good ventilation, or an outdoor alternative."
    elif activity.category == "work":
        text = "Consider home office for this date: forecast respiratory activity is high and your planned shared indoor exposure is elevated."
    elif activity.setting == "outdoor" or not activity.shared_space:
        text = "Consider quieter travel or a different travel time: forecast respiratory activity is high and your planned shared transport increases exposure."
    else:
        text = "Consider postponing this crowded indoor activity or choosing an outdoor alternative: forecast respiratory activity is high."
    return PolicyResult(
        traffic_light=light,
        burden=burden,
        exposure=exposure,
        reason_codes=reasons,
        recommendation=text,
    )
