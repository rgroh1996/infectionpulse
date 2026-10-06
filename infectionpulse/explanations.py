"""Faithful decision traces from saved forecasts and the actual policy output.

These explain the advisory rule, not TabPFN feature attributions or causal effects.
No LLM, model inference, or new probability estimates are used here.
"""

from math import isfinite

from pydantic import BaseModel, Field

REGION_NAMES = {
    "east": "Eastern Germany",
    "central_west": "Central-western Germany",
    "north_west": "North-western Germany",
    "south": "Southern Germany",
}


class DecisionExplanation(BaseModel):
    method: str = "exact_policy_trace"
    summary: str
    component_details: list[dict] = Field(default_factory=list)
    probability: dict = Field(default_factory=dict)
    alternatives: list[dict] = Field(default_factory=list)
    scope: str = (
        "Explains how the saved forecast and activity assumptions produce the traffic light. "
        "It does not attribute TabPFN's prediction to input features or establish causal effects."
    )


def probability_summary(forecasts: list[dict], unavailable: bool) -> dict:
    """Never turn withheld bounds or incidence into an infection probability."""
    base = {
        "label": "Chance of above-usual respiratory activity",
        "personal_infection_probability": None,
        "definition": (
            "Probability that regional weekly ARE exceeds its frozen seasonal historical "
            "80th-percentile reference. This is a community event, not your chance of infection "
            "and not the threshold used for the traffic light."
        ),
        "value": None,
        "regions": [],
    }
    if unavailable or not forecasts:
        return {
            **base,
            "status": "unavailable",
            "reason": "No supported, fresh forecast covers this plan.",
        }
    for forecast in forecasts:
        readiness = forecast.get("probability_readiness") or {}
        value = forecast.get("probability_above_seasonal_reference")
        ready = readiness.get("ready") is True
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and 0 <= value <= 1
        )
        entry = {
            "region": REGION_NAMES.get(
                forecast["location_id"], forecast["location_id"]
            ),
            "forecast_id": forecast["forecast_id"],
            "value": value if ready and valid else None,
            "threshold_per_100k": forecast.get("seasonal_reference_p80"),
            "reliability_error": readiness.get("reliability_ece"),
        }
        base["regions"].append(entry)
    if any(row["value"] is None for row in base["regions"]):
        return {
            **base,
            "status": "withheld",
            "reason": "A validated probability is not available for this plan. The current calibration or forecast-output checks have not all passed.",
        }
    if len(base["regions"]) == 1:
        base["value"] = base["regions"][0]["value"]
    return {
        **base,
        "status": "available",
        "reason": "Regional model estimate; regions are shown separately and are not combined into personal odds.",
    }


def exposure_description(activity: dict, component: str) -> str:
    crowding_labels = {
        "crowded": "crowded conditions",
        "uncrowded": "uncrowded conditions",
        "unknown": "unknown crowding",
    }
    if component == "commute":
        mode = {
            "public_transit": "public transit",
            "shared_car": "a shared car",
            "car_alone": "driving alone",
            "bike": "cycling",
            "walk": "walking",
        }.get(activity["commute_mode"], "travel")
        shared = activity["commute_mode"] in ("public_transit", "shared_car")
        details = f"{activity['commute_minutes']} minutes of {mode}"
        if shared:
            details += f", {crowding_labels[activity['commute_crowding']]} and unmeasured ventilation"
        return details
    shared = "in a shared space" if activity["shared_space"] else "alone"
    details = f"{activity['duration_minutes']} minutes {activity['setting']}s, {shared}"
    if activity["shared_space"]:
        details += f", {crowding_labels[activity['crowding']]}"
        if activity["setting"] == "indoor":
            details += f", {activity['ventilation']} ventilation"
    return details


def explain_assessment(result: dict, activity: dict) -> DecisionExplanation:
    forecasts = {f["forecast_id"]: f for f in result["forecasts"]}
    unavailable = result["assessment"]["traffic_light"] == "grey"
    probability = probability_summary(list(forecasts.values()), unavailable)
    if unavailable:
        reasons = ", ".join(result["assessment"]["reason_codes"]).replace("_", " ")
        return DecisionExplanation(
            summary=f"There is not enough supported evidence for {result['activity_date']} ({reasons}). No risk level or probability is inferred from missing or stale data.",
            probability=probability,
        )
    components = []
    for component in result["components"]:
        policy = component["assessment"]
        rows = []
        for forecast_id in component["forecast_ids"]:
            f = forecasts[forecast_id]
            difference = (f["q50"] / f["reference_p80"] - 1) * 100
            region = REGION_NAMES.get(f["location_id"], f["location_id"])
            relation = "above" if difference >= 0 else "below"
            rows.append(
                {
                    "forecast_id": forecast_id,
                    "region": region,
                    "median_per_100k": f["q50"],
                    "q10_per_100k": f["q10"],
                    "q90_per_100k": f["q90"],
                    "moderate_threshold_per_100k": f["reference_p50"],
                    "high_threshold_per_100k": f["reference_p80"],
                    "difference_from_high_percent": difference,
                    "text": (
                        f"{region}: {f['q50']:,.0f} expected ARE illnesses per 100,000 "
                        f"for {f['target_week_start']}–{f['target_week_end']}, "
                        f"{abs(difference):.1f}% {relation} the historical high-activity threshold "
                        f"of {f['reference_p80']:,.0f}. The model's 10th–90th percentile range "
                        f"is {f['q10']:,.0f}–{f['q90']:,.0f}."
                    ),
                }
            )
        inputs = exposure_description(activity, component["component"])
        rule = f"{policy['burden'].capitalize()} community activity + {policy['exposure']} exposure → {policy['traffic_light']}"
        if "upper_forecast_crosses_high_reference" in policy["reason_codes"]:
            rule += "; the upper forecast also crosses the high-activity threshold, requiring at least yellow for shared indoor exposure"
        components.append(
            {
                "component": component["component"],
                "traffic_light": policy["traffic_light"],
                "exposure": policy["exposure"],
                "burden": policy["burden"],
                "inputs": inputs,
                "rule": rule + ".",
                "forecasts": rows,
            }
        )
    dominant = next(
        c for c in components if c["component"] == result["dominant_component"]
    )

    # Use the forecast driving the categorical burden, not an unrelated home region.
    def burden_rank(row):
        return (
            2
            if row["median_per_100k"] >= row["high_threshold_per_100k"]
            else 1
            if row["median_per_100k"] >= row["moderate_threshold_per_100k"]
            else 0
        )

    driving = max(
        dominant["forecasts"],
        key=lambda r: (burden_rank(r), r["difference_from_high_percent"]),
    )
    difference = driving["difference_from_high_percent"]
    relation = "above" if difference >= 0 else "below"
    summary = (
        f"{driving['region']}: {driving['median_per_100k']:,.0f} expected ARE illnesses per 100,000, "
        f"{abs(difference):.1f}% {relation} the historical high-activity threshold of {driving['high_threshold_per_100k']:,.0f}. "
        f"Your {dominant['component']} involves {dominant['inputs']}. "
        f"The precaution rule combines {dominant['burden']} community activity with "
        f"{dominant['exposure']} exposure to give {dominant['traffic_light']}."
    )
    if (
        dominant["traffic_light"] == "yellow"
        and dominant["burden"] == "low"
        and dominant["exposure"] != "high"
        and "upper_forecast_crosses_high_reference"
        in next(
            c for c in result["components"] if c["component"] == dominant["component"]
        )["assessment"]["reason_codes"]
    ):
        summary += " The upper forecast crossing the high threshold also sets a minimum of yellow for shared indoor exposure."
    if len(components) > 1:
        summary += " The overall light uses the more cautious result from the destination and commute."
    return DecisionExplanation(
        summary=summary, component_details=components, probability=probability
    )
