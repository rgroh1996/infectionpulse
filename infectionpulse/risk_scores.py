"""Coarse, transparent policy scores. These are not probabilities or percentiles."""

from typing import Literal

from pydantic import BaseModel, Field

# Rows follow the frozen community-burden / activity-exposure policy matrix.
# Deliberately coarse values avoid implying measured personal infection odds.
SCORES = {
    "low": {"low": 10, "moderate": 25, "high": 45},
    "moderate": {"low": 25, "moderate": 50, "high": 60},
    "high": {"low": 30, "moderate": 65, "high": 85},
}


class PrecautionScore(BaseModel):
    value: int | None = Field(default=None, ge=0, le=100)
    label: str = "Precaution score"
    version: str = "1.1"
    kind: Literal["ordinal_policy_index"] = "ordinal_policy_index"
    scale: str = "0–100 points; green <40, yellow 40–69, red ≥70"
    interpretation: str = "A rule-based indication of suggested precautions, not a probability of infection."


def precaution_score(assessment: dict) -> PrecautionScore:
    if assessment["traffic_light"] == "grey":
        return PrecautionScore()
    value = SCORES[assessment["burden"]][assessment["exposure"]]
    if assessment["traffic_light"] == "yellow":
        # An upper interval crossing the high reference raises green to yellow.
        value = max(40, value)
    return PrecautionScore(value=value)


def brief_reason(result: dict) -> str:
    """Two short sentences with the region, observed model numbers, and plan."""
    explanation = result["explanation"]
    if result["assessment"]["traffic_light"] == "grey":
        return "No fresh forecast covers this plan. A score is unavailable until the date and source checks pass."
    component = next(
        c
        for c in explanation["component_details"]
        if c["component"] == result["dominant_component"]
    )

    # Match the community-burden ordering used in the decision explanation.
    def rank(row):
        median = row["median_per_100k"]
        level = (
            2
            if median >= row["high_threshold_per_100k"]
            else 1
            if median >= row["moderate_threshold_per_100k"]
            else 0
        )
        return level, row["difference_from_high_percent"]

    row = max(component["forecasts"], key=rank)
    return (
        f"{row['region']}: TabPFN forecasts {row['median_per_100k']:,.0f} respiratory illnesses per 100,000 "
        f"(80% range {row['q10_per_100k']:,.0f}–{row['q90_per_100k']:,.0f}; "
        f"high-activity threshold {row['high_threshold_per_100k']:,.0f}). "
        f"Your {component['component']} has {component['exposure']} exposure: {component['inputs']}."
    )


def germany_overview(service, activity_date) -> list[dict]:
    """Read exactly the selected week's four regional forecasts, never latest-other-week."""
    from datetime import timedelta

    target = activity_date - timedelta(days=activity_date.weekday())
    bundle, error = service._bundle()
    selected = (
        {}
        if error
        else {
            f.location_id: f
            for f in bundle.forecasts
            if f.indicator == "are" and f.target_week_start == target
        }
    )
    rows = []
    for region in ("east", "south", "central_west", "north_west"):
        forecast = selected.get(region)
        freshness = service._freshness(forecast) if forecast else "forecast_unavailable"
        if activity_date < service._today():
            freshness = "activity_date_in_past"
        level = "unknown"
        if forecast and freshness == "fresh":
            level = (
                "high"
                if forecast.q50 >= forecast.reference_p80
                else "moderate"
                if forecast.q50 >= forecast.reference_p50
                else "low"
            )
        rows.append(
            {
                "region_id": region,
                "level": level,
                "freshness": freshness,
                "q50": forecast.q50 if forecast and freshness == "fresh" else None,
                "q10": forecast.q10 if forecast and freshness == "fresh" else None,
                "q90": forecast.q90 if forecast and freshness == "fresh" else None,
                "week_start": target.isoformat(),
                "week_end": (target + timedelta(days=6)).isoformat(),
            }
        )
    return rows
