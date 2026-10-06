import pytest

from infectionpulse.activity_policy import Activity, assess_policy, exposure_level


@pytest.mark.parametrize(
    "burden,q50,exposure,expected",
    [
        ("low", 20, "low", "green"),
        ("low", 20, "moderate", "green"),
        ("low", 20, "high", "yellow"),
        ("moderate", 60, "low", "green"),
        ("moderate", 60, "moderate", "yellow"),
        ("moderate", 60, "high", "yellow"),
        ("high", 90, "low", "green"),
        ("high", 90, "moderate", "yellow"),
        ("high", 90, "high", "red"),
    ],
)
def test_frozen_matrix(burden, q50, exposure, expected):
    args = {
        "low": dict(setting="outdoor", crowding="uncrowded", duration_minutes=60),
        "moderate": dict(
            setting="indoor",
            ventilation="good",
            crowding="uncrowded",
            duration_minutes=60,
        ),
        "high": dict(setting="indoor", ventilation="unknown", duration_minutes=60),
    }[exposure]
    activity = Activity(**args)
    result = assess_policy(
        activity, [dict(q50=q50, q90=q50, reference_p50=50, reference_p80=80)]
    )
    assert result.traffic_light == expected
    assert result.burden == burden
    assert result.exposure == exposure


def test_upper_quantile_prevents_false_reassurance():
    activity = Activity(setting="indoor", ventilation="good", duration_minutes=15)
    result = assess_policy(
        activity, [dict(q50=10, q90=90, reference_p50=50, reference_p80=80)]
    )
    assert result.traffic_light == "yellow"
    assert "upper_forecast_crosses_high_reference" in result.reason_codes


def test_solitary_car_is_different_from_shared_transit():
    common = dict(
        setting="outdoor", crowding="uncrowded", duration_minutes=30, commute_minutes=45
    )
    assert exposure_level(Activity(**common, commute_mode="car_alone")) == "low"
    assert exposure_level(Activity(**common, commute_mode="public_transit")) == "high"


def test_missing_evidence_is_grey_even_outdoors():
    result = assess_policy(
        Activity(setting="outdoor", crowding="uncrowded", duration_minutes=10),
        [],
        "source_stale",
    )
    assert result.traffic_light == "grey"
    assert result.reason_codes == ["source_stale"]


def test_high_but_flat_does_not_need_surge_probability():
    result = assess_policy(
        Activity(category="work", setting="indoor", duration_minutes=60),
        [dict(q50=100, q90=110, reference_p50=50, reference_p80=80)],
    )
    assert result.traffic_light == "red"
    assert "home office" in result.recommendation


def test_low_exposure_stays_green_when_community_burden_is_high():
    result = assess_policy(
        Activity(setting="outdoor", crowding="uncrowded", duration_minutes=60),
        [dict(q50=100, q90=110, reference_p50=50, reference_p80=80)],
    )
    assert result.traffic_light == "green"
    assert "community_burden_high" in result.reason_codes
    assert "outdoor alternative" not in result.recommendation
