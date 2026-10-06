import pytest

from infectionpulse.activity_policy import Activity, assess_policy
from infectionpulse.risk_scores import precaution_score


@pytest.mark.parametrize("incidence", [10, 60, 90])
@pytest.mark.parametrize("exposure", ["low", "moderate", "high"])
def test_scores_follow_policy_colours_and_never_claim_probabilities(
    incidence, exposure
):
    activity = Activity(
        setting="outdoor" if exposure == "low" else "indoor",
        duration_minutes=60,
        crowding="uncrowded",
        ventilation="poor" if exposure == "high" else "good",
    )
    row = {"q50": incidence, "q90": incidence, "reference_p50": 50, "reference_p80": 80}
    policy = assess_policy(activity, [row]).model_dump()
    score = precaution_score(policy)
    assert 0 <= score.value <= 100
    expected = "green" if score.value < 40 else "yellow" if score.value < 70 else "red"
    assert policy["traffic_light"] == expected
    assert score.kind == "ordinal_policy_index"
    assert "not a probability" in score.interpretation


def test_unknown_is_not_zero_and_uncertainty_yellow_has_matching_score():
    activity = Activity(
        setting="indoor", duration_minutes=10, ventilation="good", crowding="uncrowded"
    )
    assert precaution_score(assess_policy(activity, []).model_dump()).value is None
    policy = assess_policy(
        activity, [{"q50": 10, "q90": 90, "reference_p50": 50, "reference_p80": 80}]
    )
    assert policy.traffic_light == "yellow"
    assert precaution_score(policy.model_dump()).value == 40


def test_score_increases_with_exposure_and_burden():
    levels = ["low", "moderate", "high"]
    from infectionpulse.risk_scores import SCORES

    for burden in levels:
        values = [SCORES[burden][exposure] for exposure in levels]
        assert values == sorted(values)
    for exposure in levels:
        values = [SCORES[burden][exposure] for burden in levels]
        assert values == sorted(values)
