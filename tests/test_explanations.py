from infectionpulse.explanations import probability_summary


def row(region="east", ready=True, value=0.72):
    return {
        "location_id": region,
        "forecast_id": region + "-1",
        "probability_readiness": {"ready": ready, "reliability_ece": 0.03},
        "probability_above_seasonal_reference": value,
        "seasonal_reference_p80": 5000,
        "probability_lower_bound": 0.71,
        "probability_upper_bound": 0.73,
    }


def test_probability_requires_readiness_and_never_recovers_hidden_bounds():
    for forecast in (row(ready=False), row(value=None), row(value=float("nan"))):
        result = probability_summary([forecast], False)
        assert result["status"] == "withheld"
        assert result["value"] is None
        assert result["regions"][0]["value"] is None
        assert "0.71" not in str(result)


def test_validated_probability_labels_event_and_keeps_regions_separate():
    single = probability_summary([row()], False)
    assert single["value"] == 0.72
    assert "not your chance of infection" in single["definition"]
    multiple = probability_summary([row(), row("south", value=0.4)], False)
    assert multiple["status"] == "available"
    assert multiple["value"] is None
    assert [r["value"] for r in multiple["regions"]] == [0.72, 0.4]
    assert multiple["personal_infection_probability"] is None


def test_stale_or_missing_forecast_has_no_probability_even_when_calibrated():
    result = probability_summary([row()], True)
    assert result["status"] == "unavailable"
    assert result["regions"] == []
    assert result["value"] is None
