import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_service_dashboard_loads_and_activity_changes():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    assert not app.exception
    assert app.title[0].value == "InfectionPulse"
    assert any(m.label == "Precaution score" for m in app.metric)
    assert any(h.value == "Germany · respiratory illness" for h in app.subheader)
    assert len(app.get("plotly_chart")) == 1
    assert {m.label for m in app.metric} == {
        "Precaution score",
        "Flu",
        "RSV",
        "COVID-19",
    }
    assert any(e.label == "Sources & how the score works" for e in app.expander)
    layer = next(w for w in app.selectbox if w.label == "Map view")
    assert layer.value == "Respiratory forecast · 4 regions"
    for value in ["Flu reports · 16 states", "RSV reports · 16 states"]:
        layer = next(w for w in app.selectbox if w.label == "Map view")
        layer.select(value).run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 1
    if app.sidebar.selectbox:
        app.sidebar.selectbox[0].select("01001").run()
        app.sidebar.radio[0].set_value("outdoor").run()
        assert not app.exception


def test_recorded_example_gives_a_dated_recommendation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.toggle[0].set_value(True).run()
    assert not app.exception
    assert any("Recorded example" in i.value for i in app.info)
    score = next(m for m in app.metric if m.label == "Precaution score")
    assert score.value.endswith("/100")


@pytest.mark.parametrize("missing_field", [True, False])
def test_dashboard_handles_pre_explanation_service_response(monkeypatch, missing_field):
    from infectionpulse.service import ForecastService

    def old_response(self, request):
        result = self._assess_activity(request)
        if missing_field:
            result.pop("explanation", None)
        else:
            result["explanation"] = None
        return result

    monkeypatch.setattr(ForecastService, "assess_activity", old_response)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    assert not app.exception
    assert any(m.label == "Precaution score" for m in app.metric)
    payloads = [json.loads(element.value) for element in app.json]
    assessment = next(p for p in payloads if "assessment" in p)
    assert assessment["explanation"]["method"] == "exact_policy_trace"
    assert assessment["explanation"]["summary"]
    app.sidebar.radio[0].set_value("outdoor").run()
    assert not app.exception
