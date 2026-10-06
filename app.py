"""A compact, cached respiratory advisory: score, reason, map and disease context."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from infectionpulse.activity_policy import Activity
from infectionpulse.explanations import explain_assessment
from infectionpulse.germany_map import build_germany_map, build_state_observation_map
from infectionpulse.pathogen_data import read_pathogen_context
from infectionpulse.risk_scores import brief_reason, germany_overview, precaution_score
from infectionpulse.service import AssessmentRequest, ForecastService

ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "examples/replay"
ICONS = {"green": "🟢", "yellow": "🟡", "red": "🔴", "grey": "⚪"}
st.set_page_config(page_title="InfectionPulse", page_icon="◉", layout="centered")
st.title("InfectionPulse")
st.caption("Respiratory illness outlook for your next week.")
service = ForecastService()
bundle, _ = service._bundle()
has_current = bundle is not None and any(
    f.indicator == "are" and f.target_week_end >= service._today()
    for f in bundle.forecasts
)
recorded = st.sidebar.toggle(
    "Recorded example",
    value=not has_current,
    help="Replays a real saved TabPFN forecast at the time it was issued.",
)
if recorded:
    recorded_at = datetime.fromisoformat(
        json.loads((REPLAY / "metadata.json").read_text())["recorded_at"]
    )
    service = ForecastService(
        bundle_path=REPLAY / "forecast_bundle.json",
        locations_path=ROOT / "examples/locations.json",
        now=lambda: recorded_at,
        source_status_path=REPLAY / "source_status.json",
        observations_path=REPLAY / "covid_observations.json",
        pathogens_path=REPLAY / "pathogen_observations.json",
    )
    st.info(
        f"Recorded example: real TabPFN forecast as issued on "
        f"{recorded_at:%d %b %Y}. Not current advice."
    )
locations = service._locations()
if not locations:
    st.info("District data is unavailable. Run the documented data setup first.")
    st.stop()
labels = {r["location_id"]: f"{r['name']} · {r['location_id']}" for r in locations}
ids = list(labels)
with st.sidebar:
    st.header("Your plan")
    home = st.selectbox(
        "Home district",
        ids,
        index=ids.index("11000") if "11000" in ids else 0,
        format_func=labels.get,
    )
    destination = st.selectbox(
        "Work or activity district",
        ids,
        index=ids.index("12054") if "12054" in ids else 0,
        format_func=labels.get,
    )
    current = service.get_forecast(destination)
    available = [
        date.fromisoformat(f["target_week_start"])
        for f in current["forecasts"]
        if f["indicator"] == "are"
        and date.fromisoformat(f["target_week_end"]) >= service._today()
    ]
    default_date = (
        max(service._today(), min(available))
        if available
        else service._today() + timedelta(days=7 - service._today().weekday())
    )
    activity_date = st.date_input("Activity date", default_date)
    category = st.selectbox(
        "Activity", ["work", "social", "travel", "other"], format_func=str.capitalize
    )
    setting = st.radio(
        "Setting", ["indoor", "outdoor"], format_func=str.capitalize, horizontal=True
    )
    commute = st.selectbox(
        "Commute",
        ["public_transit", "car_alone", "bike", "walk", "shared_car", "none"],
        format_func={
            "public_transit": "Public transport",
            "car_alone": "Car, alone",
            "bike": "Bike",
            "walk": "Walking",
            "shared_car": "Shared car",
            "none": "No commute",
        }.get,
    )
    with st.expander("Activity details"):
        shared = st.checkbox("Sharing the space with other people", True)
        duration = st.number_input("Activity duration (minutes)", 0, 1440, 120, 15)
        crowding = st.selectbox("Crowding", ["unknown", "uncrowded", "crowded"])
        ventilation = st.selectbox("Ventilation quality", ["unknown", "good", "poor"])
        commute_minutes = st.number_input("Commute duration (minutes)", 0, 720, 30, 5)
        commute_crowding = st.selectbox(
            "Commute crowding", ["unknown", "uncrowded", "crowded"]
        )

request = AssessmentRequest(
    home_location_id=home,
    destination_location_id=destination,
    activity_date=activity_date,
    activity=Activity(
        category=category,
        setting=setting,
        duration_minutes=duration,
        shared_space=shared,
        crowding=crowding,
        ventilation=ventilation,
        commute_mode=commute,
        commute_minutes=commute_minutes,
        commute_crowding=commute_crowding,
    ),
)
result = service.assess_activity(request)
# Tolerate an older in-memory service during a Streamlit source update.
if not result.get("explanation"):
    result["explanation"] = explain_assessment(
        result, request.activity.model_dump()
    ).model_dump(mode="json")
if not result.get("precaution_score"):
    result["precaution_score"] = precaution_score(result["assessment"]).model_dump()
assessment, explanation = result["assessment"], result["explanation"]
light, score = assessment["traffic_light"], result["precaution_score"]["value"]

st.caption(f"{activity_date:%A, %d %B} · {labels[destination]}")
with st.container(border=True):
    advice, number = st.columns([2, 1])
    title = {
        "green": "Usual precautions",
        "yellow": "Take care at indoor stops"
        if assessment["exposure"] == "low"
        else "Mask in shared indoor spaces",
        "red": "Consider home office"
        if category == "work"
        else "Consider changing your plan",
        "grey": "Not enough current evidence",
    }[light]
    advice.subheader(f"{ICONS[light]} {title}")
    number.metric("Precaution score", f"{score}/100" if score is not None else "—")
    st.write(brief_reason(result))
    st.caption(
        "Score: a fixed rule on the TabPFN forecast and your plan, not your probability of infection."
    )
    better = [
        a
        for a in explanation["alternatives"]
        if score is not None and a.get("score") is not None and a["score"] < score
    ]
    if better:
        alternative = min(better, key=lambda a: a["score"])
        st.markdown(
            f"**Alternative:** {alternative['title']} → {ICONS[alternative['traffic_light']]} **{alternative['score']}/100**"
        )

st.subheader("Germany · respiratory illness")
map_layer = st.selectbox(
    "Map view",
    [
        "Respiratory forecast · 4 regions",
        "Flu reports · 16 states",
        "RSV reports · 16 states",
    ],
)
if map_layer.startswith("Respiratory"):
    overview = germany_overview(service, activity_date)
    st.caption(
        f"Forecast week {overview[0]['week_start']}–{overview[0]['week_end']} · four forecast regions; equal colours mean the same activity category"
    )
    figure = build_germany_map(overview)
else:
    indicator = "influenza" if map_layer.startswith("Flu") else "rsv"
    path = getattr(
        service, "pathogens_path", ROOT / "data/service/pathogen_observations.json"
    )
    state_reports = [
        row
        for state in range(1, 17)
        for row in read_pathogen_context(path, f"{state:02d}", service.now())
        if row["indicator"] == indicator
    ]
    periods = sorted(
        {
            f"{row['week_start']}–{row['observed_through']}"
            for row in state_reports
            if row.get("freshness") == "fresh"
        }
    )
    st.caption(
        f"Reported week {', '.join(periods)} · 16 individual state figures · observations, not next-week forecasts"
        if periods
        else "No current state reports available; grey means missing or stale."
    )
    figure = build_state_observation_map(state_reports, indicator)
figure.update_layout(height=460)
st.plotly_chart(
    figure,
    width="stretch",
    config={"displayModeBar": False, "scrollZoom": False},
)
st.caption(
    "Hover over a state for its value and date. Map values are not your personal infection probability."
)

st.subheader("Flu, RSV & COVID")
st.caption(
    "Latest reported cases per 100,000 near your destination. These are observations, not separate disease forecasts."
)
pathogens = result.get("disease_context", [])
for column, (indicator, title) in zip(
    st.columns(3), [("influenza", "Flu"), ("rsv", "RSV"), ("covid", "COVID-19")]
):
    rows = result["local_context"] if indicator == "covid" else pathogens
    rows = [
        r
        for r in rows
        if r["indicator"] == indicator
        and r["location_id"]
        == (destination if indicator == "covid" else destination[:2])
    ]
    row = max(rows, key=lambda r: r.get("observed_through") or "") if rows else None
    fresh = bool(
        row and row.get("freshness") == "fresh" and row.get("value") is not None
    )
    column.metric(title, f"{row['value']:.2f}" if fresh else "—")
    if row:
        area = "District" if indicator == "covid" else "State"
        date_label = row.get("observed_through") or "unknown date"
        column.caption(
            f"{area} · through {date_label}"
            + (" · unavailable/stale" if not fresh else "")
        )
    else:
        column.caption("No current report available")
st.caption(
    "The score uses the broad respiratory forecast. Flu and RSV map layers show separate reported observations; pathogen reports are not added together."
)

with st.expander("Sources & how the score works"):
    st.write(
        "The score is a coarse policy scale: below 40 is green, 40–69 yellow, and 70 or more red. Community burden and exposure select its value; an uncertain upper forecast can raise it to at least 40. It is not a measured infection probability."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Community activity": ["Low", "Moderate", "High"],
                "Lower exposure": [10, 25, 50],
                "Moderate exposure": [25, 50, 65],
                "Higher exposure": [45, 60, 85],
            }
        ),
        hide_index=True,
    )
    for f in result["forecasts"]:
        st.write(
            f"{f['location_id']}: median {f['q50']:,.0f}, 10th–90th percentile {f['q10']:,.0f}–{f['q90']:,.0f} per 100,000. Observations through {f['observed_through']}; forecast issued {f['issued_at']}."
        )
        st.markdown(f"[RKI GrippeWeb source]({f['source_url']})")
    for row in pathogens:
        if row.get("source_url"):
            st.markdown(
                f"[{row['indicator'].upper()} · RKI source]({row['source_url']})"
            )
    for row in result["local_context"]:
        st.markdown(f"[COVID-19 · RKI source]({row['source_url']})")
    st.write(
        "ARE covers acute respiratory symptoms from multiple causes; it does not identify the pathogen or cover all infectious diseases. Disease-specific observations are supporting context and do not change the score numerically. The precaution thresholds are prototype rules, not clinical danger thresholds."
    )
    probability = explanation["probability"]
    if probability["status"] == "available":
        st.write(probability["definition"])
        for row in probability["regions"]:
            st.write(f"{row['region']}: {row['value']:.0%}")
    else:
        st.caption(
            "Community-event probabilities remain withheld until calibration and output checks pass."
        )
    st.json(result, expanded=False)

with st.expander("Model benchmark"):
    status_path = ROOT / "models/respiratory/latest_run.json"
    summary_path = ROOT / "reports/benchmark_summary.csv"
    if status_path.exists() or summary_path.exists():
        if status_path.exists():
            status = json.loads(status_path.read_text())
            table = pd.DataFrame(status.get("summary", []))
        else:
            table = pd.read_csv(summary_path)
        if not table.empty:
            st.dataframe(
                table[["model", "mae", "rmse"]].rename(
                    columns={
                        "model": "Model",
                        "mae": "Mean absolute error",
                        "rmse": "RMSE",
                    }
                ),
                hide_index=True,
            )
        st.caption(
            "Lower error is better. Historical forecast accuracy does not validate personal infection odds or the precaution policy."
        )
    else:
        st.info("Benchmark results are not available yet.")

    alerts_path = ROOT / "reports/alert_summary.csv"
    if alerts_path.exists():
        alerts = pd.read_csv(alerts_path)
        st.caption(
            "Warning usefulness: high community burden, before personal activity rules. "
            "Exploratory retrospective analysis; this does not measure avoided infections."
        )
        selected_alerts = alerts[alerts.policy.eq("high_burden")]
        st.dataframe(
            selected_alerts[
                [
                    "model",
                    "true_alerts",
                    "missed_events",
                    "false_alerts",
                    "precision",
                    "recall",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "true_alerts": "Detected",
                    "missed_events": "Missed",
                    "false_alerts": "False alerts",
                    "precision": "Precision",
                    "recall": "Recall",
                }
            ),
            hide_index=True,
        )
        st.caption(
            "Compare the misses and false alerts together. Uncertainty in the TabPFN–full-history LightGBM difference includes no advantage."
        )

    state_path = ROOT / "reports/state_summary.csv"
    if state_path.exists():
        states = pd.read_csv(state_path)
        st.caption(
            "Experimental 16-state disease forecasts: a separate 12-date pilot. "
            "Lower MAE is better; these models do not change your precaution score."
        )
        st.dataframe(
            states.loc[
                states.regime.eq("all"),
                ["indicator", "model", "rows", "mae", "coverage_80"],
            ].rename(
                columns={
                    "indicator": "Disease",
                    "model": "Model",
                    "rows": "Matched state-weeks",
                    "mae": "MAE",
                    "coverage_80": "Raw 80% interval coverage",
                }
            ),
            hide_index=True,
        )
        st.caption(
            "RSV history includes a reporting-system error corrected in February 2026. "
            "The report separates that period; raw intervals are not calibrated."
        )
