"""Cached public forecasts exposed identically to HTTP and MCP clients.

No import of modeling code, no downloads and no inference in request handling.
"""

import csv
import json
import os
import secrets
import unicodedata
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from infectionpulse.activity_policy import Activity, PolicyResult, assess_policy
from infectionpulse.explanations import DecisionExplanation, explain_assessment
from infectionpulse.risk_scores import PrecautionScore, precaution_score

ROOT = Path(__file__).resolve().parents[1]


class LocationRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    location_id: str = Field(pattern=r"^\d{5}$")
    name: str
    region_id: str
    geographic_level: Literal["district"] = "district"


class LocationResolution(BaseModel):
    status: str
    candidates: list[LocationRecord]
    total_matches: int = 0


class ForecastRecord(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    forecast_id: str
    model: str
    model_version: str
    location_id: str
    geographic_level: Literal["macroregion", "district", "national"]
    indicator: Literal["are", "covid"]
    units: str
    observed_through: date
    published_at: datetime | None = None
    fetched_at: datetime
    issued_at: datetime
    target_week_start: date
    target_week_end: date
    q10: float = Field(ge=0)
    q50: float = Field(ge=0)
    q90: float = Field(ge=0)
    reference_p50: float = Field(ge=0)
    reference_p80: float = Field(gt=0)
    reference_end: date
    source_url: str
    freshness: str = "unchecked"
    newer_source_available: bool = False

    @model_serializer(mode="wrap")
    def public_forecast(self, handler):
        data = handler(self)
        readiness = data.get("probability_readiness")
        ready = isinstance(readiness, dict) and readiness.get("ready") is True
        if not ready or data.get("probability_above_seasonal_reference") is None:
            for key in (
                "probability_above_seasonal_reference",
                "probability_lower_bound",
                "probability_upper_bound",
            ):
                if key in data:
                    data[key] = None
        return data

    @model_validator(mode="after")
    def coherent(self):
        if not self.q10 <= self.q50 <= self.q90:
            raise ValueError("Forecast quantiles must be ordered")
        if self.reference_p50 > self.reference_p80:
            raise ValueError("Reference percentiles must be ordered")
        if self.target_week_end != self.target_week_start + timedelta(days=6):
            raise ValueError("Forecast target must cover exactly one calendar week")
        if self.target_week_start.weekday() != 0:
            raise ValueError("Forecast week must begin Monday")
        if self.reference_end >= self.target_week_start:
            raise ValueError("Reference data must precede target week")
        for value in (self.fetched_at, self.issued_at, self.published_at):
            if value is not None and value.tzinfo is None:
                raise ValueError("Provenance timestamps require a timezone")
        return self


class ForecastBundle(BaseModel):
    schema_version: Literal[1]
    generated_at: datetime
    forecasts: list[ForecastRecord]

    @model_validator(mode="after")
    def unique(self):
        keys = [
            (f.location_id, f.indicator, f.target_week_start) for f in self.forecasts
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Ambiguous duplicate forecasts in bundle")
        return self


class LocalObservation(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    location_id: str = Field(pattern=r"^\d{5}$")
    indicator: Literal["covid"] = "covid"
    geographic_level: Literal["district"] = "district"
    kind: Literal["observation"] = "observation"
    observed_through: date
    value: float = Field(ge=0)
    units: str
    source_url: str
    source_snapshot_id: str
    fetched_at: datetime
    freshness: str = "unchecked"


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home_location_id: str = Field(pattern=r"^\d{5}$")
    destination_location_id: str = Field(pattern=r"^\d{5}$")
    activity_date: date
    activity: Activity


class AssessmentComponent(BaseModel):
    component: Literal["destination", "commute"]
    location_ids: list[str]
    forecast_ids: list[str]
    assessment: PolicyResult


class AssessmentResponse(BaseModel):
    schema_version: Literal[1] = 1
    activity_date: date
    home_location_id: str
    destination_location_id: str
    assessment: PolicyResult
    explanation: DecisionExplanation | None = None
    precaution_score: PrecautionScore | None = None
    dominant_component: Literal["destination", "commute"] = "destination"
    components: list[AssessmentComponent] = Field(default_factory=list)
    forecasts: list[ForecastRecord]
    supplementary_signals: list[ForecastRecord] = Field(default_factory=list)
    local_context: list[LocalObservation] = Field(default_factory=list)
    disease_context: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=lambda: [
            "ARE estimates apply to the named macroregion, not individual districts.",
            "Exposure rules are transparent assumptions, not learned personal infection probabilities.",
            "District COVID notifications are a separate reported-case indicator and are not combined numerically with ARE.",
            "State influenza and RSV reports are observed context; pathogen counts are not added to the broad ARE forecast or precaution score.",
            "Commute guidance uses home and destination regions, not measured conditions on a particular train or route.",
        ]
    )


def _normalize(value):
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(value).casefold())
        if not unicodedata.combining(c)
    )


class ForecastService:
    def __init__(
        self,
        bundle_path=None,
        locations_path=None,
        now=None,
        source_status_path=None,
        observations_path=None,
        pathogens_path=None,
    ):
        self.bundle_path = Path(
            bundle_path
            or os.getenv(
                "INFECTIONPULSE_BUNDLE", ROOT / "models/service/forecast_bundle.json"
            )
        )
        self.locations_path = Path(
            locations_path
            or os.getenv(
                "INFECTIONPULSE_LOCATIONS", ROOT / "data/service/locations.json"
            )
        )
        self.custom_locations = bool(
            locations_path or os.getenv("INFECTIONPULSE_LOCATIONS")
        )
        self.now = now or (lambda: datetime.now(UTC))
        self.source_status_path = Path(
            source_status_path
            or os.getenv(
                "INFECTIONPULSE_SOURCE_STATUS",
                ROOT / "data/respiratory/source_status.json",
            )
        )
        self.observations_path = Path(
            observations_path
            or os.getenv(
                "INFECTIONPULSE_OBSERVATIONS",
                ROOT / "data/service/covid_observations.json",
            )
        )
        self.pathogens_path = Path(
            pathogens_path
            or os.getenv(
                "INFECTIONPULSE_PATHOGENS",
                ROOT / "data/service/pathogen_observations.json",
            )
        )

    def _disease_context(self, location_id):
        if not self.pathogens_path.exists():
            return []
        from infectionpulse.pathogen_data import read_pathogen_context

        return read_pathogen_context(self.pathogens_path, location_id[:2], self.now())

    def _today(self):
        return self.now().astimezone(ZoneInfo("Europe/Berlin")).date()

    def _locations(self):
        catalog_path = self.locations_path
        if not catalog_path.exists():
            if self.custom_locations:
                return []
            packaged = ROOT / "examples/locations.json"
            if packaged.exists():
                catalog_path = packaged
        if not catalog_path.exists():
            from infectionpulse.respiratory_data import region_for_district

            path = ROOT / "data/demographics/district_demographics_reference.csv"
            if not path.exists():
                return []
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            return [
                {
                    "location_id": r["Kreisschluessel"].zfill(5),
                    "name": r["district_name"],
                    "region_id": region_for_district(r["Kreisschluessel"].zfill(5)),
                    "geographic_level": "district",
                }
                for r in rows
            ]
        try:
            if catalog_path.suffix == ".csv":
                with catalog_path.open() as stream:
                    rows = list(csv.DictReader(stream))
            else:
                data = json.loads(catalog_path.read_text())
                rows = data.get("locations", []) if isinstance(data, dict) else data
            return [LocationRecord.model_validate(r).model_dump() for r in rows]
        except (ValueError, TypeError, OSError):
            return []

    def _bundle(self):
        if not self.bundle_path.exists():
            return None, "forecast_artifact_missing"
        try:
            return ForecastBundle.model_validate_json(
                self.bundle_path.read_text()
            ), None
        except (ValueError, OSError):
            return None, "forecast_artifact_invalid"

    def _freshness(self, f):
        now = self.now()
        if (
            f.issued_at > now
            or f.fetched_at > now
            or (f.published_at and f.published_at > now)
        ):
            return "future_provenance"
        if self._today() - f.observed_through > timedelta(days=14):
            return "observations_stale"
        if f.published_at is None:
            return "publication_time_unknown"
        release_date = f.published_at.astimezone(ZoneInfo("Europe/Berlin")).date()
        if self._today() - release_date > timedelta(days=10):
            return "source_release_stale"
        checked_at, _ = self._poll_status(f)
        if now - checked_at > timedelta(hours=72):
            return "source_check_stale"
        return "fresh"

    def _poll_status(self, f):
        checked_at, newer = f.fetched_at, False
        if self.source_status_path.exists() and getattr(f, "source_snapshot_id", None):
            try:
                status = json.loads(self.source_status_path.read_text())
                checked = datetime.fromisoformat(
                    status["latest_checked_at"].replace("Z", "+00:00")
                )
                if (
                    checked.tzinfo is not None
                    and checked <= self.now()
                    and status["last_check_status"]
                    in ("ok", "unchanged", "updated", "success")
                ):
                    newer = status["latest_snapshot_id"] != f.source_snapshot_id
                    checked_at = max(checked_at, checked)
            except (ValueError, KeyError, TypeError, OSError):
                pass
        return checked_at, newer

    def _local_context(self, location_id):
        if not self.observations_path.exists():
            return []
        try:
            payload = json.loads(self.observations_path.read_text())
            records = [
                LocalObservation.model_validate(r)
                for r in payload["observations"]
                if r["location_id"] == location_id
            ]
        except (ValueError, KeyError, TypeError, OSError):
            return []
        if not records:
            return []
        latest = max(records, key=lambda r: r.observed_through)
        if (
            latest.fetched_at.tzinfo is None
            or latest.fetched_at > self.now()
            or latest.observed_through > self._today()
        ):
            latest.freshness = "invalid_provenance"
        elif self._today() - latest.observed_through > timedelta(days=14):
            latest.freshness = "observations_stale"
        elif self.now() - latest.fetched_at > timedelta(hours=72):
            latest.freshness = "source_check_stale"
        else:
            latest.freshness = "fresh"
        return [latest.model_dump(mode="json")]

    def resolve_location(self, query: str):
        query = query.strip()
        if not query or len(query) > 160:
            return {"status": "invalid_query", "candidates": []}
        rows = self._locations()
        if not rows:
            return {"status": "location_catalog_unavailable", "candidates": []}
        exact = [r for r in rows if str(r["location_id"]) == query]
        matches = exact or [
            r
            for r in rows
            if _normalize(query) in _normalize(r["name"])
            or query in str(r["location_id"])
        ]
        return {
            "status": "resolved"
            if len(matches) == 1
            else "ambiguous"
            if matches
            else "not_found",
            "candidates": matches[:20],
            "total_matches": len(matches),
        }

    def get_forecast(self, location_id: str, target_week_start: str | None = None):
        location = next(
            (r for r in self._locations() if r["location_id"] == location_id), None
        )
        if not location:
            return {
                "status": "location_not_found",
                "location_id": location_id,
                "forecasts": [],
            }
        bundle, error = self._bundle()
        if error:
            return {
                "status": error,
                "location": location,
                "forecasts": [],
                "local_context": self._local_context(location_id),
                "disease_context": self._disease_context(location_id),
            }
        try:
            target = (
                date.fromisoformat(target_week_start) if target_week_start else None
            )
        except ValueError:
            return {"status": "invalid_date", "location": location, "forecasts": []}
        candidates = [
            f
            for f in bundle.forecasts
            if (f.indicator == "are" and f.location_id == location["region_id"])
            or (f.indicator == "covid" and f.location_id == location_id)
        ]
        if target:
            candidates = [f for f in candidates if f.target_week_start == target]
        else:
            candidates = [f for f in candidates if f.target_week_end >= self._today()]
        forecasts = [
            {
                **f.model_dump(mode="json"),
                "freshness": self._freshness(f),
                "newer_source_available": self._poll_status(f)[1],
            }
            for f in candidates
        ]
        return {
            "status": "available" if forecasts else "forecast_unavailable",
            "location": location,
            "forecasts": forecasts,
            "generated_at": bundle.generated_at.isoformat(),
            "local_context": self._local_context(location_id),
            "disease_context": self._disease_context(location_id),
        }

    def assess_activity(self, request: AssessmentRequest):
        result = self._assess_activity(request)
        result["precaution_score"] = precaution_score(result["assessment"]).model_dump()
        explanation = explain_assessment(result, request.activity.model_dump())
        if result["assessment"]["traffic_light"] != "grey":
            alternatives = []
            if (
                request.activity.commute_mode in ("public_transit", "shared_car")
                and request.activity.commute_minutes > 0
            ):
                alternatives.append(
                    (
                        "Travel alone instead",
                        "Only the commute changes to driving alone; the activity and its duration stay the same.",
                        request.model_copy(
                            update={
                                "activity": request.activity.model_copy(
                                    update={"commute_mode": "car_alone"}
                                )
                            }
                        ),
                    )
                )
            if request.activity.category == "work":
                alternatives.append(
                    (
                        "Work from home alone",
                        "The activity moves to your home district, with no shared workspace and no commute.",
                        request.model_copy(
                            update={
                                "destination_location_id": request.home_location_id,
                                "activity": request.activity.model_copy(
                                    update={
                                        "shared_space": False,
                                        "commute_mode": "none",
                                        "commute_minutes": 0,
                                    }
                                ),
                            }
                        ),
                    )
                )
            else:
                alternatives.append(
                    (
                        "Meet outdoors without a crowd",
                        "Only the setting and crowding change; your commute and activity duration stay the same.",
                        request.model_copy(
                            update={
                                "activity": request.activity.model_copy(
                                    update={
                                        "setting": "outdoor",
                                        "crowding": "uncrowded",
                                    }
                                )
                            }
                        ),
                    )
                )
            for title, description, changed in alternatives:
                outcome = self._assess_activity(changed)
                explanation.alternatives.append(
                    {
                        "title": title,
                        "description": description,
                        "traffic_light": outcome["assessment"]["traffic_light"],
                        "score": precaution_score(outcome["assessment"]).value,
                        "original_traffic_light": result["assessment"]["traffic_light"],
                        "recommendation": outcome["assessment"]["recommendation"],
                        "interpretation": "Recomputed policy scenario, not a measured reduction in infection probability.",
                    }
                )
        result["explanation"] = explanation.model_dump(mode="json")
        return result

    def _assess_activity(self, request: AssessmentRequest):
        global_unavailable = (
            "activity_date_in_past" if request.activity_date < self._today() else None
        )
        target_start = request.activity_date - timedelta(
            days=request.activity_date.weekday()
        )
        selected, supplementary, context, per_location, errors = {}, {}, {}, {}, {}
        diseases = {}
        for location_id in {request.home_location_id, request.destination_location_id}:
            response = self.get_forecast(location_id, target_start.isoformat())
            if response["status"] == "location_not_found":
                global_unavailable = global_unavailable or "location_not_found"
            records = response["forecasts"]
            for observation in response.get("disease_context", []):
                diseases[(observation["indicator"], observation["location_id"])] = (
                    observation
                )
            for observation in response.get("local_context", []):
                context[observation["location_id"]] = LocalObservation.model_validate(
                    observation
                )
            are = [f for f in records if f["indicator"] == "are"]
            errors[location_id] = (
                response["status"] if response["status"] != "available" else None
            )
            if not are and not errors[location_id]:
                errors[location_id] = "are_forecast_unavailable"
            per_location[location_id] = []
            for record in records:
                fresh = record["freshness"]
                forecast = ForecastRecord.model_validate(record)
                if forecast.indicator == "are":
                    if fresh != "fresh":
                        errors[location_id] = errors[location_id] or fresh
                    per_location[location_id].append(forecast)
                else:
                    supplementary[forecast.forecast_id] = forecast
        components = []

        def add_component(name, locations, activity):
            forecasts = {
                f.forecast_id: f
                for location in locations
                for f in per_location[location]
            }
            selected.update(forecasts)
            unavailable = global_unavailable or next(
                (errors[location] for location in locations if errors[location]), None
            )
            policy = assess_policy(
                activity, [f.model_dump() for f in forecasts.values()], unavailable
            )
            component = AssessmentComponent(
                component=name,
                location_ids=locations,
                forecast_ids=list(forecasts),
                assessment=policy,
            )
            components.append(component)
            return component

        destination_activity = request.activity.model_copy(
            update={"commute_mode": "none", "commute_minutes": 0}
        )
        add_component(
            "destination", [request.destination_location_id], destination_activity
        )
        if (
            request.activity.commute_mode != "none"
            and request.activity.commute_minutes > 0
        ):
            shared = request.activity.commute_mode in ("public_transit", "shared_car")
            travel = Activity(
                category="travel",
                setting="indoor" if shared else "outdoor",
                duration_minutes=request.activity.commute_minutes,
                shared_space=shared,
                crowding=request.activity.commute_crowding if shared else "uncrowded",
                ventilation="unknown" if shared else "good",
            )
            component = add_component(
                "commute",
                list(
                    dict.fromkeys(
                        [request.home_location_id, request.destination_location_id]
                    )
                ),
                travel,
            )
            light = component.assessment.traffic_light
            if light == "red":
                option = (
                    "home office or quieter travel"
                    if request.activity.category == "work"
                    else "quieter travel or a different travel time"
                )
                component.assessment.recommendation = f"Consider {option}: forecast respiratory activity along your commute is high and your shared indoor journey increases exposure."
            elif light == "yellow":
                component.assessment.recommendation = (
                    "Consider a well-fitting mask or quieter travel for your shared commute: the community forecast supports extra precautions."
                    if shared
                    else "Keep routine respiratory precautions during travel, especially if you enter shared indoor spaces: regional activity is elevated."
                )
            elif light == "green":
                component.assessment.recommendation = "No extra precaution is indicated for your planned commute; keep your usual respiratory precautions."
            else:
                component.assessment.recommendation = "There is not enough current forecast evidence for your commute on this date; follow routine respiratory precautions."
        # Unknown evidence for a required component prevents an overall reassurance.
        dominant = max(
            components,
            key=lambda c: (
                {"green": 0, "yellow": 1, "red": 2, "grey": 3}[
                    c.assessment.traffic_light
                ],
                precaution_score(c.assessment.model_dump()).value or 0,
            ),
        )
        return AssessmentResponse(
            activity_date=request.activity_date,
            home_location_id=request.home_location_id,
            destination_location_id=request.destination_location_id,
            assessment=dominant.assessment,
            dominant_component=dominant.component,
            components=components,
            forecasts=list(selected.values()),
            supplementary_signals=list(supplementary.values()),
            local_context=list(context.values()),
            disease_context=list(diseases.values()),
        ).model_dump(mode="json")

    def get_source_status(self):
        bundle, error = self._bundle()
        return {
            "schema_version": 1,
            "status": error or "available",
            "checked_at": self.now().isoformat(),
            "location_count": len(self._locations()),
            "forecasts": []
            if bundle is None
            else [
                {
                    "forecast_id": f.forecast_id,
                    "indicator": f.indicator,
                    "location_id": f.location_id,
                    "geographic_level": f.geographic_level,
                    "model": f.model,
                    "model_version": f.model_version,
                    "fetched_at": f.fetched_at.isoformat(),
                    "observed_through": f.observed_through.isoformat(),
                    "published_at": f.published_at.isoformat()
                    if f.published_at
                    else None,
                    "target_week_start": f.target_week_start.isoformat(),
                    "target_week_end": f.target_week_end.isoformat(),
                    "freshness": self._freshness(f),
                    "source_url": f.source_url,
                }
                for f in bundle.forecasts
            ],
            "inference_on_request": False,
        }


def create_app(service=None, access_token=None, include_mcp=True):
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, Header, HTTPException, Query

    service = service or ForecastService()
    access_token = access_token or os.getenv("INFECTIONPULSE_ACCESS_TOKEN")
    mcp = None
    if include_mcp:
        from infectionpulse.mcp_server import create_mcp

        mcp = create_mcp(service)
        mcp_http = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        if mcp:
            async with mcp.session_manager.run():
                yield
        else:
            yield

    async def authenticate(authorization: str | None = Header(default=None)):
        if access_token and (
            not authorization
            or not secrets.compare_digest(authorization, f"Bearer {access_token}")
        ):
            raise HTTPException(401, "Invalid service access token")

    app = FastAPI(title="InfectionPulse", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def secure_mcp(request, call_next):
        if request.url.path.startswith("/mcp") and access_token:
            from starlette.responses import JSONResponse

            auth = request.headers.get("authorization", "")
            if not secrets.compare_digest(auth, f"Bearer {access_token}"):
                return JSONResponse(
                    {"detail": "Invalid service access token"}, status_code=401
                )
        return await call_next(request)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get(
        "/v1/locations",
        response_model=LocationResolution,
        dependencies=[Depends(authenticate)],
    )
    def locations(q: str = Query(min_length=1, max_length=160)):
        return service.resolve_location(q)

    @app.get("/v1/forecasts/{location_id}", dependencies=[Depends(authenticate)])
    def forecast(location_id: str, target_week_start: date | None = None):
        return service.get_forecast(
            location_id, target_week_start.isoformat() if target_week_start else None
        )

    @app.post(
        "/v1/assessments",
        response_model=AssessmentResponse,
        dependencies=[Depends(authenticate)],
    )
    def assessment(request: AssessmentRequest):
        return service.assess_activity(request)

    @app.get("/v1/status", dependencies=[Depends(authenticate)])
    def status():
        return service.get_source_status()

    if mcp:
        app.mount("/mcp", mcp_http)
    return app
