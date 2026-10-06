# InfectionPulse technical guide

**A next-week respiratory forecast that personal agents can turn into a practical precaution.**

InfectionPulse combines public German surveillance, TabPFN-3.5, a reproducible historical benchmark, and read-only REST/MCP tools. Ask about a planned office day, commute, or indoor gathering; the service returns a traffic light, one sentence, and dated evidence. It does not estimate your personal probability of infection.

## Problem statement

A case-count chart does not answer “What precautions make sense for my plans next week?” InfectionPulse separates three responsibilities:

1. TabPFN predicts community respiratory illness activity and uncertainty.
2. A versioned, transparent policy combines that evidence with an activity's setting, crowding, duration, ventilation, and commute.
3. A personal agent such as Hermes explains the result and can include it in its own scheduled morning brief.

The primary indicator is **GrippeWeb acute respiratory illness (ARE)** in four German macroregions, covering respiratory symptoms from multiple causes rather than COVID alone. **Reported district COVID incidence and state influenza/RSV incidence** provide separate observed context. These signals are not interchangeable, and neither establishes an individual's infection odds. Non-respiratory diseases are not covered.

## Try the recorded example without API credits

With Python 3.12 and a virtual environment:

```bash
python -m pip install -r requirements-service.txt
python scripts/demo.py --brief
```

This replays real saved TabPFN forecasts at their recorded historical time and explicitly says it is not current advice. Remove `--brief` for the full evidence. Generate fresh forecasts using the setup below.

For the complete local morning brief, run `python scripts/morning_brief.py --refresh` (current public checks; zero model-spending budget by default) or `python scripts/morning_brief.py --replay` (explicit historical demonstration). The brief includes the recommendation, precaution score, numerical evidence, source dates and structured JSON. See [morning-brief setup](morning-brief.md) for private activity profiles and a scheduler configuration to install later. The local renderer is deterministic; Hermes integration remains a later server-side step.

## Quick start

Python **3.12** is the tested runtime. Use the existing `.venv`, or create one:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
# On Apple Silicon, LightGBM also requires: brew install libomp
```

`requirements.txt` lists direct dependencies; `requirements-lock.txt` records the tested environment. The service-only environment can use `requirements-service.txt`.

The existing `.env` stores `TABPFN_TOKEN`. For a fresh clone, copy `.env.example` to `.env` and set the token locally. `.env` and other `.env.*` credential files are ignored; the blank `.env.example` is intentionally shareable. Do not pass the TabPFN token to an agent client.

```bash
# Public sources and immutable historical snapshots
python scripts/respiratory_data.py refresh
python scripts/respiratory_data.py covid
python scripts/pathogen_data.py refresh
python scripts/respiratory_data.py build

# Lock preparation and tune baselines on development data only
python scripts/respiratory_benchmark.py prepare

# Non-billing quote for evaluation, exploratory ladder, and current forecast
python scripts/respiratory_benchmark.py quote --include-development --include-live

# Explicit budgets prevent accidental unbounded inference. Check the quote first.
python scripts/respiratory_benchmark.py ladder --max-api-units 100000
python scripts/respiratory_benchmark.py run --max-api-units 160000
python scripts/respiratory_benchmark.py predict --max-api-units 20000
python scripts/report.py

# Read-only service and optional simple demonstration
python scripts/serve.py
streamlit run app.py
```

REST/OpenAPI: **http://127.0.0.1:8000/docs**. MCP Streamable HTTP: **http://127.0.0.1:8000/mcp/**. Streamlit: **http://localhost:8501**. The service and app read cached results and consume no inference credits.

For offline baseline evaluation, use `python scripts/respiratory_benchmark.py run --baselines-only`. Exit code 2 explicitly indicates that the full TabPFN benchmark is incomplete. This is not presented as a completed comparison. Cached requests and completed model blocks are resumed rather than silently repeated.

## Data engineering and forecast contract

The production job issues **Friday at 10:00 Europe/Berlin** and predicts the following complete Monday–Sunday week. Publication delays mean this is normally two observation steps ahead. Explicit dates and lead-time features prevent a misleading “one-week” shift. A manual bootstrap can issue at the actual current time; its timestamp is retained and it is not counted as canonical Friday evaluation evidence.

Every source snapshot has an immutable upstream identifier, checksum, observation period, availability proxy, and retrieval time. GitHub commit timestamps are a documented proxy for public availability. Later corrections cannot overwrite historical input snapshots.

The main ARE model has 14 features: latest incidence; lags 1–4 and 52; change from the previous observation; three-week mean; target-week sine/cosine; region code; latest respondent count; target-week-minus-52 incidence; and observation-to-target lead time. Its source is the all-age regional GrippeWeb series. Acceleration, weather and demographics are not inputs to this model. Models predict log incidence change relative to the latest observation, a fixed design carried forward from the earlier COVID experiment. Quantiles are transformed back to incidence units. Median imputation and missingness indicators are fitted on the training context only.

The four regions cover Germany and are queried together, rather than running a separate model for each district or user. The default live job combines four forecast rows with 208 historical calibration rows in one query batch (below the adapter's 256-row chunk limit). The hosted configuration uses eight ensemble members, so one batched request should not be described as a single neural-network forward pass. Saved forecasts are reused by the API; commute/workplace selections are evaluated by separate policy rules and are not sent to TabPFN.

Daily source checks can be scheduled independently of model inference. The default refresh command issues on Friday; `scripts/refresh.py --forecast-now --max-api-units BUDGET` can request a current forecast on other days. Identical source/target/context requests reuse saved inference, while newly eligible history can change the context. GrippeWeb observations are weekly, so running the job daily does not imply daily new regional measurements. Each forecast retains its actual observation and issue dates.

The source archive starts in September 2023. Earlier development/training rows are explicitly reconstructed from the earliest available vintage, with their actual provenance retained and a disclosed simulated availability rule for pre-archive purging. **They are retrospective, not historical as-published evidence.** Locked test features use actual available snapshots. Evaluation truth is fixed to the first release at least 35 days after target Sunday.

Current source audit: 153 snapshots; 104 planned evaluation weeks × four regions = 416 outcomes; 408 eligible forecasts and eight abstentions caused by two genuinely stale-source weeks. All 416 target outcomes have mature labels. Counts can change on a newly reconstructed run and are recorded in its report.

Broad ARE resolution is four regions, mapped from district AGS through RKI's official state mapping. Work and home in the same macroregion share one ARE forecast. COVID uses a separately dated district seven-day incidence observation, with a three-day reporting buffer. Berlin boroughs are population-aggregated. Synthetic mobility and estimated demographic values are excluded from the new ARE model.

Sources and attribution:

- [RKI GrippeWeb dataset, documentation and CC-BY-4.0 license](https://github.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts)
- [Official GrippeWeb geography mapping](https://github.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/blob/main/Kontextmaterialien/GrippeWeb_Zuordung_Regionen.tsv)
- [RKI reported COVID seven-day incidence](https://github.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland)
- [RKI laboratory-confirmed influenza reports](https://github.com/robert-koch-institut/Influenzafaelle_in_Deutschland)
- [RKI RSV reports](https://github.com/robert-koch-institut/Respiratorische_Synzytialvirusfaelle_in_Deutschland)

The influenza/RSV adapter retains two complete weeks for all 16 states, immutable source bytes, checksums and publication proxies. The first verified refresh contains 64 observations: influenza through September 13, 2026; RSV through September 6. Dates differ and remain visible. Missing state reports never become zero, and an older week is not silently substituted for an absent current report. Daily refresh failures retain the last cache; freshness is checked again when serving it.

## TabPFN-3.5 and benchmark findings

The hosted adapter explicitly requests **v3.5** and verifies returned model-version metadata. Public aggregate feature rows are sent to Prior Labs; personal activity selections are not. No local GPU is required.

The primary experiment uses N=1,000 from up to twelve preceding years, reserving 52 eligible weeks for calibration. Eight thirteen-week blocks hold model context fixed while weekly features update from available releases. Baselines include persistence, seasonal naïve, recent mean/trend, tuned LightGBM and XGBoost, and full-history GBDT references.

Locked targets cover **August 5, 2024–July 27, 2026**. Tuning uses predetermined development origins before the locked period. The exploratory ladder uses N=25, 50, 100, 500, 2000 where sufficient context exists; January 2023 has insufficient history for N=2000. Two origins and one seed make this an exploratory comparison, not robust proof of universal sample efficiency.

MAE is primary; RMSE, quantile loss, 80% interval coverage/width, Brier score, precision–recall and reliability are reported separately. Paired four-week moving-block bootstrap comparisons preserve time dependence and retain all regions together. A strong headline requires at least 5% improvement over the best baseline and a positive lower confidence bound; code reports whether that criterion passes.

**Measured output:** [benchmark report](../reports/benchmark.md), [comparison table](../reports/benchmark_summary.csv), and [machine-readable report](../reports/benchmark.json). These are generated from saved predictions; no superiority is assumed in advance. Raw per-model predictions, model wrappers, selection rows, configuration fingerprints, coverage and source-vintage IDs remain under `models/respiratory/` locally.

The completed locked evaluation has **408 predictions per model**. TabPFN's MAE is **809.98**, compared with **899.09** for equal-context LightGBM and **838.92** for full-history LightGBM: reductions of **9.9%** and **3.5%**, respectively. The paired 95% interval for its MAE advantage over full-history LightGBM is **−53.65 to 116.91** incidence points, so the predeclared strong-win criterion **does not pass**. Raw 10th–90th percentile coverage is **86.5%**, above its nominal 80%. The exploratory learning curves vary by origin and do not show consistent small-sample dominance.

The community-event probability readiness check also **does not pass**: reliability error is 0.0532 against a predeclared maximum of 0.05. The service therefore withholds numerical event probabilities and returns the forecast quantiles and transparent precaution categories. Four real forecasts have been recorded prospectively; their outcomes are still awaiting mature labels.

**Are the warnings useful?** The [warning evaluation](../reports/alert_report.md) measures the existing community high-burden trigger on the same 408/416 matched region-weeks. TabPFN detected **167/175** high-burden weeks, missed **8**, and produced **35** false alerts; full-history LightGBM detected **161/175**, missed **14**, and produced **34** false alerts. TabPFN recall is **95.4%** and precision **82.7%**. Its recall advantage is **3.4 percentage points**, with a paired 95% block interval of **−1.0 to +9.0**: promising, not conclusive superiority. These are region-weeks, not independent outbreaks or avoided infections. The original test accuracy was already inspected; an immutable alert protocol was frozen before this additional comparison, which remains explicitly exploratory. Reproduce without new inference: `python scripts/alert_benchmark.py`.

**More local forecasts:** `scripts/state_benchmark.py` evaluates influenza and RSV across 16 states using actual historical RKI releases, a fixed monthly test, purged training labels, and development-only LightGBM tuning. Influenza is primary; RSV is a separate stress test because RKI corrected a reporting-system error in February 2026. [Protocol and reproduction](state-forecasting.md), [state pilot results](../reports/state_report.md). Experimental state forecasts are separate artifacts and do not yet change the precaution score.

The completed state pilot scores **370/384 planned state-weeks per model**. It does **not** establish general TabPFN superiority: influenza MAE is **3.509** for TabPFN versus **3.267** for full-history LightGBM. On RSV, seasonal naïve is best overall (**0.545**, versus TabPFN **0.932**); after the documented correction, TabPFN's MAE is **0.580** versus **0.621** for the strongest learned baseline, but this is only five dates. TabPFN's raw 80% interval coverage is **62.9% for flu and 64.1% for RSV overall**, insufficient for presenting these intervals as calibrated. On 21 September, a genuine experimental TabPFN forecast was also generated for all 16 states' influenza reports for 28 September–4 October. RSV abstained because its observation-to-target lag exceeded the supported horizon. [Submission evidence and claim limits](../reports/submission_evidence.md).

The hosted v3.5 full-output response was tested and contained logits but no bin borders, so it cannot support an honest reconstructed posterior CDF. The production adapter instead requests an explicit 99-quantile grid. Its 10th/50th/90th percentiles provide the displayed forecast; interpolation estimates a separately labeled approximate probability of **activity above a frozen seasonal historical reference**. This is never a personal infection probability. The local full-support adapter remains tested but is not used to invent missing server information. Service percentages stay withheld unless held-out readiness checks and live output checks pass.

## Precaution policy

Version 1.2 uses regional historical P50/P80 thresholds for low/moderate/high burden. These are prototype historical comparisons, not clinical danger thresholds. Exposure categories use shared indoor time, duration, crowding, ventilation and commute. HVAC presence alone is not good ventilation.

| Community burden | Lower exposure | Moderate exposure | Higher exposure |
|---|---|---|---|
| Low | Green | Green | Yellow |
| Moderate | Green | Yellow | Yellow |
| High | Green | Yellow | Red |

An upper forecast interval crossing the high-burden threshold raises shared indoor/transit guidance to at least yellow. Green means no additional precaution suggested by this model, not guaranteed safety. Yellow suggests masking, ventilation, or an outdoor alternative. Red suggests home office or changing the activity where feasible. **Grey** means insufficient supported evidence, including unavailable dates, stale data, or missing artifacts. Destination and commute are assessed separately, with the strongest supported component supplying the final sentence. A high burden at home is not incorrectly assigned to an office in another region. District COVID context never changes the ARE rule numerically.

The qualitative precautions align with [cleaner-air guidance](https://www.cdc.gov/respiratory-viruses/prevention/air-quality.html) and [mask guidance](https://www.cdc.gov/respiratory-viruses/prevention/masks.html); the thresholds and matrix are not clinically validated.

## Agent interfaces and local deployment

The dashboard leads with **one traffic light, a 0–100 precaution score and a short numerical reason**. Below are a Germany overview and recent influenza, RSV and COVID reports. Activity details, sources, scoring rules and benchmarks are collapsed. An alternative is highlighted only when recomputing the policy produces a lower score.

**Score version 1.1** is a deliberately coarse ordinal presentation of the policy, not a calibrated probability, learned personal risk estimate, or percentage. Its points are:

| Community activity | Lower exposure | Moderate exposure | Higher exposure |
|---|---:|---:|---:|
| Low | 10 | 25 | 45 |
| Moderate | 25 | 50 | 60 |
| High | 30 | 65 | 85 |

An upper-interval precaution raises an otherwise green score to 40. Green is below 40, yellow 40–69 and red at least 70. Missing evidence produces **no score**, never zero. The strongest component supplies the overall result; ties in colour use the larger score. A zero-minute commute has no travel exposure. Pathogen reports do not get added to broad ARE counts or double-counted in this score.

The Germany map defaults to the **four-region TabPFN ARE forecast**, with selectable influenza and RSV reports for all 16 states. Each observed layer uses the state's own reported weekly incidence and displays its reporting period; it is independent of the planned activity date and is not a future prediction. Its continuous colour scale shows reported cases per 100,000, not precaution scores. Missing, stale or inconsistent-period reports remain grey. The ARE forecast layer uses the selected forecast week, with the same estimate shared by states in each macroregion. Three regions sharing a "high" category can therefore share a colour even when their numerical forecasts differ. The advisory score still uses this four-region forecast; state-specific prediction requires separate model evaluation. Local projected polygons need no map tiles or third-party map requests. Bundled geometry: Eurostat/GISCO NUTS 2024, with [attribution and source metadata](../examples/germany_states.sources.json). **© EuroGeographics for the administrative boundaries.**

These explanations are returned in the REST/MCP assessment's `explanation` field so Hermes can use the same evidence. They are deterministic policy traces, not SHAP values or LLM-generated feature attributions. [SHAP](https://shap.readthedocs.io/) explains contributions to model output; a separate offline attribution job would be needed to explain TabPFN's forecast inputs. No model-attribution or paid inference job runs when someone opens the dashboard.

Numerical event probabilities remain available only through the detailed evidence after readiness and output checks pass. They mean **regional weekly ARE exceeding its seasonal historical reference**, not personal infection probability. They are currently withheld. The annual high-activity threshold used for the traffic light and the seasonal probability reference remain distinct. No probability is reconstructed from incidence or withheld bounds; the main score does not bypass this gate.

Four tools share one implementation across REST, MCP stdio and Streamable HTTP:

| MCP tool | REST route |
|---|---|
| `resolve_location` | `GET /v1/locations?q=Berlin` |
| `get_forecast` | `GET /v1/forecasts/11000` |
| `assess_activity` | `POST /v1/assessments` |
| `get_source_status` | `GET /v1/status` |

See [agent setup and usage instructions](agent-setup.md). Actual MCP SDK sessions are tested locally. The user's existing server-side Hermes test is deferred by request; no Hermes/OpenClaw end-to-end LLM execution is claimed yet.

Local validation on September 18, 2026 passed against the saved real TabPFN-3.5 forecast: all four tools over MCP Streamable HTTP, matching REST assessments, and no network or inference in cached service reads. Recheck a running service with `python scripts/validate_local_service.py --url http://127.0.0.1:8000`; the dated result is saved locally to `reports/local-validation.json`.

```bash
python scripts/serve.py --transport stdio
# HTTP defaults to loopback. Remote binds require a separate service token.
# Export INFECTIONPULSE_ACCESS_TOKEN in the shell first.
python scripts/serve.py --host 0.0.0.0
```

For Docker, set a distinct `INFECTIONPULSE_ACCESS_TOKEN` in `.env`, start Docker, then run `docker compose up --build`. The container reads public artifacts through read-only mounts and does not receive TABPFN_TOKEN. For non-local domains set `INFECTIONPULSE_ALLOWED_HOSTS` and, where necessary, `INFECTIONPULSE_ALLOWED_ORIGINS`; do not disable MCP host protection. Use HTTPS or an SSH tunnel for remote access.

## Automated refresh

`python scripts/refresh.py` polls ARE, COVID, influenza and RSV daily. On Friday after 10:00 Berlin time it attempts the weekly forecast under the explicit `--max-api-units` ceiling; zero prevents spending. `--forecast-now` is a deliberate manual bootstrap. A file lock prevents overlapping refreshes, cached predictions prevent repeat billing, and `.runtime/refresh_status.json` records outcomes. Immutable prediction ledger records retain original issue and generation dates. `python scripts/score_live.py` scores them only after their labels mature; forecasts generated after the target week started are excluded from prospective evidence. The daily refresh runs this scorer automatically.

Example cron configuration on a host whose timezone is Europe/Berlin:

```cron
0 10 * * * cd /absolute/path/InfectionPulse && .venv/bin/python scripts/refresh.py --max-api-units 20000 >> .runtime/refresh.log 2>&1
```

Create `.runtime` first. No scheduler is installed on your machine automatically. A server in UTC should use a timezone-aware systemd timer or supported cron timezone configuration.

ARE release age must be at most ten German calendar days and observation age at most fourteen days; checks must remain recent. New releases do not rewrite past forecasts or invalidate an otherwise valid current-week forecast. Daily polling does not imply daily new respiratory observations. On source failure, the last valid cache is retained, then becomes grey as freshness expires.

## Validation and submission

```bash
python -m pytest -q
python scripts/respiratory_data.py audit
python scripts/respiratory_data.py build
python scripts/report.py
streamlit run app.py
```

Tests cover source schema/AGS mapping, vintage availability, mature labels, pre-archive disclosure, seasonal alignment, DST, posterior tails, temporal calibration separation, forecast caching, local REST/MCP transport/auth, activity rules, unsupported dates, stale-source handling, and Streamlit startup/reruns. `python scripts/demo.py` replays a saved real example at its explicitly recorded historical time without API credits; it does not claim to be current advice.

This is a hackathon prototype. Historical accuracy, operational reliability, and precaution-policy usefulness are separate claims. Prospective predictions are retained for later scoring; the time before the deadline cannot establish long-term prospective effectiveness.

An earlier district-level COVID prototype (weather and mobility features) was superseded by this ARE design and is not part of the repository; none of its measurements are reported here.
