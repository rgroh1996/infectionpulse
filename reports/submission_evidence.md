# InfectionPulse: measured claims and operational evidence

Recorded 21 September 2026. These are separate experiments, not a combined leaderboard.

**Update, 5 October 2026:** the four ARE forecasts saved on 17 September have now been compared with preliminary reports for 21–27 September. MAE is **1,412.6**, versus **2,158.25** for the original last-observation persistence reference (34.6% reduction). Three of four outcomes fall inside the saved intervals; South was underestimated (5,787 predicted versus 9,035 reported). This is one correlated target week and the labels can still change. The 16 state influenza forecasts for 28 September–4 October remain pending publication. The original 35-day mature-label scorer is unchanged. [Detailed results, chart and exact source releases](live_forecast_check.md).

| Claim | Evidence | Limit |
|---|---|---|
| TabPFN adds predictive value to the main respiratory task | ARE MAE 809.98 vs equal-context LightGBM 899.09; 9.9% reduction | Advantage over full-history LightGBM is only 3.45%; its confidence interval includes zero |
| TabPFN can detect high-burden weeks | 167/175 detected, 8 missed, 35 false alerts; strongest LightGBM 161/175, 14 missed, 34 false alerts | Exploratory after original test inspection; recall advantage CI −1.0 to +9.0 percentage points |
| Finer disease-specific predictions are technically feasible | Actual historical RKI releases, 16 states, 12 dates each for flu and RSV, all five methods completed | 370/384 matched state-weeks; monthly pilot, not broad prospective validation |
| TabPFN is not universally best | Flu: TabPFN MAE 3.509 vs full-history LightGBM 3.267. RSV overall: TabPFN 0.932 vs seasonal naïve 0.545 | RSV's historical query error affects the full-period comparison |
| Corrected-regime RSV is worth further study | TabPFN MAE 0.580 vs LightGBM 0.621 on 80 state-weeks | Only five dates; selected subgroup was declared in the protocol, but remains descriptive |
| Fresh public checks feed a working morning brief | 21 September source refresh succeeded; cached real TabPFN forecast used for 22 September fictional commute/office profile | Deterministic local renderer; no Hermes LLM/server execution or message delivery yet |
| Current state inference works | 16 experimental influenza forecasts for 28 September–4 October, generated 21 September | Research artifact only; RSV explicitly abstained for unsupported reporting lag |
| Personal advice remains traceable | Every brief carries model forecast IDs, dates, quantiles, profile assumptions and policy explanation | The 0–100 index is heuristic; no personal infection probability or causal WFH benefit is measured |

## Exact experiments

- Main ARE run: `65cf524d3927a4b8`; [forecast benchmark](benchmark.md).
- Exploratory warning protocol and input hashes: [alert_protocol.json](alert_protocol.json); [warning report](alert_report.md).
- State run: `23e8a1b6842ab06e`; [state protocol](state_protocol.json), [report](state_report.md), [per-origin table](state_per_origin.csv), [coverage](state_coverage.csv).
- State forecast models, pinned source hashes, paired predictions and training contexts are under ignored `models/state_pathogens/` and `data/pathogens/`, reproducible with [these commands](../docs/state-forecasting.md).
- Hosted responses identify `/app/tabpfn_models/tabpfn-v3.5-20260909.safetensors`. The adapter uses explicit quantiles; no missing posterior borders are invented.

## Operational demonstration

`python scripts/morning_brief.py --refresh` checks current public releases, then produces text and JSON under ignored `.runtime/morning-brief/`. The real 21 September run used the forecast issued 17 September for 21–27 September; source observations were through 13 September. Eastern Germany median: **7,881 ARE illnesses per 100,000**, 10th–90th percentiles **5,964–9,887**. For the fictional crowded office/train profile, the policy recommended considering home office, precaution index **85/100**. Source checks were fresh; no new ARE inference was needed that Monday.

An actual separate influenza inference on 21 September produced 16 state forecasts for 28 September–4 October. Berlin's median is **0.227 reported influenza cases per 100,000**, with raw 10th–90th percentiles **0.115–0.490**. This is reported notification incidence, not all infections; values cannot be compared numerically with the broad ARE survey indicator. RSV had only reports through 6 September, requiring four observation steps to the target, so it abstained. Full state output: `models/state_pathogens/forecast_bundle.json`.

The state interval coverage fell below the nominal 80% (flu **62.9%**, RSV **64.1%** overall). These local forecasts therefore remain experimental and do not alter the app's main precaution policy. The established ARE probability-readiness gate also remains unchanged.

## Reproduce the local demo

```sh
.venv/bin/python scripts/morning_brief.py --refresh
.venv/bin/python scripts/morning_brief.py --replay
.venv/bin/python scripts/alert_benchmark.py
.venv/bin/python scripts/state_benchmark.py report
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

The complete test suite passed **210 tests**; there are two existing upstream deprecation warnings. Streamlit AppTest passed, and the updated app started on a local TCP port. Changed application files pass Ruff; a repository-wide lint scan still reports pre-existing issues in `download_datasets.py` and `data/mobility/ba_pendleratlas_ingest.py`.

No scheduler was installed and no server was accessed. [Cron and later Hermes setup](../docs/morning-brief.md) are ready for the separately planned server integration. No automatic messages were sent. Do not present a scripted walkthrough as a recorded Hermes demonstration.
