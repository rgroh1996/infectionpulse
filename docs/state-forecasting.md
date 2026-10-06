# State-level influenza and RSV pilot

This experiment tests a narrower, more local prediction task than the four-region ARE service: **next calendar week's reported influenza or RSV incidence in each of Germany's 16 states**. It does not predict all infections or individual infection odds. The results are separate from the precaution score until independently validated.

## Frozen design

`reports/state_protocol.json` is written before the first model comparison and refuses silent changes. Influenza is primary; RSV is a separately reported stress test. The 12 test origins are the first Monday of every month from August 2025 through July 2026. Four development dates precede the test. Twelve temporal origins remain a small seasonal pilot even though each date contains 16 correlated states.

For each target Monday, preparation fetches the last actual RKI data-file commit available by **Friday 10:00 Europe/Berlin**. It stores the raw bytes, SHA256, commit and timestamp. Current data never substitutes for an absent historical release. Every prediction uses the latest complete source week, typically two or three observation steps before the target. Longer reporting lags abstain.

Calendar joins create lags, seasonal features and the target; missing weeks stay missing. In particular, a missing summer influenza incidence is not treated as a reported zero. One-hot state features allow pooled models to learn from all states. Median imputation and missingness indicators are fitted only on training rows. A log-change target models growth relative to the last published incidence.

Training is reconstructed from the history contained in each actual issue-time snapshot. This includes corrections already available at that issue, rather than the features as they were first reported historically. Labels are restricted to target Sundays at least **35 days before issue**. This limitation is explicit and applies equally to all learned models.

The primary learned comparison uses up to 2,000 deterministically sampled training rows for TabPFN-3.5 and LightGBM; a separately tuned full-history LightGBM tests whether more historical context closes the gap. Tree tuning uses only the four development targets, labelled from a release available before the first test. Test labels come from a pinned September 2026 mature vintage, with its exact checksum in the manifest. They are not assumed permanently final.

## Reproduce

```sh
# Public data only; first preparation downloads historical releases.
.venv/bin/python scripts/pathogen_data.py refresh
.venv/bin/python scripts/state_benchmark.py prepare --download

# Local baselines; cached public releases suffice after preparation.
.venv/bin/python scripts/state_benchmark.py run --baselines-only

# Non-billing dimensions quote, then an explicit bounded hosted run.
.venv/bin/python scripts/state_benchmark.py quote
.venv/bin/python scripts/state_benchmark.py run --max-api-units 240000

# Rebuild tables from cached predictions without inference.
.venv/bin/python scripts/state_benchmark.py report
```

The quoted units are estimates, not a hard provider-side billing cap. Completed prediction jobs are reused. Source/code hashes identify each run; a code change requires new preparation rather than silently reusing incompatible predictions. Nothing modifies the completed ARE benchmark or its paid caches.

Reports: `reports/state_report.md`, `state_results.json`, `state_summary.csv`, `state_per_origin.csv` and `state_coverage.csv`. Scoring uses the same finite, labelled state-target pairs for all completed competing models. Coverage discloses every planned model/date, missing rows and unrun models. MAE and RMSE are in reported cases per 100,000. Raw 10th–90th percentile intervals are evaluated for empirical coverage; they have no calibration guarantee. Persistence and seasonal-naive are point-only baselines and therefore have no interval coverage claim. A missing seasonal baseline falls back explicitly to persistence.

## Current experimental forecasts

```sh
.venv/bin/python scripts/pathogen_data.py refresh
.venv/bin/python scripts/state_benchmark.py predict --estimate-only
# Substitute a budget after reviewing the current quote.
.venv/bin/python scripts/state_benchmark.py predict --max-api-units 20000
```

`predict` uses the actual current time, checks recent source availability and writes `models/state_pathogens/forecast_bundle.json`. Each record identifies its disease, state, target week, original inference time, observation cutoff and uncalibrated quantiles. A source check older than 72 hours or unsupported reporting lag produces an explicit abstention. The command never relabels an old forecast as newly inferred. These research artifacts are not silently inserted into the app's established ARE recommendation or morning brief.

## RSV data-system break

RKI documents that publications from March 2025 to February 2026 omitted some reported infections because of a query error; corrected historical data was published on 23 February 2026. Therefore RSV metrics are reported both overall and separately for issues before/after correction. This is a realistic source-reliability stress test, not a clean comparison of disease dynamics. No combined influenza+RSV winning headline should conceal this break.

Official sources: [RKI influenza documentation](https://github.com/robert-koch-institut/Influenzafaelle_in_Deutschland), [RKI RSV documentation](https://github.com/robert-koch-institut/Respiratorische_Synzytialvirusfaelle_in_Deutschland). Public data attribution: Robert Koch-Institut, CC-BY-4.0.
