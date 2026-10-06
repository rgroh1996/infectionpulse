# Exploratory warning evaluation

Forecast accuracy on this test was inspected before this alert analysis. This protocol was frozen before computing alert comparisons, not before the original test. No confirmatory alert-superiority claim is permitted.

Primary alert: forecast median reaches the existing regional historical 80th-percentile burden threshold. This evaluates the app's community-burden trigger before personal activity rules; it does not validate personal risk or WFH benefit.

Matched coverage: 408/416 region-weeks. One alert means one region-week, not one person or one independent outbreak.

## High Burden

Alert: `q50 >= reference_p80`; observed event: `target_incidence >= reference_p80`.

| Model | Events detected | Missed | Alerts | False alerts | Precision | Recall | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN-3.5 | 167/175 | 8 | 202 | 35 | 82.7% | 95.4% | 15.0% |
| LightGBM full-history | 161/175 | 14 | 195 | 34 | 82.6% | 92.0% | 14.6% |
| LightGBM | 158/175 | 17 | 194 | 36 | 81.4% | 90.3% | 15.5% |
| XGBoost full-history | 153/175 | 22 | 191 | 38 | 80.1% | 87.4% | 16.3% |
| XGBoost | 150/175 | 25 | 183 | 33 | 82.0% | 85.7% | 14.2% |
| Persistence | 146/175 | 29 | 196 | 50 | 74.5% | 83.4% | 21.5% |
| Seasonal naive | 162/175 | 13 | 207 | 45 | 78.3% | 92.6% | 19.3% |
| Recent mean | 139/175 | 36 | 192 | 53 | 72.4% | 79.4% | 22.7% |
| Recent trend | 128/175 | 47 | 195 | 67 | 65.6% | 73.1% | 28.8% |

Precision = correct alerts / all alerts. Recall = alerted high-burden weeks / all high-burden weeks. False-positive rate = false alerts / weeks without the event; it differs from the share of alerts that were false.

Paired TabPFN advantage versus full-history LightGBM (percentage points; positive favors TabPFN):

- recall: +3.4; 95% block interval [-1.0, +9.0].
- precision: +0.1; 95% block interval [-2.6, +3.5].
- false_positive_rate: -0.4; 95% block interval [-3.5, +2.8].
- f1: +1.6; 95% block interval [-1.2, +4.8].

## Seasonal Event

Alert: `probability >= 0.5`; observed event: `target_incidence > seasonal_reference_p80`.

| Model | Events detected | Missed | Alerts | False alerts | Precision | Recall | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN-3.5 | 143/195 | 52 | 218 | 75 | 65.6% | 73.3% | 35.2% |
| LightGBM full-history | 139/195 | 56 | 210 | 71 | 66.2% | 71.3% | 33.3% |
| LightGBM | 143/195 | 52 | 231 | 88 | 61.9% | 73.3% | 41.3% |
| XGBoost full-history | 126/195 | 69 | 217 | 91 | 58.1% | 64.6% | 42.7% |
| XGBoost | 135/195 | 60 | 229 | 94 | 59.0% | 69.2% | 44.1% |
| Persistence | 121/195 | 74 | 207 | 86 | 58.5% | 62.1% | 40.4% |
| Seasonal naive | 143/195 | 52 | 232 | 89 | 61.6% | 73.3% | 41.8% |
| Recent mean | 103/195 | 92 | 208 | 105 | 49.5% | 52.8% | 49.3% |
| Recent trend | 131/195 | 64 | 210 | 79 | 62.4% | 67.2% | 37.1% |

Precision = correct alerts / all alerts. Recall = alerted high-burden weeks / all high-burden weeks. False-positive rate = false alerts / weeks without the event; it differs from the share of alerts that were false.

Paired TabPFN advantage versus full-history LightGBM (percentage points; positive favors TabPFN):

- recall: +2.1; 95% block interval [-7.2, +12.1].
- precision: -0.6; 95% block interval [-6.4, +4.8].
- false_positive_rate: -1.9; 95% block interval [-10.4, +5.1].
- f1: +0.6; 95% block interval [-5.3, +7.1].

## Limits

- Exploratory extension of an already-inspected retrospective test; not a preregistered alert trial.
- High burden is a historical surveillance percentile, not a clinically validated danger threshold.
- Four macroregions; repeated regional weeks are correlated. Intervals retain calendar blocks and regions together.
- No individual exposure outcomes, avoided infections, or causal benefit of working from home were measured.
- The seasonal probability diagnostic does not override the failed public probability-readiness gate.
- Unavailable forecasts abstain; performance metrics use common available rows and disclose reduced coverage.
- Historical source vintages were used for testing; early development/training features were reconstructed.

Reproduce offline: `.venv/bin/python scripts/alert_benchmark.py`. The frozen protocol hashes all source artifacts; changed inputs require a separate output directory.
