# Small training sets: forecast ranges and warnings

Locked ARE test (run `65cf524d3927a4b8`, 408 region-weeks), but every model is trained on only the last 104 rows (26 weeks × 4 regions). Exploratory: designed after the main test was inspected. *Calibrated* ranges add the project's conformal adjustment, fitted on 208 earlier calibration rows; *raw* ranges are the model's own 10–90% quantiles.

| Training rows | Model | MAE ↓ | Pinball ↓ | Raw range: inside / above | Calibrated: inside / above | Calibrated width | q90 warning: missed / false |
|---|---|---:|---:|---:|---:|---:|---:|
| 104 | TabPFN-3.5 | 1136 | 378 | 77% / 5% | 84% / 2% | 4,903 | 1 / 148 |
| 104 | LightGBM | 1128 | 410 | 53% / 16% | 77% / 8% | 3,766 | 13 / 102 |
| 104 | XGBoost | 1219 | 422 | 57% / 14% | 75% / 9% | 3,808 | 10 / 119 |
| 1,000 (main test) | TabPFN-3.5 | 810 | 261 | 87% / 4% | 87% / 4% | 3,094 | 1 / 104 |
| 1,000 (main test) | LightGBM | 899 | 285 | 79% / 8% | 81% / 7% | 2,949 | 2 / 96 |
| 1,000 (main test) | XGBoost | 965 | 304 | 82% / 7% | 83% / 7% | 3,321 | 0 / 105 |
| 1,000 (main test) | LightGBM full-history | 839 | 266 | 84% / 7% | 84% / 7% | 2,962 | 1 / 90 |

Paired differences in raw ranges, competitor minus TabPFN (four-week block bootstrap, 95% CI):

- 104 rows, LightGBM: above-range rate +11.3% [+7.1%, +16.9%]; MAE -8 [-166, +161]
- 104 rows, XGBoost: above-range rate +9.6% [+5.9%, +14.0%]; MAE +83 [-98, +245]
- 1,000 rows, LightGBM: above-range rate +4.4% [+0.7%, +9.1%]; MAE +89 [+3, +178]
- 1,000 rows, LightGBM full-history: above-range rate +3.4% [+0.0%, +8.1%]; MAE +29 [-54, +114]

## Reading

- TabPFN's raw ranges are usable without a calibration step; raw tree quantiles are overconfident with few rows.
- After calibration, LightGBM's ranges are about as honest (77% inside) with narrower width; TabPFN keeps fewer outcomes above its range but is wider.
- Warning on q90 ≥ high threshold trades misses for false alarms: TabPFN's wider ranges miss fewer high weeks and raise more false warnings. This does not show better warnings overall.
- Point accuracy (MAE) is level with LightGBM at 104 rows.

Reproduce: `python scripts/short_history_benchmark.py` (cached; new hosted calls need `--max-api-units`).
