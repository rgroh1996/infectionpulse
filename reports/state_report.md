# Experimental state forecast pilot

Run `23e8a1b6842ab06e`. Status: **complete**. Matched coverage: 370/384 planned state-weeks per model. Actual issue-time public releases; 12 monthly origins, 16 states, separately evaluated diseases. These results do not activate personal risk scores or validated probabilities.

| Disease / regime | Model | Rows | MAE | RMSE | Raw 80% coverage |
|---|---|---:|---:|---:|---:|
| influenza / all | LightGBM | 178 | 3.288 | 7.617 | 64.6% |
| influenza / all | LightGBM full-history | 178 | 3.267 | 7.619 | 68.5% |
| influenza / all | Persistence | 178 | 4.514 | 9.278 | — |
| influenza / all | Seasonal naive | 178 | 7.293 | 15.930 | — |
| influenza / all | TabPFN-3.5 | 178 | 3.509 | 7.758 | 62.9% |
| rsv / all | LightGBM | 192 | 0.967 | 2.263 | 66.1% |
| rsv / all | LightGBM full-history | 192 | 0.969 | 2.258 | 64.1% |
| rsv / all | Persistence | 192 | 1.158 | 2.432 | — |
| rsv / all | Seasonal naive | 192 | 0.545 | 1.264 | — |
| rsv / all | TabPFN-3.5 | 192 | 0.932 | 2.235 | 64.1% |
| rsv / pre_correction | LightGBM | 112 | 1.215 | 2.781 | 56.2% |
| rsv / pre_correction | LightGBM full-history | 112 | 1.215 | 2.781 | 56.2% |
| rsv / pre_correction | Persistence | 112 | 1.284 | 2.826 | — |
| rsv / pre_correction | Seasonal naive | 112 | 0.358 | 0.771 | — |
| rsv / pre_correction | TabPFN-3.5 | 112 | 1.183 | 2.761 | 56.2% |
| rsv / post_correction | LightGBM | 80 | 0.621 | 1.208 | 80.0% |
| rsv / post_correction | LightGBM full-history | 80 | 0.625 | 1.185 | 75.0% |
| rsv / post_correction | Persistence | 80 | 0.981 | 1.736 | — |
| rsv / post_correction | Seasonal naive | 80 | 0.807 | 1.733 | — |
| rsv / post_correction | TabPFN-3.5 | 80 | 0.580 | 1.145 | 75.0% |

Errors are notification cases per 100,000. Point-only baselines have no interval. See state_coverage.csv for all planned/abstained rows and state_per_origin.csv for seasonal variability. Twelve dates do not establish reliable statistical superiority.

RSV is a secondary stress test: releases before 23 February 2026 were affected by RKI's documented query error. Pre/post correction results must not be conflated.

- Issue-time training histories include revisions already present in that release, not first-report historical features.
- Mature labels are pinned to a later snapshot; this evaluates eventual notifications, not all infections.
- RSV releases omitted infections March 2025-February 2026; RKI corrected history on 23 February 2026.
- Raw predictive intervals have no coverage guarantee or public probability-readiness claim.
- Missing incidence is never converted to zero; seasonal-naive missing values use explicit persistence fallback.

Sources: [RKI influenza](https://github.com/robert-koch-institut/Influenzafaelle_in_Deutschland), [RKI RSV](https://github.com/robert-koch-institut/Respiratorische_Synzytialvirusfaelle_in_Deutschland). Full release checksums and fixed protocol are in the run manifest.
