# Next-Generation Ensemble Results

Generated at `2026-05-19T09:18:12.364667+00:00`.

## Leaderboard

| Rank | Family | Label | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | nextgen_ensemble | nextgen_full / raw | 0.6536 | 0.6159 | 0.2135 | 0.7225 | 0.0454 |
| 2 | nextgen_ensemble | nextgen_lightgbm / raw | 0.6505 | 0.6167 | 0.2139 | 0.7211 | 0.0463 |
| 3 | nextgen_ensemble | nextgen_catboost / raw | 0.6517 | 0.6169 | 0.2139 | 0.7218 | 0.0478 |
| 4 | production_stack | production ensemble raw | 0.6482 | 0.6196 | 0.2152 | 0.7177 | 0.0481 |
| 5 | nextgen_ensemble | production_retrained / raw | 0.6482 | 0.6196 | 0.2152 | 0.7177 | 0.0481 |
| 6 | nextgen_ensemble | nextgen_full / calibrated | 0.6536 | 0.6214 | 0.2157 | 0.7225 | 0.0598 |
| 7 | nextgen_ensemble | nextgen_lightgbm / calibrated | 0.6498 | 0.6219 | 0.2160 | 0.7211 | 0.0594 |
| 8 | nextgen_ensemble | nextgen_catboost / calibrated | 0.6509 | 0.6222 | 0.2161 | 0.7218 | 0.0593 |
| 9 | production_stack | production neural full fusion | 0.6482 | 0.6232 | 0.2169 | 0.7096 | 0.0417 |
| 10 | production_stack | production ensemble calibrated | 0.6482 | 0.6245 | 0.2172 | 0.7177 | 0.0604 |
| 11 | nextgen_ensemble | production_retrained / calibrated | 0.6482 | 0.6245 | 0.2172 | 0.7177 | 0.0604 |

## Verdict

- Best overall: `nextgen_full / raw`
- Best next-gen entry: `nextgen_full / raw`
- Production baseline: `production ensemble raw`
- Next-gen minus production log-loss gap: `-0.0037`

## Next-Gen Full Weights

- `elo_prob`: `1.4179`
- `enriched_lightgbm_prob`: `0.8855`
- `enriched_catboost_prob`: `0.8165`
- `xgboost_prob`: `0.5877`
- `neural_prob`: `0.5342`

## Enriched Input Feature Sets

- `catboost` uses `enriched_value_only` (131 features) -> `enriched_catboost_prob`
- `lightgbm` uses `enriched_all` (176 features) -> `enriched_lightgbm_prob`

## Notes

- `production ensemble raw` is the current saved production meta-model output.
- `production_retrained` retrains the same logistic meta-model on the same validation split, using only `neural + xgboost + elo`.
- `nextgen_full` adds enriched CatBoost and LightGBM probabilities to the production input stack.
