# Next-Generation Ensemble Results

Generated at `2026-05-17T19:19:27.145539+00:00`.

## Leaderboard

| Rank | Family | Label | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | nextgen_ensemble | nextgen_full / raw | 0.6501 | 0.6161 | 0.2137 | 0.7212 | 0.0483 |
| 2 | nextgen_ensemble | nextgen_catboost / raw | 0.6520 | 0.6166 | 0.2139 | 0.7216 | 0.0509 |
| 3 | nextgen_ensemble | nextgen_lightgbm / raw | 0.6486 | 0.6174 | 0.2143 | 0.7194 | 0.0450 |
| 4 | production_stack | production ensemble raw | 0.6482 | 0.6196 | 0.2152 | 0.7177 | 0.0481 |
| 5 | nextgen_ensemble | production_retrained / raw | 0.6482 | 0.6196 | 0.2152 | 0.7177 | 0.0481 |
| 6 | nextgen_ensemble | nextgen_full / calibrated | 0.6524 | 0.6215 | 0.2159 | 0.7212 | 0.0642 |
| 7 | nextgen_ensemble | nextgen_catboost / calibrated | 0.6520 | 0.6219 | 0.2160 | 0.7216 | 0.0670 |
| 8 | nextgen_ensemble | nextgen_lightgbm / calibrated | 0.6490 | 0.6224 | 0.2163 | 0.7194 | 0.0625 |
| 9 | production_stack | production neural full fusion | 0.6482 | 0.6232 | 0.2169 | 0.7096 | 0.0417 |
| 10 | production_stack | production ensemble calibrated | 0.6482 | 0.6245 | 0.2172 | 0.7177 | 0.0604 |
| 11 | nextgen_ensemble | production_retrained / calibrated | 0.6482 | 0.6245 | 0.2172 | 0.7177 | 0.0604 |

## Verdict

- Best overall: `nextgen_full / raw`
- Best next-gen entry: `nextgen_full / raw`
- Production baseline: `production ensemble raw`
- Next-gen minus production log-loss gap: `-0.0035`

## Next-Gen Full Weights

- `elo_prob`: `1.3052`
- `enriched_catboost_prob`: `1.0806`
- `enriched_lightgbm_prob`: `1.0790`
- `neural_prob`: `0.4292`
- `xgboost_prob`: `0.2887`

## Notes

- `production ensemble raw` is the current saved production meta-model output.
- `production_retrained` retrains the same logistic meta-model on the same validation split, using only `neural + xgboost + elo`.
- `nextgen_full` adds enriched CatBoost and LightGBM probabilities to the production input stack.
