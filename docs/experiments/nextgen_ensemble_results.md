# Next-Generation Ensemble Results

Generated at `2026-05-17T14:11:41.876020+00:00`.

## Leaderboard

| Rank | Family | Label | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | nextgen_ensemble | nextgen_full / raw | 0.6550 | 0.6137 | 0.2126 | 0.7261 | 0.0476 |
| 2 | nextgen_ensemble | nextgen_lightgbm / raw | 0.6550 | 0.6137 | 0.2126 | 0.7263 | 0.0500 |
| 3 | nextgen_ensemble | nextgen_catboost / raw | 0.6587 | 0.6147 | 0.2131 | 0.7254 | 0.0528 |
| 4 | production_stack | production ensemble raw | 0.6562 | 0.6152 | 0.2133 | 0.7250 | 0.0533 |
| 5 | nextgen_ensemble | production_retrained / raw | 0.6562 | 0.6152 | 0.2133 | 0.7250 | 0.0533 |
| 6 | production_stack | production neural full fusion | 0.6485 | 0.6170 | 0.2142 | 0.7151 | 0.0264 |
| 7 | nextgen_ensemble | nextgen_lightgbm / calibrated | 0.6542 | 0.6195 | 0.2149 | 0.7263 | 0.0650 |
| 8 | nextgen_ensemble | nextgen_full / calibrated | 0.6534 | 0.6197 | 0.2150 | 0.7261 | 0.0661 |
| 9 | nextgen_ensemble | nextgen_catboost / calibrated | 0.6591 | 0.6206 | 0.2154 | 0.7254 | 0.0657 |
| 10 | production_stack | production ensemble calibrated | 0.6570 | 0.6207 | 0.2155 | 0.7250 | 0.0640 |
| 11 | nextgen_ensemble | production_retrained / calibrated | 0.6570 | 0.6207 | 0.2155 | 0.7250 | 0.0640 |

## Verdict

- Best overall: `nextgen_full / raw`
- Best next-gen entry: `nextgen_full / raw`
- Production baseline: `production ensemble raw`
- Next-gen minus production log-loss gap: `-0.0015`

## Next-Gen Full Weights

- `elo_prob`: `1.3337`
- `xgboost_prob`: `0.8877`
- `enriched_lightgbm_prob`: `0.8764`
- `neural_prob`: `0.7565`
- `enriched_catboost_prob`: `0.4052`

## Notes

- `production ensemble raw` is the current saved production meta-model output.
- `production_retrained` retrains the same logistic meta-model on the same validation split, using only `neural + xgboost + elo`.
- `nextgen_full` adds enriched CatBoost and LightGBM probabilities to the production input stack.
