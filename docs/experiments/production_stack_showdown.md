# Production Stack Showdown

Generated at `2026-05-17T11:03:38.331538+00:00`.

## Combined Leaderboard

| Rank | Family | Label | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | production_stack | production ensemble raw | 0.6562 | 0.6152 | 0.2133 | 0.7250 | 0.0533 |
| 2 | enriched_benchmark | enriched_all / catboost | 0.6513 | 0.6165 | 0.2138 | 0.7181 | 0.0417 |
| 3 | enriched_benchmark | enriched_no_confidence / catboost | 0.6542 | 0.6168 | 0.2140 | 0.7170 | 0.0361 |
| 4 | enriched_benchmark | enriched_all / lightgbm | 0.6554 | 0.6169 | 0.2140 | 0.7164 | 0.0192 |
| 5 | production_stack | production neural full fusion | 0.6485 | 0.6170 | 0.2142 | 0.7151 | 0.0264 |
| 6 | enriched_benchmark | enriched_availability_only / catboost | 0.6489 | 0.6176 | 0.2144 | 0.7152 | 0.0363 |
| 7 | enriched_benchmark | enriched_value_only / catboost | 0.6473 | 0.6186 | 0.2148 | 0.7146 | 0.0420 |
| 8 | enriched_benchmark | enriched_lineup_only / catboost | 0.6452 | 0.6187 | 0.2149 | 0.7141 | 0.0411 |
| 9 | enriched_benchmark | legacy / catboost | 0.6489 | 0.6200 | 0.2155 | 0.7125 | 0.0365 |
| 10 | enriched_benchmark | enriched_no_confidence / xgboost | 0.6521 | 0.6201 | 0.2156 | 0.7103 | 0.0254 |
| 11 | production_stack | production ensemble calibrated | 0.6570 | 0.6207 | 0.2155 | 0.7250 | 0.0640 |
| 12 | enriched_benchmark | enriched_all / xgboost | 0.6538 | 0.6210 | 0.2159 | 0.7099 | 0.0180 |
| 13 | enriched_benchmark | enriched_lineup_only / lightgbm | 0.6509 | 0.6212 | 0.2160 | 0.7088 | 0.0181 |
| 14 | enriched_benchmark | legacy / xgboost | 0.6452 | 0.6217 | 0.2162 | 0.7097 | 0.0313 |
| 15 | enriched_benchmark | enriched_lineup_only / logistic_regression | 0.6530 | 0.6218 | 0.2153 | 0.7120 | 0.0190 |
| 16 | enriched_benchmark | enriched_no_confidence / logistic_regression | 0.6505 | 0.6223 | 0.2156 | 0.7110 | 0.0190 |
| 17 | enriched_benchmark | enriched_value_only / logistic_regression | 0.6505 | 0.6225 | 0.2158 | 0.7106 | 0.0239 |
| 18 | enriched_benchmark | enriched_all / logistic_regression | 0.6501 | 0.6225 | 0.2157 | 0.7106 | 0.0178 |
| 19 | enriched_benchmark | enriched_value_only / xgboost | 0.6505 | 0.6226 | 0.2167 | 0.7088 | 0.0336 |
| 20 | enriched_benchmark | enriched_lineup_only / xgboost | 0.6395 | 0.6235 | 0.2171 | 0.7058 | 0.0303 |
| 21 | enriched_benchmark | enriched_value_only / lightgbm | 0.6477 | 0.6236 | 0.2172 | 0.7051 | 0.0119 |
| 22 | enriched_benchmark | enriched_availability_only / logistic_regression | 0.6489 | 0.6237 | 0.2162 | 0.7090 | 0.0270 |
| 23 | enriched_benchmark | enriched_availability_only / xgboost | 0.6530 | 0.6240 | 0.2172 | 0.7083 | 0.0301 |
| 24 | enriched_benchmark | enriched_no_confidence / lightgbm | 0.6570 | 0.6251 | 0.2175 | 0.7145 | 0.0424 |
| 25 | enriched_benchmark | enriched_no_confidence / catboost_platt | 0.6513 | 0.6254 | 0.2176 | 0.7170 | 0.0607 |
| 26 | enriched_benchmark | legacy / logistic_regression | 0.6424 | 0.6260 | 0.2172 | 0.7059 | 0.0253 |
| 27 | enriched_benchmark | enriched_all / lightgbm_platt | 0.6599 | 0.6262 | 0.2181 | 0.7164 | 0.0532 |
| 28 | enriched_benchmark | legacy / lightgbm | 0.6460 | 0.6266 | 0.2185 | 0.7033 | 0.0295 |
| 29 | enriched_benchmark | enriched_all / catboost_platt | 0.6509 | 0.6268 | 0.2182 | 0.7181 | 0.0646 |
| 30 | enriched_benchmark | enriched_availability_only / catboost_platt | 0.6468 | 0.6268 | 0.2183 | 0.7152 | 0.0653 |
| 31 | enriched_benchmark | enriched_availability_only / lightgbm | 0.6493 | 0.6268 | 0.2184 | 0.7053 | 0.0313 |
| 32 | enriched_benchmark | enriched_lineup_only / catboost_platt | 0.6464 | 0.6279 | 0.2188 | 0.7141 | 0.0628 |
| 33 | enriched_benchmark | enriched_value_only / catboost_platt | 0.6521 | 0.6280 | 0.2188 | 0.7146 | 0.0633 |
| 34 | enriched_benchmark | legacy / catboost_platt | 0.6493 | 0.6286 | 0.2191 | 0.7125 | 0.0642 |
| 35 | enriched_benchmark | enriched_lineup_only / lightgbm_platt | 0.6538 | 0.6292 | 0.2195 | 0.7088 | 0.0539 |
| 36 | enriched_benchmark | legacy / xgboost_platt | 0.6464 | 0.6316 | 0.2206 | 0.7097 | 0.0599 |
| 37 | enriched_benchmark | enriched_no_confidence / lightgbm_platt | 0.6570 | 0.6322 | 0.2209 | 0.7145 | 0.0667 |
| 38 | enriched_benchmark | enriched_all / xgboost_platt | 0.6424 | 0.6323 | 0.2209 | 0.7099 | 0.0631 |
| 39 | enriched_benchmark | enriched_no_confidence / xgboost_platt | 0.6570 | 0.6324 | 0.2210 | 0.7103 | 0.0620 |
| 40 | enriched_benchmark | enriched_value_only / lightgbm_platt | 0.6464 | 0.6332 | 0.2214 | 0.7051 | 0.0526 |
| 41 | enriched_benchmark | enriched_lineup_only / xgboost_platt | 0.6473 | 0.6334 | 0.2215 | 0.7058 | 0.0635 |
| 42 | enriched_benchmark | enriched_availability_only / xgboost_platt | 0.6424 | 0.6337 | 0.2216 | 0.7083 | 0.0632 |
| 43 | enriched_benchmark | enriched_availability_only / lightgbm_platt | 0.6428 | 0.6341 | 0.2218 | 0.7053 | 0.0645 |
| 44 | enriched_benchmark | enriched_value_only / xgboost_platt | 0.6375 | 0.6348 | 0.2221 | 0.7088 | 0.0733 |
| 45 | enriched_benchmark | legacy / lightgbm_platt | 0.6456 | 0.6357 | 0.2226 | 0.7033 | 0.0660 |
| 46 | enriched_benchmark | m1_only / logistic_regression | 0.6200 | 0.6494 | 0.2289 | 0.6627 | 0.0214 |
| 47 | enriched_benchmark | m1_only / catboost | 0.6240 | 0.6527 | 0.2304 | 0.6608 | 0.0344 |
| 48 | enriched_benchmark | m1_only / xgboost | 0.6094 | 0.6608 | 0.2341 | 0.6463 | 0.0395 |
| 49 | enriched_benchmark | m1_only / lightgbm | 0.6065 | 0.6613 | 0.2344 | 0.6465 | 0.0391 |
| 50 | enriched_benchmark | m1_only / catboost_platt | 0.5931 | 0.6628 | 0.2352 | 0.6608 | 0.0709 |
| 51 | enriched_benchmark | m1_only / xgboost_platt | 0.5796 | 0.6689 | 0.2381 | 0.6463 | 0.0609 |
| 52 | enriched_benchmark | m1_only / lightgbm_platt | 0.5743 | 0.6693 | 0.2383 | 0.6465 | 0.0652 |

## Verdict

- Best overall: `production ensemble raw`
- Best production stack entry: `production ensemble raw`
- Best enriched challenger: `enriched_all / catboost`
- Best calibrated entry: `enriched_value_only / lightgbm`
- Production minus enriched log-loss gap: `-0.0013`

## Production Slice Comparison

| Model | Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| production_neural_full_fusion | high_context_confidence | 703 | 0.6245 | 0.6335 | 0.2216 | 0.6923 | 0.0379 |
| production_neural_full_fusion | low_context_confidence | 660 | 0.6485 | 0.6168 | 0.2142 | 0.7143 | 0.0222 |
| production_ensemble_raw | high_context_confidence | 703 | 0.6373 | 0.6316 | 0.2209 | 0.6985 | 0.0446 |
| production_ensemble_raw | low_context_confidence | 660 | 0.6439 | 0.6180 | 0.2146 | 0.7199 | 0.0516 |
| production_ensemble_calibrated | high_context_confidence | 703 | 0.6344 | 0.6358 | 0.2226 | 0.6985 | 0.0561 |
| production_ensemble_calibrated | low_context_confidence | 660 | 0.6500 | 0.6232 | 0.2167 | 0.7199 | 0.0600 |

## Notes

- The enriched rows are loaded from the saved benchmark report rather than retrained here.
- The production rows are recomputed from saved neural and ensemble artifacts so they use the same held-out test split and slice masks.
- Lower log loss is treated as the primary rank metric because the product goal is calibrated probability quality, not just hard-pick accuracy.
