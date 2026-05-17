# M1 Enriched Matchup Experiments

Generated at `2026-05-17T10:31:31.585183+00:00`.

## Overview

- `legacy`: `104` features
- `enriched_all`: `161` features
- `enriched_no_confidence`: `158` features
- `enriched_value_only`: `128` features
- `enriched_availability_only`: `116` features
- `enriched_lineup_only`: `125` features
- `m1_only`: `57` features

## Test Leaderboard

| Rank | Feature Set | Model | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | enriched_all | catboost | 0.6513 | 0.6165 | 0.2138 | 0.7181 | 0.0417 |
| 2 | enriched_no_confidence | catboost | 0.6542 | 0.6168 | 0.2140 | 0.7170 | 0.0361 |
| 3 | enriched_all | lightgbm | 0.6554 | 0.6169 | 0.2140 | 0.7164 | 0.0192 |
| 4 | enriched_availability_only | catboost | 0.6489 | 0.6176 | 0.2144 | 0.7152 | 0.0363 |
| 5 | enriched_value_only | catboost | 0.6473 | 0.6186 | 0.2148 | 0.7146 | 0.0420 |
| 6 | enriched_lineup_only | catboost | 0.6452 | 0.6187 | 0.2149 | 0.7141 | 0.0411 |
| 7 | legacy | catboost | 0.6489 | 0.6200 | 0.2155 | 0.7125 | 0.0365 |
| 8 | enriched_no_confidence | xgboost | 0.6521 | 0.6201 | 0.2156 | 0.7103 | 0.0254 |
| 9 | enriched_all | xgboost | 0.6538 | 0.6210 | 0.2159 | 0.7099 | 0.0180 |
| 10 | enriched_lineup_only | lightgbm | 0.6509 | 0.6212 | 0.2160 | 0.7088 | 0.0181 |
| 11 | legacy | xgboost | 0.6452 | 0.6217 | 0.2162 | 0.7097 | 0.0313 |
| 12 | enriched_lineup_only | logistic_regression | 0.6530 | 0.6218 | 0.2153 | 0.7120 | 0.0190 |
| 13 | enriched_no_confidence | logistic_regression | 0.6505 | 0.6223 | 0.2156 | 0.7110 | 0.0190 |
| 14 | enriched_value_only | logistic_regression | 0.6505 | 0.6225 | 0.2158 | 0.7106 | 0.0239 |
| 15 | enriched_all | logistic_regression | 0.6501 | 0.6225 | 0.2157 | 0.7106 | 0.0178 |
| 16 | enriched_value_only | xgboost | 0.6505 | 0.6226 | 0.2167 | 0.7088 | 0.0336 |
| 17 | enriched_lineup_only | xgboost | 0.6395 | 0.6235 | 0.2171 | 0.7058 | 0.0303 |
| 18 | enriched_value_only | lightgbm | 0.6477 | 0.6236 | 0.2172 | 0.7051 | 0.0119 |
| 19 | enriched_availability_only | logistic_regression | 0.6489 | 0.6237 | 0.2162 | 0.7090 | 0.0270 |
| 20 | enriched_availability_only | xgboost | 0.6530 | 0.6240 | 0.2172 | 0.7083 | 0.0301 |
| 21 | enriched_no_confidence | lightgbm | 0.6570 | 0.6251 | 0.2175 | 0.7145 | 0.0424 |
| 22 | enriched_no_confidence | catboost_platt | 0.6513 | 0.6254 | 0.2176 | 0.7170 | 0.0607 |
| 23 | legacy | logistic_regression | 0.6424 | 0.6260 | 0.2172 | 0.7059 | 0.0253 |
| 24 | enriched_all | lightgbm_platt | 0.6599 | 0.6262 | 0.2181 | 0.7164 | 0.0532 |
| 25 | legacy | lightgbm | 0.6460 | 0.6266 | 0.2185 | 0.7033 | 0.0295 |
| 26 | enriched_all | catboost_platt | 0.6509 | 0.6268 | 0.2182 | 0.7181 | 0.0646 |
| 27 | enriched_availability_only | catboost_platt | 0.6468 | 0.6268 | 0.2183 | 0.7152 | 0.0653 |
| 28 | enriched_availability_only | lightgbm | 0.6493 | 0.6268 | 0.2184 | 0.7053 | 0.0313 |
| 29 | enriched_lineup_only | catboost_platt | 0.6464 | 0.6279 | 0.2188 | 0.7141 | 0.0628 |
| 30 | enriched_value_only | catboost_platt | 0.6521 | 0.6280 | 0.2188 | 0.7146 | 0.0633 |
| 31 | legacy | catboost_platt | 0.6493 | 0.6286 | 0.2191 | 0.7125 | 0.0642 |
| 32 | enriched_lineup_only | lightgbm_platt | 0.6538 | 0.6292 | 0.2195 | 0.7088 | 0.0539 |
| 33 | legacy | xgboost_platt | 0.6464 | 0.6316 | 0.2206 | 0.7097 | 0.0599 |
| 34 | enriched_no_confidence | lightgbm_platt | 0.6570 | 0.6322 | 0.2209 | 0.7145 | 0.0667 |
| 35 | enriched_all | xgboost_platt | 0.6424 | 0.6323 | 0.2209 | 0.7099 | 0.0631 |
| 36 | enriched_no_confidence | xgboost_platt | 0.6570 | 0.6324 | 0.2210 | 0.7103 | 0.0620 |
| 37 | enriched_value_only | lightgbm_platt | 0.6464 | 0.6332 | 0.2214 | 0.7051 | 0.0526 |
| 38 | enriched_lineup_only | xgboost_platt | 0.6473 | 0.6334 | 0.2215 | 0.7058 | 0.0635 |
| 39 | enriched_availability_only | xgboost_platt | 0.6424 | 0.6337 | 0.2216 | 0.7083 | 0.0632 |
| 40 | enriched_availability_only | lightgbm_platt | 0.6428 | 0.6341 | 0.2218 | 0.7053 | 0.0645 |
| 41 | enriched_value_only | xgboost_platt | 0.6375 | 0.6348 | 0.2221 | 0.7088 | 0.0733 |
| 42 | legacy | lightgbm_platt | 0.6456 | 0.6357 | 0.2226 | 0.7033 | 0.0660 |
| 43 | m1_only | logistic_regression | 0.6200 | 0.6494 | 0.2289 | 0.6627 | 0.0214 |
| 44 | m1_only | catboost | 0.6240 | 0.6527 | 0.2304 | 0.6608 | 0.0344 |
| 45 | m1_only | xgboost | 0.6094 | 0.6608 | 0.2341 | 0.6463 | 0.0395 |
| 46 | m1_only | lightgbm | 0.6065 | 0.6613 | 0.2344 | 0.6465 | 0.0391 |
| 47 | m1_only | catboost_platt | 0.5931 | 0.6628 | 0.2352 | 0.6608 | 0.0709 |
| 48 | m1_only | xgboost_platt | 0.5796 | 0.6689 | 0.2381 | 0.6463 | 0.0609 |
| 49 | m1_only | lightgbm_platt | 0.5743 | 0.6693 | 0.2383 | 0.6465 | 0.0652 |

## Best Test Model

- Feature set: `enriched_all`
- Model: `catboost`
- Accuracy: `0.6513`
- Log loss: `0.6165`
- ECE: `0.0417`

## Best Model Slices

| Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| all_test | 2455 | 0.6513 | 0.6165 | 0.2138 | 0.7181 | 0.0417 |
Slice note: all_test — Entire held-out test set.
| regular_season | 2455 | 0.6513 | 0.6165 | 0.2138 | 0.7181 | 0.0417 |
Slice note: regular_season — Games with non-playoff ids.
| low_missing_value | 2455 | 0.6513 | 0.6165 | 0.2138 | 0.7181 | 0.0417 |
Slice note: low_missing_value — Bottom quartile of projected missing player value.
| high_context_confidence | 703 | 0.6302 | 0.6365 | 0.2229 | 0.6872 | 0.0286 |
Slice note: high_context_confidence — Top quartile of projected context confidence.
| low_context_confidence | 660 | 0.6439 | 0.6183 | 0.2148 | 0.7158 | 0.0550 |
Slice note: low_context_confidence — Bottom quartile of projected context confidence.

## Notes

- `legacy` keeps the original production matchup representation.
- `enriched_all` adds the new player-value, projected availability, and lineup-aware columns.
- The `enriched_*_only` variants test whether each M1 feature family adds incremental value on top of the legacy backbone.
- `m1_only` isolates only the new M1 columns to measure standalone signal.
- `*_platt` rows apply Platt scaling using validation predictions before scoring the test set.
