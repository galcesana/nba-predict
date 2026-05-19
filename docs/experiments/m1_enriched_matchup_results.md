# M1 Enriched Matchup Experiments

Generated at `2026-05-19T08:43:40.165649+00:00`.

## Overview

- `legacy`: `104` features
- `enriched_all`: `176` features
- `enriched_no_confidence`: `173` features
- `enriched_value_only`: `131` features
- `enriched_availability_only`: `116` features
- `enriched_lineup_only`: `125` features
- `m1_only`: `72` features

## Test Leaderboard

| Rank | Feature Set | Model | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | enriched_value_only | catboost | 0.6616 | 0.6163 | 0.2135 | 0.7169 | 0.0234 |
| 2 | enriched_no_confidence | catboost | 0.6528 | 0.6164 | 0.2138 | 0.7141 | 0.0263 |
| 3 | enriched_all | catboost | 0.6570 | 0.6168 | 0.2139 | 0.7157 | 0.0301 |
| 4 | enriched_all | lightgbm | 0.6551 | 0.6177 | 0.2143 | 0.7141 | 0.0258 |
| 5 | enriched_no_confidence | lightgbm | 0.6585 | 0.6178 | 0.2142 | 0.7168 | 0.0297 |
| 6 | enriched_no_confidence | logistic_regression | 0.6581 | 0.6184 | 0.2142 | 0.7142 | 0.0166 |
| 7 | enriched_all | logistic_regression | 0.6620 | 0.6187 | 0.2142 | 0.7142 | 0.0152 |
| 8 | enriched_lineup_only | catboost | 0.6517 | 0.6195 | 0.2152 | 0.7119 | 0.0325 |
| 9 | enriched_value_only | lightgbm | 0.6585 | 0.6201 | 0.2152 | 0.7159 | 0.0279 |
| 10 | enriched_value_only | logistic_regression | 0.6604 | 0.6202 | 0.2148 | 0.7127 | 0.0116 |
| 11 | enriched_all | xgboost | 0.6601 | 0.6210 | 0.2157 | 0.7096 | 0.0201 |
| 12 | enriched_value_only | xgboost | 0.6593 | 0.6218 | 0.2160 | 0.7135 | 0.0333 |
| 13 | enriched_lineup_only | lightgbm | 0.6475 | 0.6230 | 0.2168 | 0.7067 | 0.0250 |
| 14 | enriched_availability_only | catboost | 0.6559 | 0.6231 | 0.2167 | 0.7054 | 0.0155 |
| 15 | legacy | catboost | 0.6425 | 0.6236 | 0.2171 | 0.7047 | 0.0243 |
| 16 | enriched_lineup_only | logistic_regression | 0.6536 | 0.6237 | 0.2161 | 0.7094 | 0.0164 |
| 17 | enriched_availability_only | xgboost | 0.6513 | 0.6240 | 0.2172 | 0.7040 | 0.0161 |
| 18 | enriched_availability_only | lightgbm | 0.6436 | 0.6240 | 0.2172 | 0.7038 | 0.0226 |
| 19 | enriched_availability_only | logistic_regression | 0.6517 | 0.6245 | 0.2164 | 0.7078 | 0.0235 |
| 20 | legacy | lightgbm | 0.6505 | 0.6254 | 0.2180 | 0.7026 | 0.0276 |
| 21 | enriched_no_confidence | catboost_platt | 0.6555 | 0.6256 | 0.2177 | 0.7141 | 0.0455 |
| 22 | enriched_value_only | catboost_platt | 0.6608 | 0.6262 | 0.2178 | 0.7169 | 0.0573 |
| 23 | enriched_all | catboost_platt | 0.6562 | 0.6267 | 0.2181 | 0.7157 | 0.0574 |
| 24 | enriched_no_confidence | lightgbm_platt | 0.6616 | 0.6268 | 0.2182 | 0.7168 | 0.0547 |
| 25 | enriched_all | lightgbm_platt | 0.6581 | 0.6271 | 0.2183 | 0.7141 | 0.0499 |
| 26 | legacy | logistic_regression | 0.6448 | 0.6277 | 0.2180 | 0.7027 | 0.0182 |
| 27 | enriched_lineup_only | catboost_platt | 0.6532 | 0.6281 | 0.2189 | 0.7119 | 0.0594 |
| 28 | enriched_no_confidence | xgboost | 0.6570 | 0.6284 | 0.2188 | 0.7093 | 0.0464 |
| 29 | enriched_value_only | lightgbm_platt | 0.6616 | 0.6285 | 0.2190 | 0.7159 | 0.0583 |
| 30 | enriched_value_only | xgboost_platt | 0.6616 | 0.6298 | 0.2196 | 0.7135 | 0.0588 |
| 31 | enriched_all | xgboost_platt | 0.6559 | 0.6303 | 0.2198 | 0.7096 | 0.0512 |
| 32 | enriched_availability_only | catboost_platt | 0.6490 | 0.6309 | 0.2200 | 0.7054 | 0.0487 |
| 33 | legacy | xgboost | 0.6498 | 0.6314 | 0.2204 | 0.6959 | 0.0248 |
| 34 | legacy | catboost_platt | 0.6498 | 0.6314 | 0.2204 | 0.7047 | 0.0521 |
| 35 | enriched_lineup_only | xgboost | 0.6505 | 0.6318 | 0.2205 | 0.7047 | 0.0505 |
| 36 | enriched_availability_only | lightgbm_platt | 0.6444 | 0.6321 | 0.2207 | 0.7038 | 0.0504 |
| 37 | enriched_lineup_only | lightgbm_platt | 0.6490 | 0.6322 | 0.2209 | 0.7067 | 0.0632 |
| 38 | legacy | lightgbm_platt | 0.6482 | 0.6334 | 0.2215 | 0.7026 | 0.0553 |
| 39 | enriched_availability_only | xgboost_platt | 0.6425 | 0.6339 | 0.2216 | 0.7040 | 0.0557 |
| 40 | enriched_no_confidence | xgboost_platt | 0.6494 | 0.6356 | 0.2223 | 0.7093 | 0.0637 |
| 41 | m1_only | logistic_regression | 0.6436 | 0.6366 | 0.2228 | 0.6841 | 0.0235 |
| 42 | legacy | xgboost_platt | 0.6448 | 0.6389 | 0.2239 | 0.6959 | 0.0512 |
| 43 | enriched_lineup_only | xgboost_platt | 0.6375 | 0.6391 | 0.2240 | 0.7047 | 0.0666 |
| 44 | m1_only | catboost | 0.6402 | 0.6409 | 0.2246 | 0.6794 | 0.0227 |
| 45 | m1_only | xgboost | 0.6257 | 0.6476 | 0.2279 | 0.6687 | 0.0274 |
| 46 | m1_only | catboost_platt | 0.6276 | 0.6512 | 0.2296 | 0.6794 | 0.0592 |
| 47 | m1_only | lightgbm | 0.6307 | 0.6516 | 0.2297 | 0.6715 | 0.0476 |
| 48 | m1_only | xgboost_platt | 0.6097 | 0.6583 | 0.2330 | 0.6687 | 0.0619 |
| 49 | m1_only | lightgbm_platt | 0.5944 | 0.6632 | 0.2354 | 0.6715 | 0.0741 |

## Best Test Model

- Feature set: `enriched_value_only`
- Model: `catboost`
- Accuracy: `0.6616`
- Log loss: `0.6163`
- ECE: `0.0234`

## Best Model Slices

| Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| all_test | 2621 | 0.6616 | 0.6163 | 0.2135 | 0.7169 | 0.0234 |
Slice note: all_test — Entire held-out test set.
| playoffs | 166 | 0.6627 | 0.6437 | 0.2257 | 0.6555 | 0.0354 |
Slice note: playoffs — Games with NBA playoff-style game ids (`004...`).
| regular_season | 2455 | 0.6615 | 0.6145 | 0.2127 | 0.7205 | 0.0262 |
Slice note: regular_season — Games with non-playoff ids.
| high_missing_value | 656 | 0.6692 | 0.6104 | 0.2110 | 0.7250 | 0.0327 |
Slice note: high_missing_value — Top quartile of projected missing player value.
| low_missing_value | 656 | 0.6296 | 0.6400 | 0.2245 | 0.6826 | 0.0469 |
Slice note: low_missing_value — Bottom quartile of projected missing player value.
| high_context_confidence | 669 | 0.6607 | 0.6252 | 0.2174 | 0.7013 | 0.0221 |
Slice note: high_context_confidence — Top quartile of projected context confidence.
| low_context_confidence | 657 | 0.6438 | 0.6243 | 0.2174 | 0.7098 | 0.0435 |
Slice note: low_context_confidence — Bottom quartile of projected context confidence.

## Notes

- `legacy` keeps the original production matchup representation.
- `enriched_all` adds the new player-value, projected availability, and lineup-aware columns.
- The `enriched_*_only` variants test whether each M1 feature family adds incremental value on top of the legacy backbone.
- `m1_only` isolates only the new M1 columns to measure standalone signal.
- `*_platt` rows apply Platt scaling using validation predictions before scoring the test set.
