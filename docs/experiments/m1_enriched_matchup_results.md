# M1 Enriched Matchup Experiments

Generated at `2026-05-19T20:44:03.723659+00:00`.

## Overview

- `legacy`: `104` features
- `enriched_all`: `191` features
- `enriched_no_confidence`: `188` features
- `enriched_value_only`: `140` features
- `enriched_availability_only`: `122` features
- `enriched_lineup_only`: `125` features
- `m1_only`: `87` features

## Test Leaderboard

| Rank | Feature Set | Model | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | enriched_no_confidence | catboost | 0.6570 | 0.6172 | 0.2141 | 0.7131 | 0.0232 |
| 2 | enriched_all | lightgbm | 0.6559 | 0.6175 | 0.2142 | 0.7150 | 0.0256 |
| 3 | enriched_no_confidence | logistic_regression | 0.6616 | 0.6180 | 0.2139 | 0.7153 | 0.0137 |
| 4 | enriched_value_only | lightgbm | 0.6566 | 0.6183 | 0.2144 | 0.7162 | 0.0331 |
| 5 | enriched_all | logistic_regression | 0.6658 | 0.6183 | 0.2140 | 0.7153 | 0.0158 |
| 6 | enriched_value_only | catboost | 0.6601 | 0.6183 | 0.2146 | 0.7113 | 0.0151 |
| 7 | enriched_all | catboost | 0.6555 | 0.6185 | 0.2147 | 0.7113 | 0.0126 |
| 8 | enriched_no_confidence | lightgbm | 0.6616 | 0.6194 | 0.2149 | 0.7160 | 0.0309 |
| 9 | enriched_lineup_only | catboost | 0.6517 | 0.6195 | 0.2152 | 0.7119 | 0.0325 |
| 10 | enriched_value_only | logistic_regression | 0.6601 | 0.6201 | 0.2147 | 0.7132 | 0.0236 |
| 11 | enriched_availability_only | catboost | 0.6551 | 0.6202 | 0.2155 | 0.7087 | 0.0199 |
| 12 | enriched_availability_only | lightgbm | 0.6513 | 0.6220 | 0.2163 | 0.7078 | 0.0183 |
| 13 | enriched_value_only | xgboost | 0.6471 | 0.6222 | 0.2164 | 0.7076 | 0.0218 |
| 14 | enriched_no_confidence | xgboost | 0.6559 | 0.6226 | 0.2163 | 0.7122 | 0.0385 |
| 15 | enriched_all | xgboost | 0.6517 | 0.6228 | 0.2166 | 0.7066 | 0.0157 |
| 16 | enriched_lineup_only | lightgbm | 0.6475 | 0.6230 | 0.2168 | 0.7067 | 0.0250 |
| 17 | enriched_availability_only | logistic_regression | 0.6559 | 0.6235 | 0.2160 | 0.7090 | 0.0178 |
| 18 | legacy | catboost | 0.6425 | 0.6236 | 0.2171 | 0.7047 | 0.0243 |
| 19 | enriched_lineup_only | logistic_regression | 0.6536 | 0.6237 | 0.2161 | 0.7094 | 0.0164 |
| 20 | legacy | lightgbm | 0.6505 | 0.6254 | 0.2180 | 0.7026 | 0.0276 |
| 21 | enriched_no_confidence | catboost_platt | 0.6562 | 0.6256 | 0.2176 | 0.7131 | 0.0469 |
| 22 | enriched_all | lightgbm_platt | 0.6562 | 0.6262 | 0.2180 | 0.7150 | 0.0510 |
| 23 | enriched_all | catboost_platt | 0.6532 | 0.6270 | 0.2183 | 0.7113 | 0.0467 |
| 24 | enriched_value_only | catboost_platt | 0.6513 | 0.6272 | 0.2183 | 0.7113 | 0.0495 |
| 25 | legacy | logistic_regression | 0.6448 | 0.6277 | 0.2180 | 0.7027 | 0.0182 |
| 26 | enriched_availability_only | catboost_platt | 0.6452 | 0.6278 | 0.2186 | 0.7087 | 0.0493 |
| 27 | enriched_value_only | lightgbm_platt | 0.6597 | 0.6279 | 0.2187 | 0.7162 | 0.0586 |
| 28 | enriched_lineup_only | catboost_platt | 0.6532 | 0.6281 | 0.2189 | 0.7119 | 0.0594 |
| 29 | enriched_no_confidence | lightgbm_platt | 0.6551 | 0.6283 | 0.2189 | 0.7160 | 0.0543 |
| 30 | enriched_availability_only | xgboost | 0.6494 | 0.6292 | 0.2194 | 0.7027 | 0.0397 |
| 31 | enriched_no_confidence | xgboost_platt | 0.6532 | 0.6301 | 0.2197 | 0.7122 | 0.0615 |
| 32 | enriched_value_only | xgboost_platt | 0.6513 | 0.6303 | 0.2199 | 0.7076 | 0.0536 |
| 33 | enriched_availability_only | lightgbm_platt | 0.6482 | 0.6310 | 0.2203 | 0.7078 | 0.0542 |
| 34 | enriched_all | xgboost_platt | 0.6517 | 0.6314 | 0.2203 | 0.7066 | 0.0482 |
| 35 | legacy | xgboost | 0.6498 | 0.6314 | 0.2204 | 0.6959 | 0.0248 |
| 36 | legacy | catboost_platt | 0.6498 | 0.6314 | 0.2204 | 0.7047 | 0.0521 |
| 37 | enriched_lineup_only | xgboost | 0.6505 | 0.6318 | 0.2205 | 0.7047 | 0.0505 |
| 38 | enriched_lineup_only | lightgbm_platt | 0.6490 | 0.6322 | 0.2209 | 0.7067 | 0.0632 |
| 39 | legacy | lightgbm_platt | 0.6482 | 0.6334 | 0.2215 | 0.7026 | 0.0553 |
| 40 | m1_only | logistic_regression | 0.6467 | 0.6347 | 0.2219 | 0.6874 | 0.0236 |
| 41 | enriched_availability_only | xgboost_platt | 0.6429 | 0.6376 | 0.2233 | 0.7027 | 0.0606 |
| 42 | legacy | xgboost_platt | 0.6448 | 0.6389 | 0.2239 | 0.6959 | 0.0512 |
| 43 | enriched_lineup_only | xgboost_platt | 0.6375 | 0.6391 | 0.2240 | 0.7047 | 0.0666 |
| 44 | m1_only | catboost | 0.6368 | 0.6453 | 0.2267 | 0.6748 | 0.0269 |
| 45 | m1_only | lightgbm | 0.6318 | 0.6518 | 0.2298 | 0.6692 | 0.0416 |
| 46 | m1_only | xgboost | 0.6177 | 0.6565 | 0.2321 | 0.6653 | 0.0502 |
| 47 | m1_only | catboost_platt | 0.6101 | 0.6590 | 0.2333 | 0.6748 | 0.0630 |
| 48 | m1_only | lightgbm_platt | 0.5982 | 0.6634 | 0.2354 | 0.6692 | 0.0691 |
| 49 | m1_only | xgboost_platt | 0.5841 | 0.6658 | 0.2366 | 0.6653 | 0.0742 |

## Best Test Model

- Feature set: `enriched_no_confidence`
- Model: `catboost`
- Accuracy: `0.6570`
- Log loss: `0.6172`
- ECE: `0.0232`

## Best Model Slices

| Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| all_test | 2621 | 0.6570 | 0.6172 | 0.2141 | 0.7131 | 0.0232 |
Slice note: all_test — Entire held-out test set.
| playoffs | 166 | 0.6627 | 0.6365 | 0.2227 | 0.6661 | 0.0408 |
Slice note: playoffs — Games with NBA playoff-style game ids (`004...`).
| regular_season | 2455 | 0.6566 | 0.6159 | 0.2135 | 0.7156 | 0.0253 |
Slice note: regular_season — Games with non-playoff ids.
| high_missing_value | 656 | 0.6631 | 0.6118 | 0.2118 | 0.7214 | 0.0361 |
Slice note: high_missing_value — Top quartile of projected missing player value.
| low_missing_value | 656 | 0.6296 | 0.6398 | 0.2245 | 0.6808 | 0.0482 |
Slice note: low_missing_value — Bottom quartile of projected missing player value.
| high_context_confidence | 669 | 0.6577 | 0.6246 | 0.2170 | 0.7011 | 0.0243 |
Slice note: high_context_confidence — Top quartile of projected context confidence.
| low_context_confidence | 657 | 0.6377 | 0.6235 | 0.2173 | 0.7079 | 0.0503 |
Slice note: low_context_confidence — Bottom quartile of projected context confidence.

## Notes

- `legacy` keeps the original production matchup representation.
- `enriched_all` adds the new player-value, projected availability, and lineup-aware columns.
- The `enriched_*_only` variants test whether each M1 feature family adds incremental value on top of the legacy backbone.
- `m1_only` isolates only the new M1 columns to measure standalone signal.
- `*_platt` rows apply Platt scaling using validation predictions before scoring the test set.
