# M1 Enriched Matchup Experiments

Generated at `2026-05-17T19:15:04.055355+00:00`.

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
| 1 | enriched_all | lightgbm | 0.6570 | 0.6155 | 0.2133 | 0.7184 | 0.0291 |
| 2 | enriched_no_confidence | catboost | 0.6608 | 0.6168 | 0.2138 | 0.7148 | 0.0177 |
| 3 | enriched_value_only | lightgbm | 0.6654 | 0.6187 | 0.2146 | 0.7144 | 0.0277 |
| 4 | enriched_no_confidence | lightgbm | 0.6616 | 0.6187 | 0.2148 | 0.7143 | 0.0268 |
| 5 | enriched_all | catboost | 0.6635 | 0.6190 | 0.2148 | 0.7118 | 0.0176 |
| 6 | enriched_value_only | catboost | 0.6616 | 0.6195 | 0.2151 | 0.7106 | 0.0123 |
| 7 | enriched_availability_only | catboost | 0.6555 | 0.6202 | 0.2155 | 0.7093 | 0.0153 |
| 8 | enriched_no_confidence | logistic_regression | 0.6570 | 0.6203 | 0.2149 | 0.7118 | 0.0122 |
| 9 | enriched_value_only | logistic_regression | 0.6601 | 0.6204 | 0.2148 | 0.7123 | 0.0131 |
| 10 | enriched_value_only | xgboost | 0.6547 | 0.6207 | 0.2156 | 0.7117 | 0.0247 |
| 11 | enriched_all | logistic_regression | 0.6547 | 0.6213 | 0.2153 | 0.7106 | 0.0184 |
| 12 | enriched_lineup_only | lightgbm | 0.6501 | 0.6226 | 0.2166 | 0.7075 | 0.0221 |
| 13 | enriched_lineup_only | logistic_regression | 0.6547 | 0.6228 | 0.2156 | 0.7108 | 0.0216 |
| 14 | enriched_availability_only | lightgbm | 0.6463 | 0.6230 | 0.2167 | 0.7065 | 0.0183 |
| 15 | legacy | catboost | 0.6425 | 0.6236 | 0.2171 | 0.7047 | 0.0243 |
| 16 | enriched_all | lightgbm_platt | 0.6620 | 0.6238 | 0.2169 | 0.7184 | 0.0539 |
| 17 | enriched_all | xgboost | 0.6570 | 0.6242 | 0.2171 | 0.7112 | 0.0371 |
| 18 | enriched_no_confidence | xgboost | 0.6562 | 0.6248 | 0.2173 | 0.7097 | 0.0345 |
| 19 | enriched_availability_only | logistic_regression | 0.6486 | 0.6250 | 0.2167 | 0.7071 | 0.0287 |
| 20 | enriched_no_confidence | catboost_platt | 0.6551 | 0.6253 | 0.2175 | 0.7148 | 0.0495 |
| 21 | legacy | lightgbm | 0.6505 | 0.6254 | 0.2180 | 0.7026 | 0.0276 |
| 22 | enriched_lineup_only | catboost | 0.6612 | 0.6266 | 0.2182 | 0.7013 | 0.0197 |
| 23 | enriched_value_only | catboost_platt | 0.6578 | 0.6266 | 0.2181 | 0.7106 | 0.0431 |
| 24 | enriched_all | catboost_platt | 0.6475 | 0.6267 | 0.2181 | 0.7118 | 0.0462 |
| 25 | enriched_lineup_only | xgboost | 0.6498 | 0.6268 | 0.2183 | 0.7061 | 0.0397 |
| 26 | enriched_value_only | lightgbm_platt | 0.6517 | 0.6269 | 0.2183 | 0.7144 | 0.0533 |
| 27 | enriched_no_confidence | lightgbm_platt | 0.6551 | 0.6276 | 0.2187 | 0.7143 | 0.0582 |
| 28 | legacy | logistic_regression | 0.6448 | 0.6277 | 0.2180 | 0.7027 | 0.0182 |
| 29 | enriched_availability_only | catboost_platt | 0.6517 | 0.6289 | 0.2192 | 0.7093 | 0.0503 |
| 30 | enriched_availability_only | xgboost | 0.6467 | 0.6291 | 0.2196 | 0.6959 | 0.0223 |
| 31 | enriched_availability_only | lightgbm_platt | 0.6490 | 0.6311 | 0.2203 | 0.7065 | 0.0470 |
| 32 | enriched_lineup_only | lightgbm_platt | 0.6478 | 0.6314 | 0.2205 | 0.7075 | 0.0582 |
| 33 | legacy | xgboost | 0.6498 | 0.6314 | 0.2204 | 0.6959 | 0.0248 |
| 34 | legacy | catboost_platt | 0.6498 | 0.6314 | 0.2204 | 0.7047 | 0.0521 |
| 35 | enriched_value_only | xgboost_platt | 0.6604 | 0.6315 | 0.2204 | 0.7117 | 0.0574 |
| 36 | enriched_no_confidence | xgboost_platt | 0.6551 | 0.6315 | 0.2204 | 0.7097 | 0.0595 |
| 37 | enriched_all | xgboost_platt | 0.6463 | 0.6318 | 0.2206 | 0.7112 | 0.0607 |
| 38 | enriched_lineup_only | catboost_platt | 0.6436 | 0.6318 | 0.2205 | 0.7013 | 0.0440 |
| 39 | legacy | lightgbm_platt | 0.6482 | 0.6334 | 0.2215 | 0.7026 | 0.0553 |
| 40 | enriched_lineup_only | xgboost_platt | 0.6448 | 0.6357 | 0.2224 | 0.7061 | 0.0626 |
| 41 | enriched_availability_only | xgboost_platt | 0.6387 | 0.6377 | 0.2234 | 0.6959 | 0.0563 |
| 42 | legacy | xgboost_platt | 0.6448 | 0.6389 | 0.2239 | 0.6959 | 0.0512 |
| 43 | m1_only | logistic_regression | 0.6272 | 0.6429 | 0.2259 | 0.6721 | 0.0194 |
| 44 | m1_only | catboost | 0.6349 | 0.6473 | 0.2277 | 0.6657 | 0.0298 |
| 45 | m1_only | xgboost | 0.6158 | 0.6544 | 0.2311 | 0.6575 | 0.0362 |
| 46 | m1_only | lightgbm | 0.6181 | 0.6565 | 0.2321 | 0.6559 | 0.0371 |
| 47 | m1_only | catboost_platt | 0.6101 | 0.6590 | 0.2333 | 0.6657 | 0.0553 |
| 48 | m1_only | xgboost_platt | 0.6009 | 0.6643 | 0.2359 | 0.6575 | 0.0631 |
| 49 | m1_only | lightgbm_platt | 0.5918 | 0.6665 | 0.2370 | 0.6559 | 0.0659 |

## Best Test Model

- Feature set: `enriched_all`
- Model: `lightgbm`
- Accuracy: `0.6570`
- Log loss: `0.6155`
- ECE: `0.0291`

## Best Model Slices

| Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| all_test | 2621 | 0.6570 | 0.6155 | 0.2133 | 0.7184 | 0.0291 |
Slice note: all_test — Entire held-out test set.
| playoffs | 166 | 0.6386 | 0.6411 | 0.2248 | 0.6612 | 0.0485 |
Slice note: playoffs — Games with NBA playoff-style game ids (`004...`).
| regular_season | 2455 | 0.6582 | 0.6137 | 0.2125 | 0.7215 | 0.0314 |
Slice note: regular_season — Games with non-playoff ids.
| high_missing_value | 656 | 0.6616 | 0.6105 | 0.2111 | 0.7270 | 0.0492 |
Slice note: high_missing_value — Top quartile of projected missing player value.
| low_missing_value | 656 | 0.6189 | 0.6394 | 0.2242 | 0.6806 | 0.0397 |
Slice note: low_missing_value — Bottom quartile of projected missing player value.
| high_context_confidence | 662 | 0.6601 | 0.6220 | 0.2161 | 0.7079 | 0.0322 |
Slice note: high_context_confidence — Top quartile of projected context confidence.
| low_context_confidence | 669 | 0.6323 | 0.6274 | 0.2191 | 0.6980 | 0.0413 |
Slice note: low_context_confidence — Bottom quartile of projected context confidence.

## Notes

- `legacy` keeps the original production matchup representation.
- `enriched_all` adds the new player-value, projected availability, and lineup-aware columns.
- The `enriched_*_only` variants test whether each M1 feature family adds incremental value on top of the legacy backbone.
- `m1_only` isolates only the new M1 columns to measure standalone signal.
- `*_platt` rows apply Platt scaling using validation predictions before scoring the test set.
