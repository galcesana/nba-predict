# M1 Enriched Matchup Experiments

Generated at `2026-05-17T18:17:46.332995+00:00`.

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
| 1 | enriched_all | catboost | 0.6524 | 0.6154 | 0.2135 | 0.7166 | 0.0260 |
| 2 | enriched_no_confidence | lightgbm | 0.6513 | 0.6186 | 0.2148 | 0.7126 | 0.0309 |
| 3 | enriched_no_confidence | catboost | 0.6505 | 0.6193 | 0.2151 | 0.7128 | 0.0376 |
| 4 | enriched_lineup_only | catboost | 0.6520 | 0.6196 | 0.2152 | 0.7128 | 0.0374 |
| 5 | enriched_value_only | catboost | 0.6490 | 0.6203 | 0.2156 | 0.7099 | 0.0257 |
| 6 | enriched_all | lightgbm | 0.6551 | 0.6204 | 0.2157 | 0.7089 | 0.0221 |
| 7 | enriched_lineup_only | lightgbm | 0.6494 | 0.6207 | 0.2158 | 0.7086 | 0.0267 |
| 8 | enriched_value_only | lightgbm | 0.6555 | 0.6217 | 0.2162 | 0.7095 | 0.0251 |
| 9 | enriched_all | logistic_regression | 0.6471 | 0.6222 | 0.2156 | 0.7104 | 0.0253 |
| 10 | enriched_no_confidence | logistic_regression | 0.6471 | 0.6225 | 0.2158 | 0.7100 | 0.0233 |
| 11 | enriched_lineup_only | logistic_regression | 0.6475 | 0.6228 | 0.2157 | 0.7102 | 0.0325 |
| 12 | enriched_all | xgboost | 0.6467 | 0.6229 | 0.2167 | 0.7077 | 0.0263 |
| 13 | enriched_value_only | logistic_regression | 0.6490 | 0.6232 | 0.2161 | 0.7088 | 0.0194 |
| 14 | legacy | catboost | 0.6425 | 0.6236 | 0.2171 | 0.7047 | 0.0243 |
| 15 | enriched_all | catboost_platt | 0.6524 | 0.6246 | 0.2173 | 0.7166 | 0.0573 |
| 16 | enriched_no_confidence | xgboost | 0.6498 | 0.6247 | 0.2174 | 0.7066 | 0.0312 |
| 17 | enriched_availability_only | lightgbm | 0.6581 | 0.6249 | 0.2175 | 0.7099 | 0.0479 |
| 18 | enriched_value_only | xgboost | 0.6524 | 0.6249 | 0.2175 | 0.7076 | 0.0333 |
| 19 | enriched_availability_only | logistic_regression | 0.6475 | 0.6252 | 0.2169 | 0.7062 | 0.0211 |
| 20 | legacy | lightgbm | 0.6505 | 0.6254 | 0.2180 | 0.7026 | 0.0276 |
| 21 | enriched_lineup_only | xgboost | 0.6505 | 0.6255 | 0.2177 | 0.7095 | 0.0392 |
| 22 | enriched_availability_only | catboost | 0.6498 | 0.6270 | 0.2185 | 0.7001 | 0.0134 |
| 23 | enriched_no_confidence | catboost_platt | 0.6478 | 0.6276 | 0.2186 | 0.7128 | 0.0591 |
| 24 | legacy | logistic_regression | 0.6448 | 0.6277 | 0.2180 | 0.7027 | 0.0182 |
| 25 | enriched_no_confidence | lightgbm_platt | 0.6532 | 0.6280 | 0.2189 | 0.7126 | 0.0636 |
| 26 | enriched_value_only | catboost_platt | 0.6478 | 0.6281 | 0.2189 | 0.7099 | 0.0529 |
| 27 | enriched_availability_only | xgboost | 0.6494 | 0.6285 | 0.2194 | 0.6971 | 0.0176 |
| 28 | enriched_lineup_only | catboost_platt | 0.6494 | 0.6286 | 0.2191 | 0.7128 | 0.0621 |
| 29 | enriched_lineup_only | lightgbm_platt | 0.6498 | 0.6291 | 0.2194 | 0.7086 | 0.0495 |
| 30 | enriched_all | lightgbm_platt | 0.6501 | 0.6292 | 0.2195 | 0.7089 | 0.0549 |
| 31 | enriched_value_only | lightgbm_platt | 0.6574 | 0.6298 | 0.2198 | 0.7095 | 0.0598 |
| 32 | enriched_all | xgboost_platt | 0.6524 | 0.6311 | 0.2204 | 0.7077 | 0.0596 |
| 33 | legacy | xgboost | 0.6498 | 0.6314 | 0.2204 | 0.6959 | 0.0248 |
| 34 | legacy | catboost_platt | 0.6498 | 0.6314 | 0.2204 | 0.7047 | 0.0521 |
| 35 | enriched_availability_only | lightgbm_platt | 0.6494 | 0.6326 | 0.2210 | 0.7099 | 0.0673 |
| 36 | enriched_lineup_only | xgboost_platt | 0.6463 | 0.6332 | 0.2213 | 0.7095 | 0.0638 |
| 37 | legacy | lightgbm_platt | 0.6482 | 0.6334 | 0.2215 | 0.7026 | 0.0553 |
| 38 | enriched_value_only | xgboost_platt | 0.6490 | 0.6336 | 0.2215 | 0.7076 | 0.0563 |
| 39 | enriched_availability_only | catboost_platt | 0.6475 | 0.6337 | 0.2214 | 0.7001 | 0.0453 |
| 40 | enriched_no_confidence | xgboost_platt | 0.6467 | 0.6345 | 0.2219 | 0.7066 | 0.0637 |
| 41 | enriched_availability_only | xgboost_platt | 0.6391 | 0.6362 | 0.2227 | 0.6971 | 0.0505 |
| 42 | legacy | xgboost_platt | 0.6448 | 0.6389 | 0.2239 | 0.6959 | 0.0512 |
| 43 | m1_only | logistic_regression | 0.6280 | 0.6469 | 0.2277 | 0.6668 | 0.0212 |
| 44 | m1_only | catboost | 0.6162 | 0.6523 | 0.2302 | 0.6633 | 0.0416 |
| 45 | m1_only | lightgbm | 0.6101 | 0.6548 | 0.2313 | 0.6588 | 0.0329 |
| 46 | m1_only | xgboost | 0.6047 | 0.6585 | 0.2331 | 0.6617 | 0.0565 |
| 47 | m1_only | catboost_platt | 0.5902 | 0.6618 | 0.2347 | 0.6633 | 0.0674 |
| 48 | m1_only | lightgbm_platt | 0.5841 | 0.6641 | 0.2358 | 0.6588 | 0.0619 |
| 49 | m1_only | xgboost_platt | 0.5738 | 0.6672 | 0.2373 | 0.6617 | 0.0814 |

## Best Test Model

- Feature set: `enriched_all`
- Model: `catboost`
- Accuracy: `0.6524`
- Log loss: `0.6154`
- ECE: `0.0260`

## Best Model Slices

| Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| all_test | 2621 | 0.6524 | 0.6154 | 0.2135 | 0.7166 | 0.0260 |
Slice note: all_test — Entire held-out test set.
| playoffs | 166 | 0.6446 | 0.6443 | 0.2260 | 0.6573 | 0.0777 |
Slice note: playoffs — Games with NBA playoff-style game ids (`004...`).
| regular_season | 2455 | 0.6530 | 0.6134 | 0.2127 | 0.7200 | 0.0309 |
Slice note: regular_season — Games with non-playoff ids.
| high_context_confidence | 700 | 0.6371 | 0.6357 | 0.2223 | 0.6842 | 0.0166 |
Slice note: high_context_confidence — Top quartile of projected context confidence.
| low_context_confidence | 679 | 0.6480 | 0.6147 | 0.2136 | 0.7211 | 0.0421 |
Slice note: low_context_confidence — Bottom quartile of projected context confidence.

## Notes

- `legacy` keeps the original production matchup representation.
- `enriched_all` adds the new player-value, projected availability, and lineup-aware columns.
- The `enriched_*_only` variants test whether each M1 feature family adds incremental value on top of the legacy backbone.
- `m1_only` isolates only the new M1 columns to measure standalone signal.
- `*_platt` rows apply Platt scaling using validation predictions before scoring the test set.
