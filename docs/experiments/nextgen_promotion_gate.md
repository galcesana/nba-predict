# Next-Gen Promotion Gate

Generated at `2026-05-18T20:46:56.663542+00:00`.

## Promotion Status

- Status: `ready`
- Recommendation: Candidate is ready for a shadow/live promotion review.

## Gate Checks

| Check | Status | Detail |
|---|---|---|
| overall_log_loss_gain | pass | Next-gen must beat production by at least 0.001 log loss; observed -0.0035. |
| playoff_coverage | pass | Need at least 100 held-out playoff games; found 166. |
| missing_player_coverage | pass | Need at least 100 held-out games with nonzero missing-player value; found 2595. |
| critical_slice_regression | pass | No critical slice exceeded the allowed regression threshold. |

## Slice Comparison

| Slice | Games | Production Log Loss | Next-Gen Log Loss | Delta | Production Acc | Next-Gen Acc | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| all_test | 2621 | 0.6196 | 0.6161 | -0.0035 | 0.6482 | 0.6501 | scored |
| regular_season | 2455 | 0.6177 | 0.6144 | -0.0033 | 0.6493 | 0.6525 | scored |
| playoffs | 166 | 0.6477 | 0.6425 | -0.0053 | 0.6325 | 0.6145 | scored |
| season_2023_24 | 1312 | 0.6198 | 0.6177 | -0.0021 | 0.6471 | 0.6456 | scored |
| season_2024_25 | 1309 | 0.6194 | 0.6146 | -0.0048 | 0.6494 | 0.6547 | scored |
| missing_player_impact | 2595 | 0.6198 | 0.6164 | -0.0034 | 0.6486 | 0.6501 | scored |
| high_missing_value | 656 | 0.6129 | 0.6111 | -0.0019 | 0.6738 | 0.6570 | scored |
| high_context_confidence | 662 | 0.6293 | 0.6268 | -0.0026 | 0.6299 | 0.6375 | scored |
| low_context_confidence | 669 | 0.6224 | 0.6216 | -0.0008 | 0.6457 | 0.6368 | scored |
| home_back_to_back | 424 | 0.6341 | 0.6294 | -0.0047 | 0.6061 | 0.6297 | scored |
| away_back_to_back | 448 | 0.5977 | 0.5982 | 0.0005 | 0.6741 | 0.6719 | scored |
| dense_schedule | 1108 | 0.6158 | 0.6141 | -0.0017 | 0.6498 | 0.6516 | scored |

## What This Means

- The next-gen candidate clears the current promotion gates.
- Playoff coverage is now present with `166` held-out games.
- Missing-player coverage is present with `2595` held-out games.
- The next production step is a shadow/live promotion review with monitoring on playoff, calibration, and missing-player slices.
- Longer-term availability work should replace the historical absence proxy with richer official inactive history when available.
