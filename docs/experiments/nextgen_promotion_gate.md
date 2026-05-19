# Next-Gen Promotion Gate

Generated at `2026-05-19T09:20:45.427050+00:00`.

## Promotion Status

- Status: `ready`
- Recommendation: Candidate is ready for a shadow/live promotion review.

## Gate Checks

| Check | Status | Detail |
|---|---|---|
| overall_log_loss_gain | pass | Next-gen must beat production by at least 0.001 log loss; observed -0.0037. |
| playoff_coverage | pass | Need at least 100 held-out playoff games; found 166. |
| missing_player_coverage | pass | Need at least 100 held-out games with nonzero missing-player value; found 2602. |
| critical_slice_regression | pass | No critical slice exceeded the allowed regression threshold. |

## Slice Comparison

| Slice | Games | Production Log Loss | Next-Gen Log Loss | Delta | Production Acc | Next-Gen Acc | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| all_test | 2621 | 0.6196 | 0.6159 | -0.0037 | 0.6482 | 0.6536 | scored |
| regular_season | 2455 | 0.6177 | 0.6141 | -0.0035 | 0.6493 | 0.6554 | scored |
| playoffs | 166 | 0.6477 | 0.6417 | -0.0060 | 0.6325 | 0.6265 | scored |
| season_2023_24 | 1312 | 0.6198 | 0.6167 | -0.0031 | 0.6471 | 0.6532 | scored |
| season_2024_25 | 1309 | 0.6194 | 0.6151 | -0.0043 | 0.6494 | 0.6539 | scored |
| missing_player_impact | 2602 | 0.6194 | 0.6158 | -0.0036 | 0.6495 | 0.6537 | scored |
| high_missing_value | 656 | 0.6134 | 0.6080 | -0.0054 | 0.6723 | 0.6616 | scored |
| high_context_confidence | 669 | 0.6284 | 0.6251 | -0.0033 | 0.6248 | 0.6428 | scored |
| low_context_confidence | 657 | 0.6209 | 0.6194 | -0.0015 | 0.6484 | 0.6408 | scored |
| home_back_to_back | 424 | 0.6341 | 0.6294 | -0.0047 | 0.6061 | 0.6274 | scored |
| away_back_to_back | 448 | 0.5977 | 0.5956 | -0.0021 | 0.6741 | 0.6786 | scored |
| dense_schedule | 1108 | 0.6158 | 0.6135 | -0.0023 | 0.6498 | 0.6552 | scored |

## What This Means

- The next-gen candidate clears the current promotion gates.
- Playoff coverage is now present with `166` held-out games.
- Missing-player coverage is present with `2602` held-out games.
- The next production step is a shadow/live promotion review with monitoring on playoff, calibration, and missing-player slices.
- Longer-term availability work should replace the historical absence proxy with richer official inactive history when available.
