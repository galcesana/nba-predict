# Next-Gen Promotion Gate

Generated at `2026-05-17T18:52:01.320577+00:00`.

## Promotion Status

- Status: `blocked`
- Recommendation: Promote only after real missing-player coverage is validated.

## Gate Checks

| Check | Status | Detail |
|---|---|---|
| overall_log_loss_gain | pass | Next-gen must beat production by at least 0.001 log loss; observed -0.0035. |
| playoff_coverage | pass | Need at least 100 held-out playoff games; found 166. |
| missing_player_coverage | block | Need at least 100 held-out games with nonzero missing-player value; found 0. |
| critical_slice_regression | pass | No critical slice exceeded the allowed regression threshold. |

## Slice Comparison

| Slice | Games | Production Log Loss | Next-Gen Log Loss | Delta | Production Acc | Next-Gen Acc | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| all_test | 2621 | 0.6196 | 0.6161 | -0.0035 | 0.6482 | 0.6501 | scored |
| regular_season | 2455 | 0.6177 | 0.6144 | -0.0033 | 0.6493 | 0.6525 | scored |
| playoffs | 166 | 0.6477 | 0.6425 | -0.0053 | 0.6325 | 0.6145 | scored |
| season_2023_24 | 1312 | 0.6198 | 0.6177 | -0.0021 | 0.6471 | 0.6456 | scored |
| season_2024_25 | 1309 | 0.6194 | 0.6146 | -0.0048 | 0.6494 | 0.6547 | scored |
| missing_player_impact | 0 | n/a | n/a | n/a | n/a | n/a | no_data |
| high_missing_value | 0 | n/a | n/a | n/a | n/a | n/a | no_data |
| high_context_confidence | 700 | 0.6340 | 0.6310 | -0.0029 | 0.6243 | 0.6343 | scored |
| low_context_confidence | 679 | 0.6185 | 0.6163 | -0.0022 | 0.6568 | 0.6465 | scored |
| home_back_to_back | 424 | 0.6341 | 0.6294 | -0.0047 | 0.6061 | 0.6297 | scored |
| away_back_to_back | 448 | 0.5977 | 0.5982 | 0.0005 | 0.6741 | 0.6719 | scored |
| dense_schedule | 1108 | 0.6158 | 0.6141 | -0.0017 | 0.6498 | 0.6516 | scored |

## What This Means

- The aggregate next-gen gain is useful, but promotion remains gated.
- Playoff coverage is now present with `166` held-out games.
- Current missing-player value is zero throughout the held-out set, so availability-aware claims are not yet validated.
- The next production step is to add real availability/inactive history, then rerun this gate.
