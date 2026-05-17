# Next-Gen Promotion Gate

Generated at `2026-05-17T14:20:34.359867+00:00`.

## Promotion Status

- Status: `blocked`
- Recommendation: Promote only after playoff and missing-player coverage gates pass.

## Gate Checks

| Check | Status | Detail |
|---|---|---|
| overall_log_loss_gain | pass | Next-gen must beat production by at least 0.001 log loss; observed -0.0015. |
| playoff_coverage | block | Need at least 100 held-out playoff games; found 0. |
| missing_player_coverage | block | Need at least 100 held-out games with nonzero missing-player value; found 0. |
| critical_slice_regression | pass | No critical slice exceeded the allowed regression threshold. |

## Slice Comparison

| Slice | Games | Production Log Loss | Next-Gen Log Loss | Delta | Production Acc | Next-Gen Acc | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| all_test | 2455 | 0.6152 | 0.6137 | -0.0015 | 0.6562 | 0.6550 | scored |
| regular_season | 2455 | 0.6152 | 0.6137 | -0.0015 | 0.6562 | 0.6550 | scored |
| playoffs | 0 | n/a | n/a | n/a | n/a | n/a | no_data |
| season_2023_24 | 1230 | 0.6160 | 0.6153 | -0.0007 | 0.6504 | 0.6480 | scored |
| season_2024_25 | 1225 | 0.6144 | 0.6121 | -0.0023 | 0.6620 | 0.6620 | scored |
| missing_player_impact | 0 | n/a | n/a | n/a | n/a | n/a | no_data |
| high_missing_value | 0 | n/a | n/a | n/a | n/a | n/a | no_data |
| high_context_confidence | 703 | 0.6316 | 0.6313 | -0.0003 | 0.6373 | 0.6330 | scored |
| low_context_confidence | 660 | 0.6180 | 0.6164 | -0.0016 | 0.6439 | 0.6500 | scored |
| home_back_to_back | 424 | 0.6317 | 0.6299 | -0.0018 | 0.6085 | 0.6203 | scored |
| away_back_to_back | 448 | 0.5965 | 0.5972 | 0.0007 | 0.6875 | 0.6786 | scored |
| dense_schedule | 1108 | 0.6151 | 0.6139 | -0.0012 | 0.6471 | 0.6498 | scored |

## What This Means

- The aggregate next-gen gain is useful, but it is not enough by itself.
- Current enriched historical evaluation has no held-out playoff rows.
- Current missing-player value is zero throughout the held-out set, so availability-aware claims are not yet validated.
- The next production step is to add real playoff history and real availability/inactive signal, then rerun this gate.
