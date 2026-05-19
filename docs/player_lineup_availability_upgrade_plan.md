# Player + Lineup Availability Upgrade Plan

## Purpose

Improve forecast quality by making the model understand the actual roster expected to play in a
specific game, not only the team's historical averages.

The current production model is still mostly team-level. The next-gen shadow stack already adds
player-value, projected-availability, and lineup-continuity signals, but the roster intelligence
layer is still early. This plan upgrades that layer while keeping production probabilities safe
behind the existing shadow-review path.

## Guiding Principles

- Preserve strict pregame leakage safety: every player, lineup, and availability feature must be
  built from information available before the target game.
- Keep production stable until shadow validation is convincing.
- Prefer probabilistic availability over binary in/out flags.
- Track source type, timestamp, and confidence for every availability claim.
- Improve explanations as well as metrics.

## Success Criteria

- Next-gen shadow improves aggregate held-out log loss versus `ensemble_v1`.
- No material regression on playoff, missing-player, schedule-stress, or low-context-confidence
  validation slices.
- Live `Model Lab` shows interpretable roster deltas: missing value, projected minutes, confidence,
  and candidate probability movement.
- Cached feature tables are versioned so stale player/lineup artifacts rebuild automatically.

## Workstream 1: Player Value Model

Current state:

- `src/features/player_value_features.py` builds leakage-safe player rows from recent minutes,
  fantasy production, plus-minus, starter proxy, minutes share, and role stability.

Upgrade plan:

- Add expected usage proxy from recent points, assists, and rebounds.
- Add recent value per minute from fantasy production per minute.
- Add value confidence from role stability, recent appearances, and minutes share.
- Add role tier so downstream features can distinguish core starters, rotation players, and depth.
- Version the player-value artifact to force rebuilds when formulas change.

Initial implementation slice:

- `PLAYER_VALUE_FEATURE_VERSION = value_confidence_usage_v1`
- New columns:
  - `recent_usage_proxy`
  - `recent_value_per_minute`
  - `value_confidence`
  - `role_tier`
  - `player_value_model_version`

Replacement-risk implementation slice:

- `PLAYER_VALUE_FEATURE_VERSION = replacement_risk_v1`
- New columns:
  - `recent_absence_games`
  - `recent_absence_net_rating_delta`
  - `replacement_risk_score`
- The feature compares a player's prior games played versus prior games missed within the same
  rolling window, using only pre-target team net rating or point differential.
- Positive replacement risk means the team has recently performed worse when that player missed
  games, scaled by role stability and player-value confidence.

## Workstream 2: Projected Availability

Current state:

- `src/features/projected_availability.py` combines recent role baselines, official injury-report
  overrides, and a leakage-safe historical absence proxy.

Upgrade plan:

- Carry player value confidence and usage into projected availability rows.
- Emit availability-adjusted projected minutes.
- Emit projected available and missing player value per player-game.
- Version the projected-availability artifact to force rebuilds when formulas change.

Initial implementation slice:

- `PROJECTED_AVAILABILITY_VERSION = availability_value_confidence_v1`
- New columns:
  - `expected_usage_proxy`
  - `value_confidence`
  - `role_tier`
  - `projected_minutes`
  - `projected_value_available`
  - `projected_value_missing`

Replacement-risk implementation slice:

- `PROJECTED_AVAILABILITY_VERSION = replacement_risk_v1`
- New columns:
  - `recent_absence_games`
  - `recent_absence_net_rating_delta`
  - `replacement_risk_score`
  - `projected_replacement_value_missing`
- `projected_replacement_value_missing` keeps the original missing-value estimate but increases it
  when the player's historical missed-game signal indicates harder replacement conditions.

## Workstream 3: Team-Level Lineup Summaries

Current state:

- `src/features/lineup_features.py` and `src/features/build_matchup_dataset.py` aggregate
  projected availability into team-game and home/away/diff matchup features.

Upgrade plan:

- Add team-level availability-adjusted minutes.
- Add top-8 missing minutes.
- Add top-8 player-value confidence.
- Version the enriched matchup stack so stale enriched rows rebuild.

Initial implementation slice:

- `ENRICHED_FEATURE_STACK_VERSION = player_lineup_value_confidence_v1`
- New matchup summary columns:
  - `projected_minutes_available`
  - `projected_minutes_missing`
  - `projected_top8_minutes_available`
  - `projected_top8_minutes_missing`
  - `projected_top8_value_confidence_mean`

Replacement-risk implementation slice:

- `LINEUP_FEATURE_VERSION = replacement_risk_v1`
- `ENRICHED_FEATURE_STACK_VERSION = replacement_risk_v1`
- New matchup and lineup summary columns:
  - `expected_missing_replacement_risk`
  - `projected_replacement_value_missing`
  - `projected_top5_replacement_value_missing`
  - `projected_top8_replacement_value_missing`
  - `projected_top8_replacement_risk_mean`
- The experiment loaders now require the enriched parquet stack version to match the code version
  before next-gen enriched input training can proceed.

## Workstream 4: Modeling Experiments

After the feature slice lands:

1. Rebuild M1 feature stack. Complete.
2. Rerun enriched experiments. Complete: `enriched_value_only / catboost` reached `0.6163` log loss and `0.6616` accuracy.
3. Rerun next-gen ensemble with refreshed value-tuned enriched inputs. Complete: `nextgen_full / raw` reached `0.6159` log loss and `0.6536` accuracy.
4. Rerun promotion gate. Complete: gate status is `ready`.
5. Compare against the prior `nextgen_full_raw_v1` shadow baseline, then label the rebuilt value-tuned bundle as `nextgen_full_value_tuned_v2`. Complete.
6. Rebuild and benchmark `replacement_risk_v1`. Pending heavy local run after this code slice lands.

Commands:

```bash
python -m src.models.run_enriched_experiments
python -m src.models.run_nextgen_ensemble --refresh-enriched-inputs
python -m src.models.run_nextgen_validation
```

On Windows without `make`, run the Python modules directly.

## Workstream 5: Product Surface

After experiments show improvement:

- Add roster diagnostics to `Model Lab`:
  - top missing players by projected value
  - projected minutes missing by team
  - availability confidence
  - shadow probability movement
- Candidate cleared promotion gates and `nextgen_full_value_tuned_v2` is now production; keep `ensemble_v1_probability` as the rollback/baseline comparison.

## Promotion Rule

Promote only if:

- aggregate log loss improves,
- playoff slice does not regress materially,
- missing-player-impact slice improves or remains stable,
- calibration does not worsen materially,
- live shadow slates look sane in `Model Lab`.
