# Active Milestone Tracker

> This tracker replaces the old “future phases” mindset for forward work.

| Milestone | Name | Status | Purpose |
|----------|------|--------|---------|
| M1 | [Player + Lineup Foundation](m1_player_lineup_foundation.md) | `[/]` In Progress | Build player-aware, lineup-aware pregame data and features |
| M2 | Strong Baselines Refresh | `[ ]` Planned | Add tougher benchmark models and clearer leaderboard |
| M3 | Regime-Aware Forecasting | `[ ]` Planned | Handle playoffs and season regimes explicitly |
| M4 | Next-Generation Representation | `[ ]` Planned | Add stronger player/lineup-aware model architectures |
| M5 | Explainable Live Forecasting | `[ ]` Planned | Add attribution, counterfactuals, and uncertainty |

## Current Landed Slices

- Player identity mapping, season roster metadata, and historical player-game log ingestion are in place.
- `src/features/player_value_features.py` now builds leakage-safe pregame player-value rows from recent minutes, role stability, starter-rate proxies, fantasy production, and plus-minus context.
- `src/features/projected_availability.py` now builds pregame player availability rows from recent-role baselines plus official injury-report overrides, and writes an unresolved entity audit table instead of silently dropping unmatched names.
- `src/features/lineup_features.py` now turns projected availability into leakage-safe team-game rotation features such as starter continuity, top-8 continuity, bench depth quality, minutes concentration, and missing value.
- `src/features/build_matchup_dataset.py` now writes a parallel `matchup_dataset_enriched.parquet` with player-value summaries and lineup-aware team features, while preserving the legacy base matchup dataset.
- `src/models/run_enriched_experiments.py` now runs a round-two leaderboard over the enriched matchup dataset, including M1 family ablations, optional LightGBM/CatBoost slots, Platt-calibrated tree variants, and playoff/context slice evaluation.
- `src/models/run_production_showdown.py` now compares the saved enriched benchmark winners against the shipped neural full-fusion and production ensemble artifacts on the same held-out split.
- `src/models/run_nextgen_ensemble.py` now trains enriched CatBoost/LightGBM input models and evaluates expanded meta-ensembles that add those probabilities to the current `neural + xgboost + elo` stack.
- Current best candidate: `nextgen_full / raw` at `0.6137` log loss, beating the saved production raw ensemble at `0.6152` on the current held-out split.

## Next Slice

- Promote the next-gen ensemble into the production training/inference path only after adding a hard validation round for playoffs and real nonzero injury/missing-player cases.
- Build a regime-aware split/report next, because the current held-out split still does not stress playoff games or meaningful missing-player rows.
