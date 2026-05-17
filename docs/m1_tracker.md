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

## Next Slice

- Start model experiments against `matchup_dataset_enriched.parquet` so the new player-aware representation is actually measured.
- Decide which enriched columns should be promoted into the next-generation training stack versus kept as diagnostic/context features.
