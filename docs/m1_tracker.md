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
- `src/features/projected_availability.py` now builds pregame player availability rows from recent-role baselines plus official injury-report overrides, and writes an unresolved entity audit table instead of silently dropping unmatched names.
- `src/features/lineup_features.py` now turns projected availability into leakage-safe team-game rotation features such as starter continuity, top-8 continuity, bench depth quality, minutes concentration, and missing value.

## Next Slice

- Add player-value features so missing-player impact is grounded in stronger pregame role/value estimates.
- Join the new projected availability and lineup outputs into enriched matchup rows for future model experiments.
