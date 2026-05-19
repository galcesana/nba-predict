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
- `src/features/player_value_features.py` now builds leakage-safe pregame player-value rows from recent minutes, role stability, starter-rate proxies, fantasy production, plus-minus context, expected usage proxy, value per minute, value confidence, and role tier.
- `src/features/projected_availability.py` now builds pregame player availability rows from recent-role baselines, official injury-report overrides, and a leakage-safe historical absence proxy that discounts rotation players who missed prior team games. It now also emits availability-adjusted projected minutes, available value, and missing value per player.
- `src/features/lineup_features.py` now turns projected availability into leakage-safe team-game rotation features such as starter continuity, top-8 continuity, bench depth quality, minutes concentration, and missing value.
- `src/features/build_matchup_dataset.py` now writes a parallel `matchup_dataset_enriched.parquet` with player-value summaries, projected minutes/missing-value confidence summaries, and lineup-aware team features, while preserving the legacy base matchup dataset.
- `docs/player_lineup_availability_upgrade_plan.md` captures the detailed upgrade plan for making the model understand tonight's actual roster and lineup quality.
- `src/models/run_enriched_experiments.py` now runs a round-two leaderboard over the enriched matchup dataset, including M1 family ablations, optional LightGBM/CatBoost slots, Platt-calibrated tree variants, and playoff/context slice evaluation.
- `src/models/run_production_showdown.py` now compares the saved enriched benchmark winners against the shipped neural full-fusion and production ensemble artifacts on the same held-out split.
- `src/models/run_nextgen_ensemble.py` now trains model-specific enriched CatBoost/LightGBM input models and evaluates expanded meta-ensembles that add those probabilities to the current `neural + xgboost + elo` stack.
- `src/models/run_nextgen_validation.py` now runs a promotion gate across aggregate, per-season, schedule-stress, context-confidence, playoff, and missing-player slices before any live model promotion.
- `src/models/predict.py`, `src/app/predict_today.py`, and `src/app/publish_today.py` now support opt-in `nextgen_full_value_tuned_v2` shadow outputs through `--nextgen-shadow` / `NBA_PREDICT_NEXTGEN_SHADOW=1`, while production final probabilities remain `ensemble_v1`.
- The Streamlit dashboard now includes a `Model Lab` page that compares production vs next-gen shadow probabilities, candidate deltas, pick flips, and enriched CatBoost/LightGBM component outputs for the loaded slate.
- The daily GitHub Actions publisher now runs `python -m src.app.publish_today --nextgen-shadow`, so `Model Lab` updates automatically with each scheduled deployment publish.
- The small next-gen shadow bundle and historical player-log inference bundle are tracked so clean checkouts can emit candidate probabilities for review.
- `src/data/fetch_games.py` and `src/data/fetch_player_logs.py` now support configured `Regular Season` + `Playoffs` ingestion and preserve `season_type` into processed rows for regime-aware evaluation.
- `src/models/run_enriched_experiments.py` now detects stale cached matchup/player feature artifacts when the game universe or feature-stack version changes, so adding playoff rows or changing availability logic forces the enriched stack to rebuild instead of silently reusing old caches.
- `src/models/run_nextgen_ensemble.py` now also invalidates stale enriched input prediction caches, and `src/models/run_production_showdown.py` reports saved-production coverage when old artifacts do not cover every enriched test row.
- Latest enriched-feature experiment after the roster-value upgrade: `enriched_value_only / catboost` at `0.6163` log loss, `0.6616` accuracy, and `0.7169` ROC-AUC. This is now the configured CatBoost input family for the next shadow ensemble rebuild.
- Current best value-tuned candidate: `nextgen_full / raw` at `0.6159` log loss and `0.6536` accuracy, beating the saved production raw ensemble at `0.6196` log loss on the current held-out split.
- Current promotion status: `ready` for shadow/live promotion review. Playoff coverage passes with 166 held-out playoff games, and missing-player-impact coverage now covers 2,602 held-out games through `historical_absence_proxy_v1`.

## Next Slice

- Run a live shadow publication with `python -m src.app.publish_today --nextgen-shadow` and review `Model Lab` before any final production flip.
- Keep monitoring playoff, calibration, and missing-player slices before promoting the shadow probability to the default final probability.
