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
- `src/features/player_value_features.py` now builds leakage-safe pregame player-value rows from recent minutes, role stability, starter-rate proxies, fantasy production, plus-minus context, expected usage proxy, value per minute, value confidence, role tier, and prior missed-game replacement risk.
- `src/features/projected_availability.py` now builds pregame player availability rows from recent-role baselines, official injury-report overrides, and a leakage-safe historical absence proxy that discounts rotation players who missed prior team games. It now also emits availability-adjusted projected minutes, available value, missing value, and replacement-risk-weighted missing value per player.
- `src/features/lineup_features.py` now turns projected availability into leakage-safe team-game rotation features such as starter continuity, top-8 continuity, bench depth quality, minutes concentration, missing value, and expected missing replacement risk.
- `src/features/build_matchup_dataset.py` now writes a parallel `matchup_dataset_enriched.parquet` with player-value summaries, projected minutes/missing-value confidence summaries, replacement-risk-weighted missing-value summaries, and lineup-aware team features, while preserving the legacy base matchup dataset.
- `docs/player_lineup_availability_upgrade_plan.md` captures the detailed upgrade plan for making the model understand tonight's actual roster and lineup quality.
- `src/models/run_enriched_experiments.py` now runs a round-two leaderboard over the enriched matchup dataset, including M1 family ablations, optional LightGBM/CatBoost slots, Platt-calibrated tree variants, and playoff/context slice evaluation.
- `src/models/run_production_showdown.py` now compares the saved enriched benchmark winners against the shipped neural full-fusion and production ensemble artifacts on the same held-out split.
- `src/models/run_nextgen_ensemble.py` now trains model-specific enriched CatBoost/LightGBM input models and evaluates expanded meta-ensembles that add those probabilities to the current `neural + xgboost + elo` stack.
- `src/models/run_nextgen_validation.py` now runs a promotion gate across aggregate, per-season, schedule-stress, context-confidence, playoff, and missing-player slices before any live model promotion.
- `src/models/predict.py`, `src/app/predict_today.py`, and `src/app/publish_today.py` now promote `nextgen_full_value_tuned_v2` as the default production model while retaining `ensemble_v1_probability` as a baseline comparison field.
- The Streamlit dashboard now includes a `Model Lab` page that compares the promoted next-gen probability against the previous `ensemble_v1` baseline, candidate deltas, pick flips, and enriched CatBoost/LightGBM component outputs for the loaded slate.
- The daily GitHub Actions publisher runs `python -m src.app.publish_today --nextgen-shadow`, so the deployed slate refreshes the promoted next-gen forecast and comparison fields together.
- The small next-gen artifact bundle and historical player-log inference bundle are tracked so clean checkouts can emit promoted next-gen probabilities.
- `src/data/fetch_games.py` and `src/data/fetch_player_logs.py` now support configured `Regular Season` + `Playoffs` ingestion and preserve `season_type` into processed rows for regime-aware evaluation.
- `src/models/run_enriched_experiments.py` now detects stale cached matchup/player feature artifacts when the game universe or feature-stack version changes, so adding playoff rows or changing availability logic forces the enriched stack to rebuild instead of silently reusing old caches.
- `src/models/run_nextgen_ensemble.py` now also invalidates stale enriched input prediction caches, and `src/models/run_production_showdown.py` reports saved-production coverage when old artifacts do not cover every enriched test row.
- Latest enriched-feature experiment after the roster-value upgrade: `enriched_value_only / catboost` at `0.6163` log loss, `0.6616` accuracy, and `0.7169` ROC-AUC. This is now the configured CatBoost input family for the promoted next-gen ensemble.
- Current best value-tuned candidate: `nextgen_full / raw` at `0.6159` log loss and `0.6536` accuracy, beating the saved production raw ensemble at `0.6196` log loss on the current held-out split.
- Current promotion status: promoted to live production. The validation gate passed with 166 held-out playoff games and 2,602 held-out missing-player-impact games through `historical_absence_proxy_v1`.
- `replacement_risk_v1` has been evaluated but should not be promoted as-is: its best result was `enriched_no_confidence / catboost` at `0.6172` log loss and `0.6570` accuracy, below the current value-tuned best of `0.6163` log loss and `0.6616` accuracy. It did improve the playoff slice, so it remains useful as a future regime-aware experiment.

## Next Slice

- Monitor the promoted model in `Model Lab`, especially playoff, calibration, and missing-player slices.
- Keep `NBA_PREDICT_PRODUCTION_MODEL=ensemble` available as the rollback switch if live behavior looks wrong.
- Do not refresh next-gen production inputs from `replacement_risk_v1` yet.
- Next modeling slice should focus on direct live injury-report integration into projected availability, confirmed/expected starters, or a playoff-gated version of replacement risk.
