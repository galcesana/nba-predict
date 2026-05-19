# Current System Implementation Summary

> **Canonical summary of how the current production system works today.**
> Use this document before changing the model, data flows, or deployment logic.

---

## 1. System Scope

The current system is a calibrated NBA pregame forecasting pipeline that:

- trains on historical NBA game data
- predicts `P(home_win)` for scheduled games
- publishes weekly live forecast windows
- exposes results through Streamlit and FastAPI

The system is primarily **team-level**.

It already includes live injury/news overlays, but those overlays are still limited compared to a full player- and lineup-aware system.

---

## 2. Historical Data Foundation

Historical data lives under:

- `data/raw/`
- `data/interim/`
- `data/processed/`

Key tracked mapping:

- `data/mappings/team_to_idx.json`

Core historical processed tables:

- `data/processed/games.parquet`
- `data/processed/team_game_logs/team_game_logs.parquet`
- `data/processed/injury_features/injury_features.parquet`
- `data/processed/news_features/news_features.parquet`

The game and team-log fetch path now supports both `Regular Season` and `Playoffs`
from `configs/data_sources.yaml`. New processed rows preserve a `season_type` column
so evaluation can separate regular-season and playoff regimes.

New M1 foundation code now also exists for:

- player identity mapping
- season roster metadata loading
- processed player-game log building
- player-value feature building
- projected player availability snapshots
- lineup and rotation feature building
- enriched matchup-row generation

These utilities are present, but they are not yet fully integrated into the deployed forecasting model.

Historical modeling is built around anonymous team IDs `0-29`, with team names reserved for collection and UI only.

---

## 3. Feature Streams

The current predictor uses four feature streams.

### 3.1 Team performance stream

Built from:

- rolling team-game logs
- recent form windows
- matchup dataset features
- sequence builder utilities

Conceptually:

- recent game sequences per team
- schedule/rest context
- matchup deltas between home and away teams

This is the strongest and most mature stream in the current model.

### 3.2 Injury stream

Built in:

- [src/features/injury_features.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/features/injury_features.py>)

Current behavior:

- if official live injury-report rows are available, they are aggregated into team-level features
- otherwise the fallback is a heuristic proxy based on recent performance instability

Current injury features:

- `players_out_count`
- `players_questionable_count`
- `starter_out_count`
- `minutes_missing`
- `usage_missing`
- `estimated_value_missing`
- `injury_data_available`

Important limitation:

- this is still team-level aggregation, not player-level value modeling

### 3.3 News/sentiment stream

Built in:

- [src/features/news_features.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/features/news_features.py>)

Current behavior:

- cached news scores are loaded when available
- live news scores are fetched for current live forecast windows
- if no usable rows exist, the stream falls back to zero vectors

Current news features:

- `weighted_sentiment_24h`
- `weighted_sentiment_72h`
- `article_volume_24h`
- `negative_ratio_72h`
- `sentiment_volatility_72h`
- `avg_llm_confidence`
- `news_available`

Important limitation:

- coverage is often sparse, especially outside the immediate live window

### 3.4 Schedule/context stream

Built from:

- rest days
- back-to-backs
- schedule structure
- other matchup context features

This stream is stable and useful, but still relatively simple.

---

## 4. Model Stack

### 4.1 Baselines

Historical baselines include:

- Elo
- logistic regression
- XGBoost

The current deployed inference pipeline still loads:

- Elo ratings
- XGBoost model
- scaler artifact

### 4.1.1 Player foundation utilities

The repo now also includes the current M1 foundation files:

- [src/anonymization/player_mapping.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/anonymization/player_mapping.py>)
- [src/data/player_metadata.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/data/player_metadata.py>)
- [src/data/fetch_player_logs.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/data/fetch_player_logs.py>)

- player-value feature builders
- projected availability builders
- lineup and rotation feature builders
- enriched matchup-row builders

These provide:

- stable anonymous player indices
- season roster metadata normalization
- processed historical player-game logs
- leakage-safe pregame player-value estimates
- team-aware player-name resolution for injury-style aliases
- leakage-safe projected availability rows with explicit source type, timestamp, and confidence
- lineup continuity, depth, and missing-value features derived from those projected rows
- a parallel enriched matchup dataset for future model experiments

They are the first real player-and-lineup feature slice of the new roadmap, but they do not yet change the deployed forecast representation.

### 4.2 Neural model

Core model file:

- [src/models/matchup_fusion_model.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/models/matchup_fusion_model.py>)

Current structure:

- shared team sequence encoder
- context MLP
- injury encoder MLP
- news encoder MLP
- fusion head over concatenated home/away/difference/product representations

Current design strengths:

- clean multi-stream structure
- shared home/away encoders
- explicit support for ablations by stream

Current design limitations:

- team-level rather than player-/lineup-level
- injury and news streams are relatively low-dimensional
- fusion is still simple compared to the complexity of NBA matchup dynamics

### 4.3 Ensemble and calibration

Current production forecasting uses an ensemble layer and calibrator stored under:

- `models/ensembles/`

The ensemble is still intentionally simple and interpretable.

---

## 5. Training Pipeline

Main training entrypoint:

- [src/models/train.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/models/train.py>)

Current training behavior includes:

- time-based train/validation/test splits
- GRU-based sequence learning
- optional injury/news stream inclusion
- early stopping
- ablation support

Important discipline already enforced:

- no random-split evaluation
- rolling / season-aware data handling
- reproducibility through pinned seeds and tracked artifacts

---

## 6. Inference Pipeline

Main inference entrypoint:

- [src/models/predict.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/models/predict.py>)

Current behavior:

- loads trained Elo, XGBoost, neural, ensemble, and calibrator artifacts
- rebuilds the same feature streams used during training
- attempts live injury/news overlays for target games
- falls back to zero or proxy defaults when live context is missing
- returns prediction payloads with `context_details`
- optionally emits `nextgen_full_raw_v1` shadow probabilities when `--nextgen-shadow` or `NBA_PREDICT_NEXTGEN_SHADOW=1` is enabled

Important current limitation:

- production final probabilities still come from `ensemble_v1` until a separate promotion flips the default model version
- when no live aux data exists, inference still zero-fills or fallback-fills the aux streams rather than reasoning over player-level uncertainty

---

## 7. Live Forecast Generation

Forecast scripts:

- [src/app/predict_today.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/app/predict_today.py>)
- [src/app/publish_today.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/app/publish_today.py>)

Current publish design:

- weekly forecast window
- published JSON artifacts committed under `published/`
- manifest written to `published/manifest.json`
- no-games windows preserve the previous `latest.json`
- playoff filtering limits the board to the next scheduled game per series

Live publish metadata includes:

- status
- target date
- window start/end
- model version
- games count
- context coverage summary

---

## 8. Live Context Overlay

Phase 11 added live overlays rather than fully retraining the core model around live player-level data.

### Injury overlay

- official NBA injury-report PDFs are parsed when available
- team-level features are overlaid on top of fallback features
- coverage is partial for later-week games when reports do not yet exist

### News overlay

- current-slate team news is collected
- relevant articles are scored into structured sentiment rows
- game-level aggregates are built from those rows

The dashboard and manifest expose whether a prediction used:

- `live`
- `partial`
- `fallback`

for both injury and news context.

### 8.1 Projected availability foundation

Builder:

- [src/features/projected_availability.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/features/projected_availability.py>)

Current behavior:

- seeds each team-game from a recent-role baseline built only from prior player logs
- resolves official injury-report names to player IDs using season-aware aliases and team-aware filtering
- overlays official statuses onto those baseline rows without losing the player's historical role context
- applies a conservative historical absence proxy when a rotation player missed prior team games before the target date
- writes unresolved names to an audit table instead of silently dropping them

The absence proxy is not an official inactive feed. It is a leakage-safe historical signal derived only from prior player-game appearances, intended to make missing-player value measurable in backtests until richer official inactive history is available.

Current output fields include:

- `game_id`
- `date`
- `season`
- `team_idx`
- `player_id`
- `player_idx`
- `player_name`
- `status`
- `availability_score`
- `projection_confidence`
- `source_type`
- `availability_model_version`
- `source_timestamp`
- `report_reason`
- `recent_games_played`
- `expected_minutes`
- `player_value_score`
- `role_score`

### 8.2 Lineup and rotation feature foundation

Builder:

- [src/features/lineup_features.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/features/lineup_features.py>)

Current behavior:

- ranks projected players by effective role value after availability discounts
- infers projected starters and top-8 rotations
- compares those projected groups against the last prior game and recent prior games only
- computes leakage-safe team-game features such as starter continuity, top-8 continuity, minutes concentration, bench depth quality, rotation stability, lineup familiarity, and missing value

These new M1 outputs are not yet joined into the deployed training and inference rows, but the historical feature layer now exists and is test-covered.

### 8.3 Player-value foundation

Builder:

- [src/features/player_value_features.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/features/player_value_features.py>)

Current behavior:

- builds one pregame player-value row per candidate rotation player and team-game
- uses only prior games before the target date
- estimates role/value from recent minutes, minutes share, fantasy production, plus-minus, starter-rate proxy, and role stability
- ranks players within each team-game by a simple composite `player_value_score`

Current output fields include:

- `recent_minutes_avg`
- `recent_minutes_share`
- `recent_fantasy_points_avg`
- `recent_plus_minus_avg`
- `recent_starter_rate`
- `recent_role_stability`
- `player_value_score`
- `rotation_rank`

This layer is still heuristic, but it is materially richer than treating all missing players as equal.

### 8.4 Enriched matchup rows

Builder:

- [src/features/build_matchup_dataset.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/features/build_matchup_dataset.py>)

Current behavior:

- still writes the legacy team-level `matchup_dataset.parquet`
- now also writes a parallel `matchup_dataset_enriched.parquet`
- merges in home/away lineup features and projected player-value summaries
- computes diff columns for the new player-aware team aggregates

Important note:

- the deployed models do not use this enriched dataset yet
- it exists specifically to support the next model-experiment phase without breaking the current production stack

---

## 9. Product Surfaces

### Streamlit dashboard

Main app:

- [src/app/streamlit_app.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/app/streamlit_app.py>)

Data helpers:

- [src/app/dashboard_data.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/app/dashboard_data.py>)

The dashboard shows:

- weekly slate
- game detail
- archive
- performance
- calibration
- team form
- injury impact
- news sentiment

### FastAPI service

API app:

- [src/app/api.py](</C:/Users/galce/OneDrive/שולחן העבודה/FOLDERS/projects/nba-predict/src/app/api.py>)

Current endpoints:

- `/health`
- `/manifest`
- `/forecast/week`
- `/forecast/game/{game_id}`
- `/metrics`

The API is intentionally thin and reuses the same loaders as the dashboard.

---

## 10. Artifact Layout

Tracked publish artifacts:

- `published/daily/YYYY-MM-DD.json`
- `published/daily/latest.json`
- `published/manifest.json`

Tracked minimal inference bundle:

- selected processed tables
- scaler and neural checkpoint
- ensemble artifacts

Local-only outputs:

- `predictions/daily/`
- `predictions/historical_backtests/`

Additional M1 processed outputs:

- `data/processed/player_value_features/player_value_features.parquet`
- `data/processed/projected_availability/projected_availability.parquet`
- `data/processed/lineup_features/lineup_features.parquet`
- `data/processed/matchup_rows/matchup_dataset_enriched.parquet`

Round-two enriched experiment outputs:

- `docs/experiments/m1_enriched_matchup_results.json`
- `docs/experiments/m1_enriched_matchup_results.md`

The runner lives at `src/models/run_enriched_experiments.py` and compares the legacy feature
stack against enriched M1 variants, feature-family ablations, calibrated tree variants, and
playoff/context slices. It also invalidates cached enriched, projected-availability, and lineup
artifacts when feature-stack version stamps change, so rebuilt experiments actually pick up
availability logic changes.

Production showdown outputs:

- `docs/experiments/production_stack_showdown.json`
- `docs/experiments/production_stack_showdown.md`

The showdown runner lives at `src/models/run_production_showdown.py` and re-scores the saved
production neural full-fusion and ensemble artifacts against the latest enriched benchmark report.

Next-generation ensemble outputs:

- `docs/experiments/nextgen_ensemble_results.json`
- `docs/experiments/nextgen_ensemble_results.md`

The next-gen ensemble runner lives at `src/models/run_nextgen_ensemble.py`. It trains enriched
CatBoost/LightGBM input models, merges those probabilities with the current production
`neural + xgboost + elo` inputs, and scores expanded logistic meta-model variants.

After rebuilding with playoff rows, `nextgen_full / raw` is the best candidate on the current
held-out split: `0.6161` log loss, `0.6501` accuracy, and `0.7212` ROC-AUC. The saved production
raw ensemble remains the comparison baseline at `0.6196` log loss.

Next-generation promotion gate outputs:

- `docs/experiments/nextgen_promotion_gate.json`
- `docs/experiments/nextgen_promotion_gate.md`

The promotion gate runner lives at `src/models/run_nextgen_validation.py`. It compares production
raw ensemble probabilities against next-gen raw probabilities across aggregate, per-season,
schedule-stress, context-confidence, playoff, and missing-player-impact slices.

Current promotion status is `ready` for shadow/live promotion review: the next-gen candidate clears
the aggregate log-loss check, playoff coverage gate, missing-player coverage gate, and critical
slice-regression gate.

Shadow inference is available but opt-in. `python -m src.app.predict_today --nextgen-shadow` and
`python -m src.app.publish_today --nextgen-shadow` keep the production final probability as
`ensemble_v1` while adding `component_outputs.nextgen_shadow_probability`, enriched CatBoost/LightGBM
probabilities, and `shadow_model_version=nextgen_full_raw_v1` for review.

The scheduled GitHub Actions publisher uses `python -m src.app.publish_today --nextgen-shadow`, so
tracked deployment slates refresh both the production forecast and the shadow review fields.

The Streamlit dashboard exposes those candidate outputs in `Model Lab`, a dedicated review page that
shows production vs shadow home-win probabilities, candidate deltas, pick flips, candidate component
probabilities, and the shadow artifact version for the loaded slate.

The first coverage fix is complete: historical game and player-log fetchers request playoff rows,
carry `season_type` forward, and the rebuilt evaluation now includes 166 held-out playoff games.
The second coverage fix is implemented as `historical_absence_proxy_v1`; the rebuilt evaluation now
includes 2,595 held-out missing-player-impact games. Longer-term availability work should replace
this proxy with richer official inactive history when available.

---

## 11. Current Strengths

The current system is already strong in these ways:

- leakage-aware historical modeling discipline
- calibrated probability framing
- clear separation between data, features, models, and product surfaces
- live publishing path
- honest context-coverage reporting
- dashboard + API parity through shared loaders

---

## 12. Current Weaknesses

These are the most important limitations to remember before extending the system:

1. **The model is still mostly team-level.**
2. **The new player foundation is not yet integrated into training or inference.**
3. **The new player-value layer is still heuristic and not yet learned end-to-end.**
4. **Injury value is still heuristic in many cases.**
5. **News coverage is still sparse and often fallback-heavy.**
6. **Playoff handling is stronger in publishing logic than in model design.**
7. **Displayed explanations are still partly heuristic rather than fully learned attribution.**
8. **The ensemble is simple and not yet context-aware or regime-aware.**
9. **The next-gen candidate is ready for shadow/live review, but official inactive-history coverage is still a future quality upgrade.**

---

## 13. What Should Not Be Accidentally Broken

Any future redesign should preserve:

- anonymous team IDs as the internal join backbone
- strict pregame leakage safety
- time-based validation
- calibrated probabilities as the main output
- graceful fallback behavior for incomplete live context
- parity between the dashboard and API forecast source

---

## 14. How To Use This Summary

Use this doc before:

- changing feature schemas
- replacing the model backbone
- refactoring publishing
- adding player/lineup data
- changing dashboard/API forecast semantics

For future work planning, pair this document with:

- [next_generation_model_roadmap.md](next_generation_model_roadmap.md)
