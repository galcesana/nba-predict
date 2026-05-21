# Context Store V1 Plan

> Canonical design reference for building a prospective forecast-context warehouse.

## Summary

Context Store V1 will preserve the exact information available to the live model before each game:

- scheduled game metadata
- model-visible feature values
- live injury context
- live news/article context
- prediction outputs and component probabilities
- later final outcomes in a separate table

The purpose is not to backfill context we never had. The purpose is to stop losing live context from this point forward so future models can train on real prospective injury/news/lineup signals.

## Implementation Status

Phase 13A through Phase 13C are implemented:

- `src.context_store.schema` initializes the DuckDB database and matching Parquet directories.
- `src.context_store.writer` appends successful publish snapshots to `forecast_runs`, `game_snapshots`, `prediction_outputs`, `model_features`, `injury_context`, `news_articles`, and `news_scores`.
- `src.data.fetch_news` now annotates fetched articles with inclusion/exclusion metadata so rejected betting, promo, stale, duplicate, or low-relevance articles can still be audited later.
- `src.app.publish_today` preserves the existing `published/` JSON contract and records context snapshots after successful production publishes.
- `.github/workflows/publish_daily.yml` uploads the generated DuckDB/Parquet context bundle as a GitHub Actions artifact for each successful publish run, and can optionally copy the same zip to Google Drive when Drive secrets are configured.

Phase 13D through Phase 13F remain planned: outcome hydration, future training export, and dashboard/data-quality views. V1 model-feature capture stores numeric fields available in the published payload and component/context metadata; deeper hidden tensors or raw training matrices are intentionally deferred until the context store has real prospective volume.

## Core Rules

1. Every pregame record must include `run_id`, `game_id`, and `as_of_utc`.
2. Pregame context is append-only. It is never rewritten after the prediction run.
3. Postgame results are stored only in `outcomes`.
4. Future training must choose a declared snapshot policy before joining outcomes.
5. Full article bodies are not stored. Store title, source, link, timestamps, summary hash, extracted scores, and exclusion reason.
6. Context Store V1 must not change the existing published forecast JSON contract.

## Storage Default

Use local DuckDB plus Parquet:

```text
data/context_store/context.duckdb
data/context_store/parquet/
  forecast_runs/
  game_snapshots/
  model_features/
  injury_context/
  news_articles/
  news_scores/
  prediction_outputs/
  outcomes/
```

Generated context-store artifacts should be gitignored in Phase 13A. They are intended as a local/prospective data asset first, not a tracked deployment artifact.

## Table Contracts

### `forecast_runs`

One row per publish attempt.

```text
run_id
as_of_utc
target_date
window_start
window_end
timezone
git_commit
model_version
production_model
shadow_enabled
status
games_count
schedule_status
injury_status
news_status
created_by
```

### `game_snapshots`

One row per game per run.

```text
run_id
game_id
game_date
game_time_utc
season
slate_type
home_team_idx
away_team_idx
home_team_abbr
away_team_abbr
game_label
game_sub_label
series_text
game_status_text
schedule_source
if_necessary
as_of_utc
```

### `model_features`

Long-format exact model-visible features.

```text
run_id
game_id
entity_scope
feature_group
feature_name
feature_value
feature_dtype
source
as_of_utc
```

Use long format so new feature families can be added without schema migrations. Future training exports can pivot this table wide.

### `injury_context`

One row per team/player/status item when player-level detail exists. Team-level fallback rows are allowed when player-level detail is unavailable.

```text
run_id
game_id
team_idx
player_name_or_id
status
reason
report_generated_at
report_source_url
injury_data_available
estimated_value_missing
players_out_count
players_questionable_count
source_mode
as_of_utc
```

`source_mode` should use the same vocabulary as the prediction payload: `live`, `partial`, `pending`, or `fallback`.

### `news_articles`

Article metadata and inclusion status.

```text
run_id
game_id
team_idx
article_id
published_at
collected_at
source
title
summary_hash
link
included_in_model
excluded_reason
article_relevance
article_relevance_reason
as_of_utc
```

Expected `excluded_reason` examples:

- `betting_or_promo`
- `low_relevance`
- `stale`
- `duplicate`

### `news_scores`

Structured article-level scores.

```text
run_id
game_id
team_idx
article_id
overall_sentiment
injury_concern
pressure
team_cohesion
motivation
llm_confidence
scorer_version
as_of_utc
```

### `prediction_outputs`

One row per game per run.

```text
run_id
game_id
home_win_probability
away_win_probability
predicted_winner
confidence_bucket
elo_probability
tabular_probability
sequence_probability
ensemble_v1_probability
nextgen_probability
final_probability
top_model_factors_json
context_details_json
as_of_utc
```

### `outcomes`

Hydrated later, after games finish.

```text
game_id
final_home_score
final_away_score
home_win
completed_at
outcome_source
hydrated_at
```

This table is intentionally separate from every pregame context table.

## Implementation Phases

### Phase 13A - Context Store Schema

Create the storage foundation without changing prediction behavior.

- add path constants for `data/context_store/`
- add `.gitignore` rules for generated DuckDB and Parquet files
- add schema initialization for the empty database and Parquet directories
- add schema tests for required columns and leakage boundaries

### Phase 13B - Save Published Forecast Context

Save a snapshot for each successful publish.

- write `forecast_runs`, `game_snapshots`, and `prediction_outputs`
- write long-format `model_features` from model-visible payload/features
- preserve existing `published/` behavior
- avoid partial writes on failed publish

### Phase 13C - Capture Live Injury And News Inputs

Preserve the context that cannot be reconstructed from historical box scores.

- save injury feature rows used by prediction
- save news feature rows used by prediction
- save article metadata and inclusion/exclusion status
- preserve source timestamps for reports, articles, and collection time

### Phase 13D - Outcome Hydration

Add actual results later without contaminating pregame context.

- fetch final score/winner for stored game ids
- write only to `outcomes`
- make hydration idempotent
- skip unfinished games safely

### Phase 13E - Future Training Export

Convert accumulated prospective snapshots into a training dataset.

```bash
python -m src.context_store.export_training_dataset \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --snapshot-policy latest-before-tipoff
```

Supported snapshot policies:

- `latest-before-tipoff`
- `first-published`
- `morning-publish`

The exporter must choose one snapshot per game before joining outcomes.

### Phase 13F - Dashboard/Data Quality View

Expose context-store health and coverage.

- stored runs
- stored games
- injury coverage over time
- news coverage over time
- excluded article rate
- outcome hydration rate

## Public Interfaces

Documented CLI targets:

```bash
python -m src.context_store.init
python -m src.context_store.hydrate_outcomes
python -m src.context_store.export_training_dataset --start-date YYYY-MM-DD --end-date YYYY-MM-DD --snapshot-policy latest-before-tipoff
```

Generated paths:

```text
data/context_store/context.duckdb
data/context_store/parquet/
data/processed/context_training/context_training_dataset.parquet
```

## Test Plan

Add `tests/test_context_store.py` with coverage for:

- schema initialization
- append-only forecast run writes
- prediction payload serialization
- injury/news context capture
- betting/promo article exclusion persistence
- outcome hydration isolation
- training export snapshot selection

Focused validation:

```bash
python -m pytest tests/test_context_store.py tests/test_live_context.py tests/test_publishing.py -q
```

Full validation:

```bash
python -m ruff check .
python -m pytest tests -q
```

## Assumptions

- Context Store V1 starts local: DuckDB + Parquet, not Postgres/Supabase.
- Phase 13A adds `.gitignore` rules before any code writes generated context-store files.
- Full article bodies are not stored.
- The first implementation covers Phase 13A through Phase 13C.
- Outcome hydration and training export are documented now but implemented after snapshots begin accumulating.
- `latest-before-tipoff` is the default future training snapshot policy.
- Existing publish, dashboard, and API behavior must remain unchanged while context capture is added.
