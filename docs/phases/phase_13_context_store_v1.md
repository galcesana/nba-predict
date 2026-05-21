# Phase 13 - Context Store V1

| Field | Value |
|-------|-------|
| **Size** | L (1-2 weeks for 13A-13C) |
| **Status** | `[ ]` Planned |
| **Depends on** | Phase 10 + Phase 11 + Phase 12 |
| **Unlocks** | Prospective injury/news/context training corpus for future models |

---

## Goal

Build a prospective context store that saves exactly what the live model knew before each game:

- live schedule state
- model-visible feature values
- official injury context
- filtered/scored news context
- prediction outputs
- later outcomes in a separate leakage-safe table

The immediate target is Phase 13A through Phase 13C. Outcome hydration, training export, and dashboard quality views are planned follow-up slices.

---

## Reference Design

Canonical design doc:

- [../context_store_v1_plan.md](../context_store_v1_plan.md)

Core storage default:

```text
data/context_store/context.duckdb
data/context_store/parquet/
```

Core leakage rule:

```text
Pregame context is append-only. Outcomes are hydrated later into a separate table.
```

---

## Phase 13A - Context Store Schema

Goal: create the storage foundation without changing prediction behavior.

Deliverables:

- [ ] Add path constants for `data/context_store/`
- [ ] Add explicit `.gitignore` rules for generated context store DB/parquet files
- [ ] Add `src/context_store/` package scaffold
- [ ] Add schema definitions for:
  - `forecast_runs`
  - `game_snapshots`
  - `model_features`
  - `injury_context`
  - `news_articles`
  - `news_scores`
  - `prediction_outputs`
  - `outcomes`
- [ ] Add `python -m src.context_store.init`
- [ ] Add tests proving empty tables initialize successfully
- [ ] Add tests proving pregame tables do not contain outcome columns

Acceptance checks:

```bash
python -m pytest tests/test_context_store.py -q
python -m ruff check src/context_store tests/test_context_store.py
```

---

## Phase 13B - Save Published Forecast Context

Goal: every successful publish writes a full context snapshot.

Deliverables:

- [ ] Add a context-store writer for published payload + manifest
- [ ] Generate a stable `run_id` per publish attempt
- [ ] Save one `forecast_runs` row per successful publish
- [ ] Save one `game_snapshots` row per game
- [ ] Save one `prediction_outputs` row per game
- [ ] Save long-format `model_features` rows for model-visible features available from the prediction run
- [ ] Preserve existing `published/` JSON behavior
- [ ] Avoid partial context-store writes on failed publish

Acceptance checks:

```bash
python -m pytest tests/test_context_store.py tests/test_publishing.py -q
python -m ruff check src/context_store src/app/publish_today.py tests/test_context_store.py tests/test_publishing.py
```

---

## Phase 13C - Capture Live Injury And News Inputs

Goal: preserve live context that future training cannot reconstruct from historical box scores.

Deliverables:

- [ ] Save injury feature rows used by prediction into `injury_context`
- [ ] Save news feature rows used by prediction into `news_scores`
- [ ] Save article metadata into `news_articles`
- [ ] Store whether each article was included or excluded from model features
- [ ] Store exclusion reasons such as `betting_or_promo`, `low_relevance`, `stale`, and `duplicate`
- [ ] Store injury report timestamp/source URL, article publish time, and news collection time
- [ ] Ensure every context row has `as_of_utc`

Acceptance checks:

```bash
python -m pytest tests/test_context_store.py tests/test_live_context.py tests/test_publishing.py -q
python -m ruff check src/context_store src/features src/nlp tests/test_context_store.py tests/test_live_context.py
```

---

## Phase 13D - Outcome Hydration

Goal: add actual results later without contaminating pregame context.

Deliverables:

- [ ] Add `python -m src.context_store.hydrate_outcomes`
- [ ] Fetch final score/winner for stored game ids
- [ ] Save results only to `outcomes`
- [ ] Make hydration idempotent
- [ ] Skip unfinished games safely
- [ ] Add tests proving pregame context tables remain unchanged

Acceptance checks:

```bash
python -m pytest tests/test_context_store.py -q
```

---

## Phase 13E - Future Training Export

Goal: convert accumulated prospective snapshots into a future training dataset.

Deliverables:

- [ ] Add `python -m src.context_store.export_training_dataset`
- [ ] Support `latest-before-tipoff`, `first-published`, and `morning-publish` snapshot policies
- [ ] Choose one forecast snapshot per game before joining outcomes
- [ ] Pivot `model_features` wide
- [ ] Write `data/processed/context_training/context_training_dataset.parquet`
- [ ] Refuse rows without outcomes unless explicitly allowed

Acceptance checks:

```bash
python -m pytest tests/test_context_store.py -q
```

---

## Phase 13F - Dashboard/Data Quality View

Goal: make the stored corpus visible and auditable.

Deliverables:

- [ ] Add context-store summary loader
- [ ] Show stored forecast runs and games captured
- [ ] Show injury/news coverage over time
- [ ] Show excluded article rate
- [ ] Show outcome hydration rate
- [ ] Add empty-state behavior when no context store exists

Acceptance checks:

```bash
python -m pytest tests/test_dashboard.py tests/test_context_store.py -q
```

---

## Definition of Done For Phase 13A-13C

- [ ] Context store initializes locally without requiring external services
- [ ] Generated context-store files are gitignored
- [ ] Successful publish writes forecast, game, prediction, feature, injury, and news context rows
- [ ] Failed publish leaves context store untouched
- [ ] Betting/prop/promo articles can be stored as excluded instead of model-fed
- [ ] All pregame context rows include `as_of_utc`
- [ ] Existing Streamlit, FastAPI, and `published/` contracts remain unchanged
- [ ] Focused validation passes
- [ ] Full validation passes
- [ ] Tracking docs are updated

---

## Notes & Learnings

- Planned after the live news relevance fix. The current system can fetch and feed live news, but without a context store those prospective signals are lost after each run.
- The first implementation should prioritize append-only capture over sophisticated analytics.
