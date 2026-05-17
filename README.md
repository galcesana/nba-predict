# NBA Game Outcome Prediction Model

> Predict calibrated win probabilities for NBA games using historical performance, team sequences, injury impact, and structured live news/team-spirit signals.

---

## Overview

This project builds an NBA game-outcome prediction system that outputs **calibrated probabilities**, not just hard picks. For a scheduled game, the model outputs:

```json
{
  "home_win_probability": 0.64,
  "away_win_probability": 0.36,
  "confidence": "medium"
}
```

The system fuses four signal streams:

| Stream | Source | Encoder |
|--------|--------|---------|
| **Team Performance** | Historical game sequences (last 20 games) | GRU / TCN |
| **Injury Impact** | Official injury reports / proxy injury signals | MLP |
| **News / Team Spirit** | Structured sentiment from recent team articles | MLP |
| **Schedule Context** | Rest days, back-to-back, travel | MLP |

All streams feed into a **matchup fusion model** -> **calibration layer** -> `P(home_win)`.

> **This is a sports analytics and forecasting project, not a betting project.**

---

## Core Principles

1. **Predict probabilities, not just winners** - a 56% prediction is more honest than "home wins".
2. **No data leakage** - only information available before tip-off is used.
3. **Anonymous team IDs** - the model learns from team features, not team names.
4. **LLM = feature extractor** - the LLM extracts structured signals, it does not predict winners.
5. **Reproducibility** - pinned seeds, versioned data snapshots, logged experiments.

---

## Project Structure

```text
nba-predict/
|- README.md
|- requirements.txt
|- Makefile
|- streamlit_app.py
|- configs/                      # YAML configuration
|- data/
|  |- raw/                       # Cached API responses (Parquet) - gitignored
|  |- interim/                   # Cleaned intermediate data - gitignored
|  |- processed/                 # Feature tables (minimal inference bundle tracked)
|  `- mappings/                  # team_to_idx.json
|- models/                       # Saved model artifacts (minimal inference bundle tracked)
|- predictions/                  # Local daily + backtest prediction JSONs
|- published/                    # Tracked deployment forecast JSONs
|- src/
|  |- data/providers/            # DataProvider ABC + implementations
|  |- anonymization/             # Team/player -> anonymous ID
|  |- nlp/                       # LLM sentiment extraction pipeline
|  |- features/                  # Rolling, schedule, injury, news features
|  |- models/                    # Elo, tabular, neural, ensemble, calibration
|  |- app/                       # Daily prediction, publishing, Streamlit dashboard, FastAPI service
|  `- utils/                     # Logging, paths, validation
|- tests/                        # Comprehensive test suite
`- docs/phases/                  # Phase-by-phase implementation guides
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### 1. Set Up the Environment

```bash
git clone <repo-url>
cd nba-predict

python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# or: .venv\Scripts\activate  # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

You can also install dependencies with:

```bash
make setup
```

### 2. Build the Local Data and Models

On a fresh checkout, the full local pipeline still expects the historical processed tables and trained model artifacts.

```bash
make fetch-data
make build-features
make train-baseline
make train-model
make evaluate
```

What each step does:

- `make fetch-data` downloads and caches historical NBA data. `configs/data_sources.yaml`
  now includes both `Regular Season` and `Playoffs` under `season_types`, so a full
  rebuild can populate regime-aware historical rows.
- `make build-features` creates the processed game table and team-game logs used by training and inference.
- `make train-baseline` trains the Elo and tabular benchmark models.
- `make train-model` trains the fusion model and ensemble artifacts.
- `make evaluate` writes the evaluation outputs used by the dashboard.

To fetch only one population during development, pass an explicit season type:

```bash
python -m src.data.fetch_games --season-type "Regular Season"
python -m src.data.fetch_games --season-type Playoffs
python -m src.data.fetch_player_logs --season-type Playoffs
```

Those commands rewrite the processed tables for the selected fetch scope, so use the
full default commands before training or promotion-gate evaluation.

For the active M1 player-and-lineup foundation work, there are also standalone builders for the new
player-aware tables:

```bash
python -m src.data.fetch_player_logs
python -m src.features.player_value_features
python -m src.features.projected_availability
python -m src.features.lineup_features
make build-m1-features
```

These write:

- `data/processed/player_game_logs/player_game_logs.parquet`
- `data/processed/player_value_features/player_value_features.parquet`
- `data/processed/projected_availability/projected_availability.parquet`
- `data/processed/projected_availability/unresolved_injury_entities.parquet`
- `data/processed/lineup_features/lineup_features.parquet`
- `data/processed/matchup_rows/matchup_dataset_enriched.parquet`

They are not yet part of the deployed model's main inference stack, but they are the active
foundation for the next-generation roadmap.

To benchmark the enriched representation itself, run:

```bash
python -m src.models.run_enriched_experiments
```

That experiment suite writes:

- `docs/experiments/m1_enriched_matchup_results.json`
- `docs/experiments/m1_enriched_matchup_results.md`

It compares the legacy matchup rows against the enriched M1 variants, includes feature-family
ablations, and will also score optional `LightGBM` / `CatBoost` leaderboard entries when those
libraries are installed in the environment.

To compare those enriched winners against the shipped neural + ensemble production stack without
retraining everything, run:

```bash
python -m src.models.run_production_showdown
```

That writes:

- `docs/experiments/production_stack_showdown.json`
- `docs/experiments/production_stack_showdown.md`

To train the next-generation ensemble that adds enriched CatBoost and LightGBM probabilities on
top of the current `neural + xgboost + elo` stack, run:

```bash
python -m src.models.run_nextgen_ensemble
```

That writes tracked reports to:

- `docs/experiments/nextgen_ensemble_results.json`
- `docs/experiments/nextgen_ensemble_results.md`

Generated next-gen model artifacts are stored under `models/ensembles_nextgen/` and are ignored by
git.

Before promoting that candidate into live inference, run the promotion gate:

```bash
python -m src.models.run_nextgen_validation
```

That writes:

- `docs/experiments/nextgen_promotion_gate.json`
- `docs/experiments/nextgen_promotion_gate.md`

The gate scores production vs next-gen across regular-season, per-season, schedule-stress, context-confidence,
playoff, and missing-player slices. The rebuilt evaluation now includes playoff coverage; promotion remains
blocked until the historical evaluation includes real nonzero missing-player impact rows.

If `games.parquet` changes after a new fetch, the enriched experiment runner checks cached
matchup/player artifacts and rebuilds stale regular-season-only caches automatically.

### 3. Generate Local Predictions

To create a local prediction file for the current weekly slate:

```bash
make predict-today
```

Or run the script directly for a specific date:

```bash
python -m src.app.predict_today --date 2026-05-16
```

This writes to:

```text
predictions/daily/YYYY-MM-DD.json
```

### 4. Run a Historical Backtest

```bash
python -m src.app.run_backtest --start-date 2024-01-15 --end-date 2024-01-16
```

This writes to:

```text
predictions/historical_backtests/backtest_<start>_to_<end>.json
```

### 5. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard reads from:

- local forecasts in `predictions/daily/`
- tracked deployment forecasts in `published/daily/`
- publish status in `published/manifest.json`
- saved reports in `predictions/historical_backtests/`
- processed tables in `data/processed/`
- saved evaluation artifacts in `models/`
- bundled fallback data in `src/app/bundled_data/`

Source precedence is:

1. local `predictions/daily/*.json`
2. tracked `published/daily/*.json`
3. bundled fallback data

For Streamlit Cloud deployment, set the main file path to:

```text
streamlit_app.py
```

### 6. Publish a Deployment Forecast

To publish a tracked forecast snapshot for the deployed app:

```bash
python -m src.app.publish_today
```

Or publish a specific date manually:

```bash
python -m src.app.publish_today --date 2024-01-15
```

This command:

- resolves the target date in `America/New_York` by default
- runs inference in a temporary workspace
- attempts to enrich the slate with the latest official injury report snapshot and live team-news context
- writes `published/daily/YYYY-MM-DD.json`
- updates `published/daily/latest.json`
- writes `published/manifest.json`
- leaves the current published slate untouched if publishing fails

Automation is defined in `.github/workflows/publish_daily.yml`, which schedules the publish job daily at `15:05 UTC` and also supports `workflow_dispatch`.

### 7. Launch the API Service

```bash
make serve-api
```

Or run it directly:

```bash
uvicorn src.app.api:app --reload
```

The API serves:

- `GET /health`
- `GET /manifest`
- `GET /forecast/week`
- `GET /forecast/game/{game_id}`
- `GET /metrics`

---

## App Features

The Streamlit app is organized into the following pages:

- `This Week's Games` shows the live forecast window, date-grouped matchups, confidence bands, probability bars, top factors, and live context coverage for each publish.
- `Game Detail` lets you inspect one matchup in depth, including component model outputs and recent team form.
- `Archive` combines local forecasts, published forecasts, and historical backtests into one searchable table.
- `Performance` summarizes model comparison results, rolling validation trends, and ensemble behavior.
- `Calibration` shows how well predicted probabilities line up with actual outcomes, including error by probability bucket.
- `Team Form` highlights recent record, point differential, and net-rating trends for a selected team.
- `Injury Impact` summarizes the current slate's official injury-report coverage when available and falls back honestly when later-week games do not have reports yet.
- `News Sentiment` summarizes current live article coverage and makes it clear when fallback news features are still in use.

The repo also now includes a FastAPI service layer:

- `GET /health` returns service health plus forecast freshness and coverage
- `GET /manifest` exposes the tracked publish manifest used by deployment consumers
- `GET /forecast/week` returns the current weekly slate and date buckets
- `GET /forecast/game/{game_id}` returns one matchup with derived team labels
- `GET /metrics` returns model comparison, calibration, ensemble weights, and rolling validation

The dashboard also shows forecast-source status, including:

- `Published this week`
- `No games scheduled in this forecast window`
- `Showing previous published slate`
- `Showing bundled example slate`

---

## Typical Workflow

For day-to-day use, the simplest flow is:

1. Set up the environment once with `pip install -r requirements.txt`.
2. Build data and train models once with `make fetch-data`, `make build-features`, `make train-baseline`, `make train-model`, and `make evaluate`.
3. Generate a local slate with `make predict-today`.
4. Publish a deployment-ready slate with `python -m src.app.publish_today`.
5. Optionally run backtests for past date ranges.
6. Launch `streamlit run streamlit_app.py` to explore predictions and diagnostics.
7. Launch `uvicorn src.app.api:app --reload` when you want programmatic access to the same published slate.

---

## Quality Checks

```bash
make test
ruff check src/ tests/
```

---

## Planning And History

For current and future planning, use:

- [docs/next_generation_model_roadmap.md](docs/next_generation_model_roadmap.md) for the active roadmap
- [docs/current_system_implementation_summary.md](docs/current_system_implementation_summary.md) for how the current system works
- [docs/m1_player_lineup_foundation.md](docs/m1_player_lineup_foundation.md) for the first active build milestone
- [docs/m1_tracker.md](docs/m1_tracker.md) for milestone status

Historical implementation history is preserved in:

- [docs/phases/all_phases.md](docs/phases/all_phases.md)
- [docs/nba_game_prediction_project_plan.md](docs/nba_game_prediction_project_plan.md) as a legacy archive

| Phase | Name | Status |
|-------|------|--------|
| 0 | Project Setup | Complete |
| 1 | Historical Data Foundation | Complete |
| 2 | Leakage-Safe Feature Table | Complete |
| 3 | Baseline Models | Complete |
| 4 | Team Sequence Model | Complete |
| 5 | Injury Features | Complete |
| 6 | LLM News/Sentiment Layer | Complete |
| 7 | Full Fusion Model | Complete |
| 8 | Daily Prediction System | Complete |
| 9 | Product Dashboard | Complete |
| 10 | Live Publishing Layer | Complete |
| 11 | Live Context + Playoff Hardening | Complete |
| 12 | API Service Layer | Complete |

**Current data:** 15,412 games across 12 seasons (2014-2026), including 983 playoff games. The current next-gen experiment candidate is `nextgen_full / raw` at 0.6161 log loss on the playoff-aware held-out split, with promotion still blocked until real missing-player validation coverage exists.

---

## Architecture

```text
Historical team sequences -> Team Performance Encoder (GRU)
Structured injury reports -> Injury Impact Encoder (MLP)
Structured live article signals -> News/Spirit Encoder (MLP)
Schedule context (rest, B2B) -> Context Encoder (MLP)
                               -> Matchup Fusion Network
                               -> Calibration Layer
                               -> P(home team wins)
```

---

## Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Data source | `nba_api` behind `DataProvider` interface | Swap-ready if the API breaks |
| Team identity | Anonymous indices (0-29) | Prevent name memorization |
| Sequence model | GRU first, then TCN/Transformer | Simpler models first |
| News backfill | None pre-2023-24 | No reliable historical article corpus |
| LLM / sentiment path | Structured article scoring today, LLM-ready schema preserved | Live deployment stays deterministic while keeping the schema extensible |
| LLM schema | 7 core fields (MVP) | Reduce noise, expand later |
| Normalization | Per-season StandardScaler | Accounts for era changes |
| Validation | Time-based + rolling splits | Simulates real forecasting |
| Playoffs | Included in historical evaluation | Live board filters to the next game per series; promotion gate tracks playoff slices |
| Ensemble | Logistic regression meta-model | Simple, interpretable |

---

## Metrics

The model is evaluated as a **probability model**, not just a classifier:

| Metric | Why |
|--------|-----|
| **Log Loss** | Primary metric - penalizes confident wrong predictions |
| **Brier Score** | Proper scoring rule for probability quality |
| **Calibration Error** | Do predicted 60% games actually win about 60%? |
| **Accuracy** | Simple but not sufficient alone |
| **ROC-AUC** | Discrimination ability |

Target performance: **about 62-66% accuracy with well-calibrated probabilities**.

> A well-calibrated 62% model is better than an overconfident 65% model.

---

## Testing

Every phase ends with comprehensive tests. Run the full suite:

```bash
make test
# or: pytest tests/ -v --tb=short
```

Key test categories:

- **Leakage tests** - verify no future data is used
- **Schema tests** - validate data shapes and types
- **Model tests** - check training, gradients, outputs
- **Prediction tests** - verify output format and reasonableness
- **Integration tests** - end-to-end pipeline checks

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- [nba_api](https://github.com/swar/nba_api) for NBA.com data access
- Basketball Reference for validation data
- OpenAI / Anthropic / Google for LLM APIs
