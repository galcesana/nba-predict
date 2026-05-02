# CLAUDE.md

> Context file for AI assistants working on this project.

## Project Summary

NBA game outcome prediction system that outputs **calibrated win probabilities** using historical performance sequences, injury impact, LLM-extracted news sentiment, and schedule context. This is a sports analytics project, not a betting project.

## Current State

- **Phase 0 complete** — scaffolding, configs, DataProvider, 21 tests passing
- **Phase 1 complete** — 14,429 games fetched across 12 seasons (2014-25), 33 tests passing
- **Phase 2 next** — leakage-safe rolling features and matchup dataset
- See `docs/phases/all_phases.md` for the full phase tracker
- See `docs/nba_game_prediction_project_plan.md` for the comprehensive project plan

## Project Structure

```
nba-predict/
├── configs/                      # YAML configs (data_sources, model, features)
├── data/
│   ├── raw/                      # Cached API responses (Parquet) — gitignored
│   ├── interim/                  # Cleaned intermediate data — gitignored
│   ├── processed/                # Feature tables — gitignored
│   └── mappings/                 # team_to_idx.json (tracked in git)
├── src/
│   ├── data/providers/           # DataProvider ABC + NbaApiProvider
│   ├── anonymization/            # Team/player → anonymous ID mapping
│   ├── nlp/                      # LLM sentiment extraction pipeline
│   ├── features/                 # Feature engineering (rolling, schedule, injury, news)
│   ├── models/                   # Elo, tabular, neural, ensemble, calibration
│   ├── app/                      # Streamlit dashboard
│   └── utils/                    # paths.py, logging.py
├── models/                       # Saved model artifacts — gitignored
├── predictions/                  # Output JSONs — gitignored
├── tests/                        # pytest test suite
├── docs/phases/                  # Phase implementation guides (10 phases)
└── notebooks/                    # Exploratory analysis
```

## Critical Rules

1. **No data leakage** — For a game on date D, only use data from before D. This is the most important rule. All rolling features must use `.shift(1)`. Never include the target game's stats in its features.
2. **Anonymous team IDs** — Internally use indices 0–29 (see `data/mappings/team_to_idx.json`). Team names are only for data collection and UI display.

## Mandatory Workflow

**After completing any phase, you MUST update ALL tracking docs before moving on:**

1. `docs/phases/all_phases.md` — update the Status column for the completed phase
2. `docs/phases/phase_XX_*.md` — check off all deliverables, fill in Notes & Learnings
3. `CLAUDE.md` — update the "Current State" section
4. Commit and push the tracking updates

**This is not optional. Do not start the next phase until tracking is updated.**
3. **LLM is a feature extractor** — The LLM extracts structured sentiment scores from news articles. It does NOT predict game winners.
4. **Calibrated probabilities** — The model outputs P(home_win), not hard predictions. Calibration matters more than accuracy.
5. **Reproducibility** — Pin seeds (42), cache raw data as Parquet, log git commit + config with every experiment.

## Tech Stack

- Python 3.10+, pandas, numpy, scikit-learn, PyTorch
- XGBoost/LightGBM for baselines
- `nba_api` for data (behind `DataProvider` interface)
- GPT-4o-mini (temp=0) for sentiment extraction
- Streamlit for dashboard
- pytest for testing

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data source abstraction | `DataProvider` ABC | `nba_api` endpoints are fragile; swap-ready |
| Historical news backfill | None pre-2023-24 | Zero vector + `news_available=0` flag |
| Sequence padding | Zero-pad + binary mask | Handles season openers and early games |
| LLM schema | 7 core fields MVP | Reduce noise; expand after correlation analysis |
| Feature normalization | Per-season StandardScaler | Fit on training data only |
| Validation | Time-based + rolling splits | Never random split — simulates real forecasting |
| Scope | Regular season only (V1) | Playoffs deferred to Phase 10+ |
| Ensemble meta-model | Logistic regression | Simple, interpretable, hard to overfit |

## Conventions

- **Configs** are YAML in `configs/`. Load with `yaml.safe_load()`.
- **Paths** use `src/utils/paths.py` constants — never hardcode paths.
- **Logging** via `src/utils/logging.py` — call `setup_logging()` in entry points.
- **Tests** live in `tests/`, one file per phase: `test_project_structure.py`, `test_data_foundation.py`, etc.
- **Data** is gitignored. Raw API responses cached as Parquet. Mappings are tracked.
- **Models** saved to `models/{baselines,neural,calibrators,ensembles}/`.

## Running

```bash
pip install -r requirements.txt
pytest tests/ -v              # run tests
make fetch-data               # download NBA data
make build-features           # build feature tables
make train-baseline           # train Elo + XGBoost
make train-model              # train neural model
make predict-today            # generate today's predictions
```

## Architecture (Target)

```
Team game sequences → GRU encoder ─────────────┐
Injury vectors      → Injury MLP  ─────────────┤
News sentiment      → News MLP    ─────────────┤ → Fusion MLP → Calibration → P(home_win)
Schedule context    → Context MLP ─────────────┘
```

Home and away teams use **shared encoders** (same weights). Matchup is modeled via concatenation + difference + element-wise product of encoded states.
