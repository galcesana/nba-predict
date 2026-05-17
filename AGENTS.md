# AGENTS.md

> Context file for AI assistants working on this project.

## Project Summary

NBA game outcome prediction system that outputs **calibrated win probabilities** using historical performance sequences, injury impact, structured news sentiment, and schedule context. This is a sports analytics project, not a betting project.

## Current State

- **Phase 0 complete** - scaffolding, configs, DataProvider, 21 tests passing
- **Phase 1 complete** - 14,429 games fetched across 12 seasons (2014-26), 33 tests passing
- **Phase 2 complete** - leakage-safe rolling features + schedule features + matchup dataset (14,429 rows x 110 cols), 47 tests passing
- **Phase 3 complete** - Elo, logistic regression, XGBoost baselines trained + evaluated (best: XGBoost 64.5% acc, 0.622 log loss), 59 tests passing
- **Phase 4 complete** - GRU sequence model trained (65.1% acc, 0.619 log loss, beats all baselines), 73 tests passing
- **Phase 5 complete** - injury proxy features (variance-based) implemented and tested, 83 tests passing
- **Phase 6 complete** - news/sentiment proxy features (zero-vectors) implemented and tested, 93 tests passing
- **Phase 7 complete** - full 4-stream fusion model trained (log loss 0.6169) and logistic ensemble trained (65.6% acc, 0.615 log loss), 105 tests passing
- **Phase 8 complete** - Daily Prediction System implemented, verified on a historical daily slate and backtest window, 111 tests passing
- **Phase 9 complete** - Streamlit dashboard implemented and browser-verified, 121 tests passing
- **Phase 10 complete** - live publishing layer implemented with tracked published forecasts, dashboard source precedence, and GitHub Actions automation, 129 tests passing
- **Phase 11 complete** - live context ingestion added for official injury reports and team news, weekly playoff publishing hardened, 140 tests passing
- **Phase 12 complete** - FastAPI service layer implemented for health, manifest, weekly forecast, game detail, and metrics access, 148 tests passing
- **Active milestone** - M1 Player + Lineup Foundation in progress; player mapping, season roster metadata, historical player-log ingestion, player-value features, projected availability, lineup/rotation features, enriched matchup rows, next-gen ensemble experiments, a promotion gate, and playoff-capable ingestion are added. Current gate status: playoff coverage passes; promotion is blocked until nonzero missing-player validation coverage exists.
- See `docs/phases/all_phases.md` for the full phase tracker
- See `docs/current_system_implementation_summary.md` for the current implementation reference
- See `docs/next_generation_model_roadmap.md` for the active forward roadmap
- See `docs/m1_player_lineup_foundation.md` for the concrete next implementation target
- `docs/nba_game_prediction_project_plan.md` is retained only as a legacy archive

## Project Structure

```text
nba-predict/
|- configs/                      # YAML configs (data_sources, model, features)
|- data/
|  |- raw/                       # Cached API responses (Parquet) - gitignored
|  |- interim/                   # Cleaned intermediate data - gitignored
|  |- processed/                 # Feature tables (minimal inference bundle tracked)
|  `- mappings/                  # team_to_idx.json (tracked in git)
|- published/                    # Tracked deployment forecast JSONs
|- src/
|  |- data/providers/            # DataProvider ABC + NbaApiProvider
|  |- anonymization/             # Team/player -> anonymous ID mapping
|  |- nlp/                       # Structured article scoring + sentiment pipeline
|  |- features/                  # Feature engineering (rolling, schedule, injury, news)
|  |- models/                    # Elo, tabular, neural, ensemble, calibration
|  |- app/                       # Daily prediction, publishing, Streamlit dashboard, FastAPI service
|  `- utils/                     # paths.py, logging.py
|- models/                       # Saved model artifacts (minimal inference bundle tracked)
|- predictions/                  # Local output JSONs - gitignored
|- tests/                        # pytest test suite
|- docs/phases/                  # Phase implementation guides (13 phases)
`- notebooks/                    # Exploratory analysis
```

## Critical Rules

1. **No data leakage** - For a game on date D, only use data from before D. This is the most important rule. All rolling features must use `.shift(1)`. Never include the target game's stats in its features.
2. **Anonymous team IDs** - Internally use indices 0-29 (see `data/mappings/team_to_idx.json`). Team names are only for data collection and UI display.
3. **LLM is a feature extractor** - The LLM extracts structured sentiment scores from news articles. It does NOT predict game winners.
4. **Calibrated probabilities** - The model outputs `P(home_win)`, not hard predictions. Calibration matters more than accuracy.
5. **Reproducibility** - Pin seeds (42), cache raw data as Parquet, log git commit + config with every experiment.

## Mandatory Workflow

**After completing any phase, you MUST update ALL tracking docs before moving on:**

1. `docs/phases/all_phases.md` - update the Status column for the completed phase
2. `docs/phases/phase_XX_*.md` - check off all deliverables, fill in Notes & Learnings
3. `AGENTS.md` - update the "Current State" section
4. Commit and push the tracking updates

**This is not optional. Do not start the next phase until tracking is updated.**

## Tech Stack

- Python 3.10+, pandas, numpy, scikit-learn, PyTorch
- XGBoost/LightGBM for baselines
- `nba_api` for data (behind `DataProvider` interface)
- GPT-4o-mini (temp=0) for sentiment extraction
- Streamlit for dashboard
- FastAPI + Uvicorn for service delivery
- pytest for testing

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data source abstraction | `DataProvider` ABC | `nba_api` endpoints are fragile; swap-ready |
| Historical news backfill | None pre-2023-24 | Zero vector + `news_available=0` flag |
| Sequence padding | Zero-pad + binary mask | Handles season openers and early games |
| LLM schema | 7 core fields MVP | Reduce noise; expand after correlation analysis |
| Feature normalization | Per-season StandardScaler | Fit on training data only |
| Validation | Time-based + rolling splits | Never random split - simulates real forecasting |
| Scope | Regular season + playoff-aware evaluation | Historical playoff rows are included; live board filters playoff series conservatively |
| Ensemble meta-model | Logistic regression | Simple, interpretable, hard to overfit |

## Conventions

- **Configs** are YAML in `configs/`. Load with `yaml.safe_load()`.
- **Paths** use `src/utils/paths.py` constants - never hardcode paths.
- **Logging** via `src/utils/logging.py` - call `setup_logging()` in entry points.
- **Tests** live in `tests/`, one file per phase: `test_project_structure.py`, `test_data_foundation.py`, etc.
- **Data** is mostly gitignored. Raw API responses stay cached locally; the minimal Phase 10 inference bundle is tracked.
- **Models** save to `models/{baselines,neural,calibrators,ensembles}/`; only the minimal inference bundle is tracked.

## Running

```bash
pip install -r requirements.txt
pytest tests/ -v                 # run tests
make fetch-data                  # download NBA data
make build-features              # build feature tables
make train-baseline              # train Elo + XGBoost
make train-model                 # train neural model
make predict-today               # generate today's predictions
python -m src.app.publish_today  # publish deployment forecast JSONs
streamlit run streamlit_app.py   # launch dashboard
make serve-api                   # launch FastAPI service
python -m src.data.fetch_player_logs       # build player-game logs for M1
python -m src.features.player_value_features  # build pregame player-value rows
python -m src.features.projected_availability  # build projected availability rows
python -m src.features.lineup_features         # build lineup/rotation feature rows
make build-m1-features                         # refresh the full M1 feature stack
python -m src.models.run_enriched_experiments  # benchmark enriched feature families
python -m src.models.run_nextgen_ensemble      # train expanded next-gen ensemble
python -m src.models.run_nextgen_validation    # run promotion gate before live promotion
```

`configs/data_sources.yaml` includes `Regular Season` and `Playoffs`; the rebuilt promotion gate now
includes held-out playoff rows.

## Architecture (Target)

```text
Team game sequences -> GRU encoder -----------+
Injury vectors      -> Injury MLP ------------|
News sentiment      -> News MLP --------------| -> Fusion MLP -> Calibration -> P(home_win)
Schedule context    -> Context MLP -----------+
```

Home and away teams use **shared encoders** (same weights). Matchup is modeled via concatenation + difference + element-wise product of encoded states.
