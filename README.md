# 🏀 NBA Game Outcome Prediction Model

> Predict calibrated win probabilities for NBA games using historical performance, team sequences, injury impact, and LLM-extracted news/team-spirit signals.

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
| **Injury Impact** | Official injury reports | MLP |
| **News / Team Spirit** | LLM-extracted sentiment from news articles | MLP |
| **Schedule Context** | Rest days, back-to-back, travel | MLP |

All streams feed into a **matchup fusion model** → **calibration layer** → P(home win).

> **This is a sports analytics and forecasting project, not a betting project.**

---

## Core Principles

1. **Predict probabilities, not just winners** — a 56% prediction is more honest than "home wins"
2. **No data leakage** — only information available before tip-off is used
3. **Anonymous team IDs** — the model learns from team features, not team names
4. **LLM = feature extractor** — the LLM extracts structured signals, it does not predict winners
5. **Reproducibility** — pinned seeds, versioned data snapshots, logged experiments

---

## Project Structure

```
nba-outcome-model/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
│
├── configs/                      # YAML/JSON configuration
│   ├── data_sources.yaml
│   ├── model_config.yaml
│   └── feature_config.yaml
│
├── data/
│   ├── raw/                      # Cached API responses (Parquet)
│   ├── interim/                  # Cleaned intermediate data
│   ├── processed/                # Feature tables ready for modeling
│   └── mappings/                 # team_to_idx.json, player_to_idx.json
│
├── src/
│   ├── data/                     # Data fetching & cleaning
│   │   └── providers/            # DataProvider ABC + implementations
│   ├── anonymization/            # Team/player name → anonymous ID
│   ├── nlp/                      # LLM sentiment extraction pipeline
│   ├── features/                 # Feature engineering (rolling, schedule, injury, news)
│   ├── models/                   # Elo, tabular, neural, ensemble, calibration
│   ├── app/                      # Daily prediction scripts + Streamlit dashboard
│   └── utils/                    # Logging, paths, validation
│
├── models/                       # Saved model artifacts
├── predictions/                  # Daily + backtest prediction JSONs
├── notebooks/                    # Exploratory analysis
├── tests/                        # Comprehensive test suite
└── docs/                         # Architecture docs + implementation phases
    └── phases/                   # Phase-by-phase implementation guides
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### 1. Set Up the Environment

Clone the repository, create a virtual environment, and install the dependencies:

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

On a fresh checkout, the app needs local data, processed features, and trained model artifacts before it can generate predictions or populate the dashboard.

Run the pipeline in this order:

```bash
# Fetch historical NBA games and box scores
make fetch-data

# Build leakage-safe feature tables
make build-features

# Train Elo and tabular baseline models
make train-baseline

# Train the full neural / fusion model
make train-model

# Evaluate saved models
make evaluate
```

What each step does:

- `make fetch-data` downloads and caches the historical NBA data used by the project.
- `make build-features` creates the processed game table and team-game logs that power both training and inference.
- `make train-baseline` trains the Elo and tabular benchmark models.
- `make train-model` trains the sequence-based fusion model and ensemble artifacts used by the app.
- `make evaluate` writes the evaluation outputs that the dashboard uses for performance views.

### 3. Generate Predictions

To create a prediction file for today's slate:

```bash
make predict-today
```

Or run the script directly for a specific date:

```bash
python -m src.app.predict_today --date 2026-05-16
```

This script:

- fetches the NBA schedule for the target date
- loads only games and team logs from before that date
- runs the prediction pipeline without leakage
- saves the output to `predictions/daily/YYYY-MM-DD.json`

### 4. Run a Historical Backtest

To simulate the live prediction workflow on past dates:

```bash
python -m src.app.run_backtest --start-date 2024-01-15 --end-date 2024-01-16
```

This backtest runs day by day, only using prior information for each date, and saves a report to:

```text
predictions/historical_backtests/backtest_<start>_to_<end>.json
```

### 5. Launch the Dashboard

Start the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

The dashboard reads from:

- the latest file in `predictions/daily/`
- saved reports in `predictions/historical_backtests/`
- processed tables in `data/processed/`
- saved evaluation artifacts in `models/`
- a bundled snapshot in `src/app/bundled_data/` when generated local artifacts are not available

On Streamlit Cloud, the app can fall back to the bundled snapshot so the deployed UI still opens with a representative slate and diagnostics even before a live forecast has been published.

Dashboard screenshots:
- `docs/screenshots/dashboard_today.png`
- `docs/screenshots/dashboard_calibration.png`

### App Features

The Streamlit app is organized into the following pages:

- `Today's Games` shows the latest prediction slate, game confidence, probability bars, and the top factors behind each forecast.
- `Game Detail` lets you inspect one matchup in depth, including component model outputs and recent team form.
- `Archive` combines saved daily predictions and historical backtests into one searchable table.
- `Performance` summarizes model comparison results, rolling validation trends, and ensemble behavior.
- `Calibration` shows how well predicted probabilities line up with actual outcomes, including error by probability bucket.
- `Team Form` highlights recent record, point differential, and net-rating style trends for a selected team.
- `Injury Impact` displays the available injury feature view for each team. At this stage, those inputs are still proxy-based rather than a full live injury feed.
- `News Sentiment` shows the current news feature view for each team and makes it clear when the zero-vector fallback is being used.

### Typical Workflow

For day-to-day use, the simplest flow is:

1. Set up the environment once with `pip install -r requirements.txt`.
2. Build data and train models once with `make fetch-data`, `make build-features`, `make train-baseline`, `make train-model`, and `make evaluate`.
3. Generate a slate with `make predict-today`.
4. Optionally run backtests for past date ranges.
5. Launch `streamlit run streamlit_app.py` to explore predictions and diagnostics.

For Streamlit Cloud deployment, set the main file path to:

```text
streamlit_app.py
```

### Quality Checks

```bash
make test
ruff check src/ tests/
```

---

## Implementation Phases

The project is built in 10 phases. See [docs/phases/all_phases.md](docs/phases/all_phases.md) for the full tracker.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Project Setup | ✅ Complete |
| 1 | Historical Data Foundation | ✅ Complete |
| 2 | Leakage-Safe Feature Table | ✅ Complete |
| 3 | Baseline Models | ✅ Complete |
| 4 | Team Sequence Model | ✅ Complete |
| 5 | Injury Features | ✅ Complete |
| 6 | LLM News/Sentiment Layer | ✅ Complete |
| 7 | Full Fusion Model | ✅ Complete |
| 8 | Daily Prediction System | ג… Complete |
| 9 | Product Dashboard | ג… Complete |

**Current data:** 14,429 games across 12 seasons (2014–2026), Ensemble Model (65.6% acc, 0.615 log loss), daily predictions + backtests + dashboard working, 121 tests passing.

---

## Architecture

```
Historical team sequences ──→ Team Performance Encoder (GRU)
                                        │
Structured injury reports ──→ Injury Impact Encoder (MLP)
                                        │
LLM-extracted news signals ──→ News/Spirit Encoder (MLP)
                                        │
Schedule context (rest, B2B) ──→ Context Encoder (MLP)
                                        │
                                        ▼
                              Matchup Fusion Network
                                        │
                                        ▼
                              Calibration Layer
                                        │
                                        ▼
                              P(home team wins)
```

---

## Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Data source | `nba_api` behind `DataProvider` interface | Swap-ready if API breaks |
| Team identity | Anonymous indices (0-29) | Prevent name memorization |
| Sequence model | GRU first, then TCN/Transformer | Simpler models first |
| News backfill | None pre-2023-24 | No reliable historical articles |
| LLM model | GPT-4o-mini (temp=0) | Cost-efficient, deterministic |
| LLM schema | 7 core fields (MVP) | Reduce noise, expand later |
| Normalization | Per-season StandardScaler | Accounts for era changes |
| Validation | Time-based + rolling splits | Simulates real forecasting |
| Playoffs | Excluded from V1 | Different dynamics, small sample |
| Ensemble | Logistic regression meta-model | Simple, interpretable |

---

## Metrics

The model is evaluated as a **probability model**, not just a classifier:

| Metric | Why |
|--------|-----|
| **Log Loss** | Primary metric — penalizes confident wrong predictions |
| **Brier Score** | Proper scoring rule for probability quality |
| **Calibration Error** | Do predicted 60% games actually win ~60%? |
| **Accuracy** | Simple but not sufficient alone |
| **ROC-AUC** | Discrimination ability |

Target performance: **~62-66% accuracy with well-calibrated probabilities**.

> A well-calibrated 62% model is better than an overconfident 65% model.

---

## Testing

Every phase ends with comprehensive tests. Run the full suite:

```bash
make test
# or: pytest tests/ -v --tb=short
```

Key test categories:
- **Leakage tests** — verify no future data is used
- **Schema tests** — validate data shapes and types
- **Model tests** — check training, gradients, outputs
- **Prediction tests** — verify output format and reasonableness
- **Integration tests** — end-to-end pipeline checks

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- [nba_api](https://github.com/swar/nba_api) for NBA.com data access
- Basketball Reference for validation data
- OpenAI / Anthropic / Google for LLM APIs
