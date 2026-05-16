# NBA Game Outcome Prediction Model

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
| **Injury Impact** | Official injury reports / proxy injury signals | MLP |
| **News / Team Spirit** | LLM-extracted sentiment from news articles | MLP |
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
|  |- app/                       # Daily prediction + publishing scripts + Streamlit dashboard
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

- `make fetch-data` downloads and caches historical NBA data.
- `make build-features` creates the processed game table and team-game logs used by training and inference.
- `make train-baseline` trains the Elo and tabular benchmark models.
- `make train-model` trains the fusion model and ensemble artifacts.
- `make evaluate` writes the evaluation outputs used by the dashboard.

### 3. Generate Local Predictions

To create a local prediction file for today's slate:

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
- writes `published/daily/YYYY-MM-DD.json`
- updates `published/daily/latest.json`
- writes `published/manifest.json`
- leaves the current published slate untouched if publishing fails

Automation is defined in `.github/workflows/publish_daily.yml`, which schedules the publish job daily at `15:05 UTC` and also supports `workflow_dispatch`.

---

## App Features

The Streamlit app is organized into the following pages:

- `This Week's Games` shows the live forecast window, date-grouped matchups, confidence bands, probability bars, and the top factors behind each forecast.
- `Game Detail` lets you inspect one matchup in depth, including component model outputs and recent team form.
- `Archive` combines local forecasts, published forecasts, and historical backtests into one searchable table.
- `Performance` summarizes model comparison results, rolling validation trends, and ensemble behavior.
- `Calibration` shows how well predicted probabilities line up with actual outcomes, including error by probability bucket.
- `Team Form` highlights recent record, point differential, and net-rating trends for a selected team.
- `Injury Impact` displays the available injury feature view for each team. These signals are still proxy-based rather than a full live injury feed.
- `News Sentiment` shows the current news feature view for each team and makes it clear when fallback news features are in use.

The dashboard also shows forecast-source status, including:

- `Published today`
- `No games today`
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

---

## Quality Checks

```bash
make test
ruff check src/ tests/
```

---

## Implementation Phases

The project is built in 11 phases. See [docs/phases/all_phases.md](docs/phases/all_phases.md) for the full tracker.

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

**Current data:** 14,429 games across 12 seasons (2014-2026), Ensemble Model (65.6% acc, 0.615 log loss), daily predictions + weekly published deployment forecasts + dashboard working, 131 tests passing.

---

## Architecture

```text
Historical team sequences -> Team Performance Encoder (GRU)
Structured injury reports -> Injury Impact Encoder (MLP)
LLM-extracted news signals -> News/Spirit Encoder (MLP)
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
