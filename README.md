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
│   ├── app/                      # Streamlit dashboard + API
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

### Setup

```bash
git clone <repo-url>
cd nba-outcome-model

python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Run Pipeline

```bash
# Fetch historical data
make fetch-data

# Build feature tables
make build-features

# Train baseline models
make train-baseline

# Train neural model
make train-model

# Evaluate all models
make evaluate

# Predict today's games
make predict-today

# Run tests
make test
```

### Launch Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

---

## Implementation Phases

The project is built in 10 phases. See [docs/phases/all_phases.md](docs/phases/all_phases.md) for the full tracker.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Project Setup | ✅ Complete |
| 1 | Historical Data Foundation | ✅ Complete |
| 2 | Leakage-Safe Feature Table | `[ ]` |
| 3 | Baseline Models | `[ ]` |
| 4 | Team Sequence Model | `[ ]` |
| 5 | Injury Features | `[ ]` |
| 6 | LLM News/Sentiment Layer | `[ ]` |
| 7 | Full Fusion Model | `[ ]` |
| 8 | Daily Prediction System | `[ ]` |
| 9 | Product Dashboard | `[ ]` |

**Current data:** 14,429 games across 12 seasons (2014–2026), 28,878 team game logs with 25 stat columns, 33 tests passing.

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
