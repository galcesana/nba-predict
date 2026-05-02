# Phase 0 — Project Setup

| Field | Value |
|-------|-------|
| **Size** | S (1–2 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Nothing |
| **Unlocks** | Phase 1 |

---

## Goal

Set up a clean, professional repository: folder structure, configs, environment, `DataProvider` interface, logging, Makefile, and initial tests.

---

## Deliverables Checklist

- [x] Repository initialized with `.gitignore`
- [x] `pyproject.toml` and `requirements.txt`
- [x] `.env.example` with placeholder keys
- [x] Full directory structure created
- [x] `DataProvider` ABC (`src/data/providers/base.py`)
- [x] `NbaApiProvider` skeleton (`src/data/providers/nba_api_provider.py`)
- [x] Config files (data_sources, model_config, feature_config)
- [x] `team_to_idx.json` mapping (all 30 NBA teams → indices 0–29)
- [x] Logging setup (`src/utils/logging.py`)
- [x] Path utilities (`src/utils/paths.py`)
- [x] Makefile with standard targets
- [x] Basic test structure
- [x] `README.md`

---

## Key Implementation Details

### DataProvider interface

```python
# src/data/providers/base.py
from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def fetch_games(self, season: str) -> pd.DataFrame: ...
    @abstractmethod
    def fetch_team_game_logs(self, season: str) -> pd.DataFrame: ...
    @abstractmethod
    def fetch_box_scores(self, game_id: str) -> pd.DataFrame: ...
    @abstractmethod
    def fetch_player_info(self, season: str) -> pd.DataFrame: ...
```

### NbaApiProvider skeleton

```python
# src/data/providers/nba_api_provider.py
# Implements DataProvider with:
#   - 1 req/s rate limiting
#   - Exponential backoff (2s, 4s, 8s) on failures
#   - Max 3 retries
#   - Response schema validation
#   - All methods raise NotImplementedError (implemented in Phase 1)
```

### Team mapping (data/mappings/team_to_idx.json)

```json
{
  "ATL": 0, "BOS": 1, "BKN": 2, "CHA": 3, "CHI": 4,
  "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9,
  "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 14,
  "MIA": 15, "MIL": 16, "MIN": 17, "NOP": 18, "NYK": 19,
  "OKC": 20, "ORL": 21, "PHI": 22, "PHX": 23, "POR": 24,
  "SAC": 25, "SAS": 26, "TOR": 27, "UTA": 28, "WAS": 29
}
```

### Makefile targets

```makefile
fetch-data, build-features, train-baseline, train-model,
evaluate, predict-today, test, lint
```

### Configs to create

- `configs/data_sources.yaml` — provider, seasons, rate limits, cache settings
- `configs/model_config.yaml` — sequence_length, seed, elo params, neural params, ensemble
- `configs/feature_config.yaml` — rolling windows, MVP feature list, normalization

---

## Verification Tests

Run: `pytest tests/test_project_structure.py -v`

```python
# tests/test_project_structure.py

def test_directory_structure():
    """All required directories exist."""
    # Check: data/raw, data/processed, data/mappings, configs/, src/, models/, tests/

def test_team_mapping():
    """team_to_idx.json has 30 teams with unique indices 0-29."""

def test_data_provider_interface():
    """DataProvider ABC and NbaApiProvider importable; NbaApiProvider is subclass."""

def test_configs_load():
    """All 3 YAML config files parse without errors."""

def test_paths_resolve():
    """Path constants (PROJECT_ROOT, DATA_DIR, etc.) resolve to real paths."""

def test_logging_setup():
    """setup_logging() runs without error."""

def test_makefile_exists():
    """Makefile exists at project root."""
```

**Actual: 21/21 pass (expanded beyond original 7).**

---

## Definition of Done

- [x] All 21 verification tests pass
- [x] `git status` is clean (initial commit)
- [x] `pip install -r requirements.txt` succeeds
- [x] `make test` runs without import errors

---

## Notes & Learnings

```
Completed: 2026-05-02
Tests expanded from 7 to 21 (6 test classes covering structure, mappings,
  providers, configs, utilities, and project files).
All 30 NBA teams mapped to indices 0-29.
DataProvider ABC with 4 abstract methods + NbaApiProvider with retry logic.
Makefile has 8 targets: setup, fetch-data, build-features, train-baseline,
  train-model, evaluate, predict-today, test, lint.
```
