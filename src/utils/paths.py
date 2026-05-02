"""Centralized path constants for the project.

All paths are resolved relative to the project root (two levels up from this file).
Import these constants instead of hardcoding paths throughout the codebase.
"""

from pathlib import Path

# Project root: nba-predict/ (two levels up from src/utils/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MAPPINGS_DIR = DATA_DIR / "mappings"

# Configuration
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"

# Prediction outputs
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"
