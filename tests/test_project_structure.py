"""Phase 0 verification tests — Project structure, configs, and imports.

Run with: pytest tests/test_project_structure.py -v
Expected: 7/7 pass
"""

import json
from pathlib import Path

import yaml


def _project_root() -> Path:
    """Get project root (one level up from tests/)."""
    return Path(__file__).resolve().parent.parent


class TestDirectoryStructure:
    """Verify all required directories exist."""

    def test_directory_structure(self):
        """All required directories exist."""
        root = _project_root()
        required_dirs = [
            "data",
            "data/raw",
            "data/raw/nba_api",
            "data/raw/injuries",
            "data/raw/news",
            "data/interim",
            "data/interim/cleaned_games",
            "data/interim/cleaned_boxscores",
            "data/interim/anonymized_articles",
            "data/processed",
            "data/processed/team_game_logs",
            "data/processed/matchup_rows",
            "data/processed/injury_features",
            "data/processed/news_features",
            "data/mappings",
            "configs",
            "src",
            "src/data",
            "src/data/providers",
            "src/anonymization",
            "src/nlp",
            "src/features",
            "src/models",
            "src/app",
            "src/utils",
            "models",
            "models/baselines",
            "models/neural",
            "models/calibrators",
            "models/ensembles",
            "predictions",
            "predictions/daily",
            "predictions/historical_backtests",
            "tests",
            "docs",
            "notebooks",
        ]
        for d in required_dirs:
            path = root / d
            assert path.exists(), f"Missing directory: {d}"
            assert path.is_dir(), f"Not a directory: {d}"


class TestTeamMapping:
    """Verify team_to_idx.json is valid."""

    def test_team_mapping_exists(self):
        """team_to_idx.json exists."""
        path = _project_root() / "data" / "mappings" / "team_to_idx.json"
        assert path.exists(), "team_to_idx.json not found"

    def test_team_mapping_has_30_teams(self):
        """Mapping has exactly 30 NBA teams."""
        path = _project_root() / "data" / "mappings" / "team_to_idx.json"
        with open(path) as f:
            mapping = json.load(f)
        assert len(mapping) == 30, f"Expected 30 teams, got {len(mapping)}"

    def test_team_mapping_indices_0_to_29(self):
        """Indices are 0-29 with no gaps or duplicates."""
        path = _project_root() / "data" / "mappings" / "team_to_idx.json"
        with open(path) as f:
            mapping = json.load(f)
        indices = sorted(mapping.values())
        assert indices == list(range(30)), f"Indices must be 0-29, got {indices}"


class TestDataProviderInterface:
    """Verify DataProvider ABC and NbaApiProvider are importable and correct."""

    def test_data_provider_importable(self):
        """DataProvider ABC can be imported."""
        from src.data.providers.base import DataProvider
        assert DataProvider is not None

    def test_nba_api_provider_importable(self):
        """NbaApiProvider can be imported."""
        from src.data.providers.nba_api_provider import NbaApiProvider
        assert NbaApiProvider is not None

    def test_nba_api_provider_is_subclass(self):
        """NbaApiProvider inherits from DataProvider."""
        from src.data.providers.base import DataProvider
        from src.data.providers.nba_api_provider import NbaApiProvider
        assert issubclass(NbaApiProvider, DataProvider)

    def test_nba_api_provider_has_all_methods(self):
        """NbaApiProvider has all required abstract methods."""
        from src.data.providers.nba_api_provider import NbaApiProvider
        required_methods = [
            "fetch_games",
            "fetch_team_game_logs",
            "fetch_box_scores",
            "fetch_player_info",
        ]
        for method in required_methods:
            assert hasattr(NbaApiProvider, method), f"Missing method: {method}"


class TestConfigs:
    """Verify all config files load correctly."""

    def test_data_sources_config(self):
        """data_sources.yaml parses and has required keys."""
        path = _project_root() / "configs" / "data_sources.yaml"
        assert path.exists(), "data_sources.yaml not found"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "provider" in data
        assert "seasons" in data
        assert "rate_limit" in data

    def test_model_config(self):
        """model_config.yaml parses and has required keys."""
        path = _project_root() / "configs" / "model_config.yaml"
        assert path.exists(), "model_config.yaml not found"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "random_seed" in data
        assert "sequence" in data
        assert "elo" in data
        assert "neural" in data

    def test_feature_config(self):
        """feature_config.yaml parses and has required keys."""
        path = _project_root() / "configs" / "feature_config.yaml"
        assert path.exists(), "feature_config.yaml not found"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "rolling_windows" in data
        assert "mvp_features" in data
        assert "normalization" in data

    def test_mvp_features_count(self):
        """MVP feature list has at least 15 features."""
        path = _project_root() / "configs" / "feature_config.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        features = data["mvp_features"]
        assert len(features) >= 15, f"Expected ≥15 MVP features, got {len(features)}"


class TestUtilities:
    """Verify utility modules work correctly."""

    def test_paths_importable(self):
        """Path constants can be imported."""
        from src.utils.paths import (
            CONFIGS_DIR,
            DATA_DIR,
            MODELS_DIR,
            PREDICTIONS_DIR,
            PROJECT_ROOT,
            PUBLISHED_DIR,
        )
        assert PROJECT_ROOT is not None
        assert DATA_DIR is not None
        assert CONFIGS_DIR is not None
        assert MODELS_DIR is not None
        assert PREDICTIONS_DIR is not None
        assert PUBLISHED_DIR == PROJECT_ROOT / "published"

    def test_paths_resolve_to_real_dirs(self):
        """Path constants point to existing directories."""
        from src.utils.paths import CONFIGS_DIR, DATA_DIR, PROJECT_ROOT
        assert PROJECT_ROOT.exists(), f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"
        assert DATA_DIR.exists(), f"DATA_DIR does not exist: {DATA_DIR}"
        assert CONFIGS_DIR.exists(), f"CONFIGS_DIR does not exist: {CONFIGS_DIR}"

    def test_logging_setup(self):
        """setup_logging() runs without error."""
        from src.utils.logging import setup_logging
        setup_logging()  # Should not raise


class TestProjectFiles:
    """Verify essential project files exist."""

    def test_makefile_exists(self):
        """Makefile exists at project root."""
        assert (_project_root() / "Makefile").exists()

    def test_gitignore_exists(self):
        """.gitignore exists at project root."""
        assert (_project_root() / ".gitignore").exists()

    def test_pyproject_toml_exists(self):
        """pyproject.toml exists at project root."""
        assert (_project_root() / "pyproject.toml").exists()

    def test_requirements_txt_exists(self):
        """requirements.txt exists at project root."""
        assert (_project_root() / "requirements.txt").exists()

    def test_requirements_pin_artifact_sklearn_runtime(self):
        """Production artifacts should load under the sklearn version they were saved with."""
        text = (_project_root() / "requirements.txt").read_text(encoding="utf-8")

        assert 'scikit-learn==1.6.0; python_version < "3.14"' in text
        assert 'scikit-learn>=1.8,<1.9; python_version >= "3.14"' in text

    def test_readme_exists(self):
        """README.md exists at project root."""
        assert (_project_root() / "README.md").exists()

    def test_env_example_exists(self):
        """.env.example exists at project root."""
        assert (_project_root() / ".env.example").exists()

    def test_publish_workflow_runs_at_israel_morning(self):
        """The deployed publisher should run once per day at 07:00 Israel time."""
        workflow = _project_root() / ".github" / "workflows" / "publish_daily.yml"
        text = workflow.read_text(encoding="utf-8")

        assert 'cron: "0 4 * * *"' in text
        assert 'cron: "0 5 * * *"' in text
        assert 'ZoneInfo("Asia/Jerusalem")' in text
        assert "local_time.hour == 7" in text
        assert "python -m src.app.publish_today --nextgen-shadow" in text

    def test_ci_workflow_runs_lint_and_tests(self):
        """CI should enforce lint plus clean-checkout tests."""
        workflow = _project_root() / ".github" / "workflows" / "ci.yml"
        text = workflow.read_text(encoding="utf-8")

        assert "python -m ruff check ." in text
        assert "python -m pytest tests -q" in text
        assert "--ignore=tests/test_baselines.py" in text
        assert "--ignore=tests/test_data_foundation.py" in text
        assert "--ignore=tests/test_features.py" in text
        assert "--ignore=tests/test_no_leakage.py" in text
        assert "pull_request:" in text
        assert "workflow_dispatch:" in text
        assert "published/**" in text
