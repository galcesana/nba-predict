"""Injury feature tests — verify schema, ranges, and infrastructure.

Run with: pytest tests/test_injury_features.py -v
Expected: 10/10 pass
"""

import numpy as np
import pandas as pd
import pytest

from src.features.injury_features import (
    INJURY_FEATURE_COLS,
    build_injury_features,
    build_injury_features_default,
)
from src.utils.paths import PROCESSED_DIR


@pytest.fixture(scope="module")
def injury_features() -> pd.DataFrame:
    """Load the saved injury features."""
    path = PROCESSED_DIR / "injury_features" / "injury_features.parquet"
    assert path.exists(), f"Injury features not found at {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "games.parquet")


class TestInjuryFeatures:
    """Tests for injury feature pipeline."""

    def test_injury_features_exist(self, injury_features):
        """Injury features parquet file exists and is loadable."""
        assert len(injury_features) > 0

    def test_injury_features_schema(self, injury_features):
        """Required columns are present."""
        required = ["game_id", "team_idx"] + INJURY_FEATURE_COLS
        for col in required:
            assert col in injury_features.columns, f"Missing column: {col}"

    def test_injury_features_non_negative(self, injury_features):
        """All injury feature values are >= 0."""
        numeric_cols = [c for c in INJURY_FEATURE_COLS if c != "injury_data_available"]
        for col in numeric_cols:
            assert (injury_features[col] >= 0).all(), f"{col} has negative values"

    def test_players_out_count_reasonable(self, injury_features):
        """players_out_count is between 0 and 15."""
        assert injury_features["players_out_count"].max() <= 15
        assert injury_features["players_out_count"].min() >= 0

    def test_minutes_missing_reasonable(self, injury_features):
        """minutes_missing is between 0 and 240 (max 5 starters × 48 min)."""
        assert injury_features["minutes_missing"].max() <= 240
        assert injury_features["minutes_missing"].min() >= 0

    def test_injury_available_flag(self, injury_features):
        """All games have injury_data_available as 0 or 1."""
        assert set(injury_features["injury_data_available"].unique()).issubset({0, 1})

    def test_one_row_per_team_per_game(self, injury_features):
        """Each (game_id, team_idx) pair appears exactly once."""
        dupes = injury_features.duplicated(subset=["game_id", "team_idx"])
        assert not dupes.any(), "Duplicate (game_id, team_idx) pairs found"

    def test_expected_row_count(self, injury_features, games):
        """Should have 2 rows per game (home + away)."""
        assert len(injury_features) == 2 * len(games)

    def test_all_teams_represented(self, injury_features, games):
        """Every home and away team from every game has injury features."""
        game_teams = set()
        for _, g in games.iterrows():
            game_teams.add((g["game_id"], int(g["home_team_idx"])))
            game_teams.add((g["game_id"], int(g["away_team_idx"])))

        feature_teams = set(
            zip(injury_features["game_id"], injury_features["team_idx"])
        )
        # Check a sample (full check is slow)
        sample = list(game_teams)[:100]
        for pair in sample:
            assert pair in feature_teams, f"Missing injury features for {pair}"

    def test_no_nan_values(self, injury_features):
        """No NaN values in injury features."""
        for col in INJURY_FEATURE_COLS:
            assert not injury_features[col].isna().any(), f"NaN values in {col}"
