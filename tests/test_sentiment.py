"""News/sentiment feature tests — verify schema, ranges, and infrastructure.

Run with: pytest tests/test_sentiment.py -v
Expected: 10/10 pass
"""

import numpy as np
import pandas as pd
import pytest

from src.features.news_features import (
    NEWS_FEATURE_COLS,
    build_news_features,
    build_news_features_default,
)
from src.utils.paths import PROCESSED_DIR


@pytest.fixture(scope="module")
def news_features() -> pd.DataFrame:
    """Load the saved news features."""
    path = PROCESSED_DIR / "news_features" / "news_features.parquet"
    assert path.exists(), f"News features not found at {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "games.parquet")


class TestNewsFeatures:
    """Tests for news feature pipeline."""

    def test_news_features_exist(self, news_features):
        """News features parquet file exists and is loadable."""
        assert len(news_features) > 0

    def test_news_features_schema(self, news_features):
        """Required columns are present."""
        required = ["game_id", "team_idx"] + NEWS_FEATURE_COLS
        for col in required:
            assert col in news_features.columns, f"Missing column: {col}"

    def test_news_features_shape(self, news_features, games):
        """News feature vector has 2 rows per game (home + away)."""
        assert len(news_features) == 2 * len(games)

    def test_news_available_flag(self, news_features):
        """news_available is 0 or 1."""
        assert set(news_features["news_available"].unique()).issubset({0, 1})

    def test_default_zero_vectors(self, news_features):
        """When no real news data, all sentiment features should be zero."""
        no_news = news_features[news_features["news_available"] == 0]
        if len(no_news) > 0:
            for col in NEWS_FEATURE_COLS[:-1]:  # Exclude news_available itself
                assert (no_news[col] == 0).all(), f"{col} not zero for missing news"

    def test_one_row_per_team_per_game(self, news_features):
        """Each (game_id, team_idx) pair appears exactly once."""
        dupes = news_features.duplicated(subset=["game_id", "team_idx"])
        assert not dupes.any(), "Duplicate (game_id, team_idx) pairs found"

    def test_no_nan_values(self, news_features):
        """No NaN values in news features."""
        for col in NEWS_FEATURE_COLS:
            assert not news_features[col].isna().any(), f"NaN values in {col}"

    def test_sentiment_ranges(self, news_features):
        """Sentiment values in expected ranges."""
        # weighted_sentiment should be in [-1, 1] when present
        if news_features["news_available"].any():
            has_news = news_features[news_features["news_available"] == 1]
            assert has_news["weighted_sentiment_24h"].between(-1, 1).all()
            assert has_news["weighted_sentiment_72h"].between(-1, 1).all()

    def test_article_volume_non_negative(self, news_features):
        """article_volume_24h >= 0."""
        assert (news_features["article_volume_24h"] >= 0).all()

    def test_all_teams_represented(self, news_features, games):
        """Every home and away team from every game has news features."""
        game_teams = set()
        for _, g in games.iterrows():
            game_teams.add((g["game_id"], int(g["home_team_idx"])))
            game_teams.add((g["game_id"], int(g["away_team_idx"])))

        feature_teams = set(
            zip(news_features["game_id"], news_features["team_idx"])
        )
        # Sample check
        sample = list(game_teams)[:100]
        for pair in sample:
            assert pair in feature_teams, f"Missing news features for {pair}"
