"""Phase 1 verification tests — Historical data foundation.

Run with: pytest tests/test_data_foundation.py -v
Expected: 13/13 pass
"""

import pandas as pd
import pytest

from src.utils.paths import PROCESSED_DIR, RAW_DIR


@pytest.fixture(scope="module")
def games_df() -> pd.DataFrame:
    """Load the processed games table."""
    path = PROCESSED_DIR / "games.parquet"
    assert path.exists(), f"Games table not found: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def team_logs_df() -> pd.DataFrame:
    """Load the processed team game logs."""
    path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
    assert path.exists(), f"Team game logs not found: {path}"
    return pd.read_parquet(path)


class TestGamesTable:
    """Verify the processed games table."""

    def test_games_table_exists(self):
        """Processed games table saved and loadable."""
        path = PROCESSED_DIR / "games.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) > 0

    def test_games_table_schema(self, games_df):
        """Games table has required columns."""
        required = [
            "game_id", "date", "season",
            "home_team_idx", "away_team_idx",
            "home_score", "away_score", "home_win",
        ]
        for col in required:
            assert col in games_df.columns, f"Missing column: {col}"

    def test_games_table_no_nulls(self, games_df):
        """No null values in critical columns."""
        critical = ["game_id", "date", "home_team_idx", "away_team_idx",
                     "home_score", "away_score", "home_win"]
        for col in critical:
            null_count = games_df[col].isna().sum()
            assert null_count == 0, f"Column {col} has {null_count} nulls"

    def test_team_indices_valid(self, games_df):
        """All team_idx values are in range 0-29."""
        for col in ["home_team_idx", "away_team_idx"]:
            assert games_df[col].min() >= 0, f"{col} has values < 0"
            assert games_df[col].max() <= 29, f"{col} has values > 29"

    def test_home_win_binary(self, games_df):
        """home_win column contains only 0 and 1."""
        unique_vals = set(games_df["home_win"].unique())
        assert unique_vals <= {0, 1}, f"home_win has unexpected values: {unique_vals}"

    def test_no_duplicate_games(self, games_df):
        """No duplicate game_id values."""
        dupes = games_df["game_id"].duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate game_ids"

    def test_season_coverage(self, games_df):
        """At least 10 seasons of data present."""
        n_seasons = games_df["season"].nunique()
        assert n_seasons >= 10, f"Only {n_seasons} seasons found, expected >= 10"

    def test_games_per_season(self, games_df):
        """Each full season has roughly 1,230 games (±200 for COVID/lockout)."""
        frame = games_df
        if "season_type" in frame.columns:
            frame = frame[frame["season_type"] == "Regular Season"]
        season_counts = frame.groupby("season").size()
        for season, count in season_counts.items():
            assert 800 <= count <= 1300, (
                f"Season {season} has {count} games, expected 800-1300"
            )


class TestTeamGameLogs:
    """Verify the team game logs."""

    def test_team_game_logs_exist(self):
        """Team game logs table saved and loadable."""
        path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) > 0

    def test_team_game_logs_two_rows_per_game(self, team_logs_df):
        """Each game_id appears exactly twice (home + away)."""
        counts = team_logs_df.groupby("game_id").size()
        bad = counts[counts != 2]
        assert len(bad) == 0, (
            f"{len(bad)} games don't have exactly 2 rows. Examples: {bad.head()}"
        )

    def test_team_game_logs_schema(self, team_logs_df):
        """Team game logs have all required stat columns."""
        required = [
            "game_id", "date", "season", "team_idx", "opponent_team_idx",
            "is_home", "won", "points_for", "points_against", "point_diff",
            "ts_pct", "efg_pct", "turnover_pct",
            "off_rebound_pct", "def_rebound_pct",
            "free_throw_rate", "assist_pct",
        ]
        for col in required:
            assert col in team_logs_df.columns, f"Missing column: {col}"


class TestRawCache:
    """Verify raw data caching."""

    def test_raw_cache_exists(self):
        """Parquet cache files exist in data/raw/nba_api/."""
        cache_dir = RAW_DIR / "nba_api"
        parquet_files = list(cache_dir.glob("*.parquet"))
        assert len(parquet_files) >= 10, (
            f"Expected >= 10 cached parquet files, found {len(parquet_files)}"
        )
