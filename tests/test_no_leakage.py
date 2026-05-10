"""Leakage prevention tests — the most critical tests in the project.

Run with: pytest tests/test_no_leakage.py -v
Expected: 5/5 pass
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.paths import PROCESSED_DIR


@pytest.fixture(scope="module")
def matchup_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet"
    assert path.exists(), f"Matchup dataset not found: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def team_logs() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def rolling_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_rolling_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Compute on the fly if not saved separately
    from src.features.rolling_features import compute_rolling_features
    team_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    return compute_rolling_features(team_logs)


class TestNoLeakage:
    """Verify that no future data leaks into features."""

    def test_no_future_games_used(self, team_logs, rolling_features):
        """For 100 random rows, verify rolling features use only prior data.

        Strategy: for each sampled row, check that the season_win_pct
        (expanding mean of 'won') matches manual computation from prior games.
        """
        rng = np.random.RandomState(42)
        # Only test rows with enough history (game 5+)
        eligible = rolling_features[rolling_features["season_games_played"] >= 5].copy()
        sample_idx = rng.choice(len(eligible), size=min(100, len(eligible)), replace=False)
        sample = eligible.iloc[sample_idx]

        for _, row in sample.iterrows():
            team = row["team_idx"]
            season = row["season"]
            game_date = pd.Timestamp(row["date"])

            # Get all prior games for this team in this season
            mask = (
                (team_logs["team_idx"] == team)
                & (team_logs["season"] == season)
                & (team_logs["date"] < game_date)
            )
            prior = team_logs[mask]

            if len(prior) == 0:
                continue

            expected_win_pct = prior["won"].mean()
            actual_win_pct = row["season_win_pct"]

            assert abs(expected_win_pct - actual_win_pct) < 0.001, (
                f"Leakage detected! team={team}, date={game_date}, "
                f"expected={expected_win_pct:.4f}, actual={actual_win_pct:.4f}"
            )

    def test_no_target_boxscore_used(self, team_logs, rolling_features):
        """A game's own stats must NOT be in its rolling features.

        For 50 rows: manually compute last_5_point_diff from prior 5 games,
        verify it matches the rolling feature.
        """
        rng = np.random.RandomState(123)
        eligible = rolling_features[rolling_features["season_games_played"] >= 6].copy()
        sample_idx = rng.choice(len(eligible), size=min(50, len(eligible)), replace=False)
        sample = eligible.iloc[sample_idx]

        for _, row in sample.iterrows():
            team = row["team_idx"]
            season = row["season"]
            game_date = pd.Timestamp(row["date"])

            prior = team_logs[
                (team_logs["team_idx"] == team)
                & (team_logs["season"] == season)
                & (team_logs["date"] < game_date)
            ].sort_values("date")

            if len(prior) < 5:
                continue

            expected = prior["point_diff"].iloc[-5:].mean()
            actual = row["last_5_point_diff"]

            assert abs(expected - actual) < 0.01, (
                f"Target boxscore leaked! team={team}, date={game_date}, "
                f"expected_last5_pd={expected:.2f}, actual={actual:.2f}"
            )

    def test_rolling_features_shifted_correctly(self, team_logs, rolling_features):
        """For a specific team's 10th game: verify last_5_win_pct
        is computed from games 5-9 (not 6-10).
        """
        # Pick team_idx=1 (BOS), first season
        seasons = sorted(rolling_features["season"].unique())
        first_season = seasons[0]

        team_data = rolling_features[
            (rolling_features["team_idx"] == 1)
            & (rolling_features["season"] == first_season)
        ].sort_values("date")

        if len(team_data) < 10:
            pytest.skip("Not enough games for team 1 in first season")

        # 10th game (index 9, 0-indexed) should use games 5-9 for last_5
        row = team_data.iloc[9]
        game_date = pd.Timestamp(row["date"])

        prior = team_logs[
            (team_logs["team_idx"] == 1)
            & (team_logs["season"] == first_season)
            & (team_logs["date"] < game_date)
        ].sort_values("date")

        expected = prior["won"].iloc[-5:].mean()
        actual = row["last_5_win_pct"]

        assert abs(expected - actual) < 0.001, (
            f"Shift error! expected={expected:.4f}, actual={actual:.4f}"
        )

    def test_season_opener_has_nan_or_zero(self, rolling_features):
        """First game of each season should have NaN for rolling features.

        (No prior games to compute from.)
        """
        openers = rolling_features[rolling_features["season_games_played"] == 0]
        assert len(openers) > 0, "No season openers found"

        # All rolling features should be NaN for season openers
        rolling_cols = [c for c in openers.columns
                        if c.startswith(("season_", "last_5_", "last_10_"))
                        and c != "season_games_played"]

        for col in rolling_cols:
            all_nan = openers[col].isna().all()
            assert all_nan, (
                f"Season opener should have NaN for {col}, "
                f"but has non-NaN values: {openers[col].dropna().head()}"
            )

    def test_features_before_target(self, matchup_df):
        """target_home_win must NOT appear in any feature column."""
        feature_cols = [c for c in matchup_df.columns
                        if c.startswith(("home_", "away_", "diff_"))]
        for col in feature_cols:
            assert "target" not in col.lower(), f"Target leaked into feature: {col}"
            assert "home_win" not in col.lower() or col.startswith("diff_") is False, (
                f"Suspicious column: {col}"
            )
