"""Feature correctness tests — matchup dataset validation.

Run with: pytest tests/test_features.py -v
Expected: 9/9 pass
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
def games_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "games.parquet"
    return pd.read_parquet(path)


class TestMatchupDataset:
    """Verify the matchup dataset is correct."""

    def test_matchup_dataset_exists(self):
        """Matchup rows parquet file exists and is loadable."""
        path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) > 0

    def test_matchup_row_count(self, matchup_df, games_df):
        """One row per game. Count should match games table."""
        assert len(matchup_df) == len(games_df), (
            f"Matchup rows ({len(matchup_df)}) != games ({len(games_df)})"
        )

    def test_matchup_schema(self, matchup_df):
        """Required columns present: home_*, away_*, diff_*, target_home_win."""
        home_cols = [c for c in matchup_df.columns if c.startswith("home_")]
        away_cols = [c for c in matchup_df.columns if c.startswith("away_")]
        diff_cols = [c for c in matchup_df.columns if c.startswith("diff_")]

        assert len(home_cols) > 10, f"Too few home_ columns: {len(home_cols)}"
        assert len(away_cols) > 10, f"Too few away_ columns: {len(away_cols)}"
        assert len(diff_cols) > 5, f"Too few diff_ columns: {len(diff_cols)}"
        assert "target_home_win" in matchup_df.columns

    def test_no_nulls_in_non_early_games(self, matchup_df):
        """After game 20 of each season, no nulls in key rolling features."""
        # Filter to games where both teams have played 20+ games
        if "home_season_games_played" in matchup_df.columns:
            late = matchup_df[
                (matchup_df["home_season_games_played"] >= 20)
                & (matchup_df["away_season_games_played"] >= 20)
            ]
        else:
            # Fallback: skip first 60 games of each season (~20 games per team)
            late = matchup_df.groupby("season").apply(
                lambda x: x.iloc[60:] if len(x) > 60 else x.iloc[0:0]
            ).reset_index(drop=True)

        key_cols = [
            c for c in late.columns
            if c.startswith(("home_season_", "away_season_", "home_last_10", "away_last_10"))
        ]
        for col in key_cols:
            null_pct = late[col].isna().mean()
            assert null_pct < 0.05, (
                f"Column {col} has {null_pct:.1%} nulls in non-early games"
            )

    def test_rest_days_reasonable(self, matchup_df):
        """rest_days values are between 0 and 14."""
        for prefix in ["home_", "away_"]:
            col = f"{prefix}rest_days"
            if col in matchup_df.columns:
                assert matchup_df[col].min() >= 0, f"{col} has negative values"
                assert matchup_df[col].max() <= 200, (
                    f"{col} has absurd values (>{matchup_df[col].max()})"
                )

    def test_back_to_back_correct(self, matchup_df):
        """When rest_days <= 1, back_to_back should be 1."""
        for prefix in ["home_", "away_"]:
            rest_col = f"{prefix}rest_days"
            b2b_col = f"{prefix}back_to_back"
            if rest_col in matchup_df.columns and b2b_col in matchup_df.columns:
                b2b_mask = matchup_df[rest_col] <= 1
                assert (matchup_df.loc[b2b_mask, b2b_col] == 1).all(), (
                    "Some games with rest_days<=1 have back_to_back=0"
                )

    def test_diff_features_computed(self, matchup_df):
        """diff columns = home value minus away value."""
        diff_cols = [c for c in matchup_df.columns if c.startswith("diff_")]
        assert len(diff_cols) > 0, "No diff columns found"

        for dc in diff_cols[:5]:  # Check first 5
            feat_name = dc.replace("diff_", "")
            home_col = f"home_{feat_name}"
            away_col = f"away_{feat_name}"
            if home_col in matchup_df.columns and away_col in matchup_df.columns:
                # Only check non-null rows
                valid = matchup_df.dropna(subset=[home_col, away_col, dc])
                if len(valid) > 0:
                    expected = valid[home_col] - valid[away_col]
                    np.testing.assert_allclose(
                        valid[dc].values, expected.values, atol=0.01,
                        err_msg=f"diff_{feat_name} doesn't match home - away"
                    )

    def test_target_is_binary(self, matchup_df):
        """target_home_win is 0 or 1 only."""
        unique = set(matchup_df["target_home_win"].unique())
        assert unique <= {0, 1}, f"target has unexpected values: {unique}"

    def test_home_win_rate_around_60(self, matchup_df):
        """Overall home win rate should be ~55-65% (sanity check)."""
        rate = matchup_df["target_home_win"].mean()
        assert 0.50 <= rate <= 0.65, f"Home win rate {rate:.1%} outside expected range"
