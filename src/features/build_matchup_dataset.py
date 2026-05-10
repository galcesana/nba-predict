"""Build the final matchup dataset: one row per game with home/away features + diff.

Merges rolling features + schedule features for both home and away teams,
computes difference features, and attaches the target label.

Usage:
    python -m src.features.build_matchup_dataset
"""

import logging

import pandas as pd

from src.features.rolling_features import compute_rolling_features
from src.features.schedule_features import compute_schedule_features
from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Key features to compute home-minus-away differences for
DIFF_FEATURES = [
    "season_win_pct", "last_10_win_pct",
    "season_point_diff", "last_10_point_diff",
    "season_net_rating", "last_10_net_rating",
    "season_off_rating", "season_def_rating",
    "last_10_off_rating", "last_10_def_rating",
    "season_pace", "last_10_pace",
    "season_ts_pct", "season_efg_pct",
    "season_turnover_pct", "season_assist_pct",
]


def build_matchup_dataset(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build the matchup dataset by merging home/away rolling + schedule features.

    Args:
        games: Games table (one row per game).
        team_logs: Team game logs (one row per team per game).

    Returns:
        Matchup dataset with one row per game, home/away/diff features, and target.
    """
    # Compute rolling and schedule features
    logger.info("Computing rolling features...")
    rolling = compute_rolling_features(team_logs)

    logger.info("Computing schedule features...")
    schedule = compute_schedule_features(team_logs)

    # Merge rolling + schedule on (game_id, team_idx)
    team_features = rolling.merge(schedule, on=["game_id", "team_idx"], how="left")

    # Identify feature columns (exclude metadata)
    meta_cols = {"game_id", "team_idx", "date", "season"}
    feature_cols = [c for c in team_features.columns if c not in meta_cols]

    # Split into home and away features
    home_features = team_features.merge(
        games[["game_id", "home_team_idx"]],
        left_on=["game_id", "team_idx"],
        right_on=["game_id", "home_team_idx"],
    ).drop(columns=["home_team_idx"])

    away_features = team_features.merge(
        games[["game_id", "away_team_idx"]],
        left_on=["game_id", "team_idx"],
        right_on=["game_id", "away_team_idx"],
    ).drop(columns=["away_team_idx"])

    # Rename columns with home_/away_ prefix
    home_renamed = home_features[["game_id"] + feature_cols].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    away_renamed = away_features[["game_id"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )

    # Start from games table
    matchup = games[["game_id", "date", "season", "home_team_idx", "away_team_idx"]].copy()

    # Merge home and away features
    matchup = matchup.merge(home_renamed, on="game_id", how="left")
    matchup = matchup.merge(away_renamed, on="game_id", how="left")

    # Compute diff features (home - away)
    for feat in DIFF_FEATURES:
        home_col = f"home_{feat}"
        away_col = f"away_{feat}"
        if home_col in matchup.columns and away_col in matchup.columns:
            matchup[f"diff_{feat}"] = matchup[home_col] - matchup[away_col]

    # Add target
    matchup["target_home_win"] = games.set_index("game_id").loc[
        matchup["game_id"], "home_win"
    ].values

    # Sort by date
    matchup = matchup.sort_values("date").reset_index(drop=True)

    logger.info(
        "Matchup dataset: %d rows, %d columns, home win rate: %.1f%%",
        len(matchup),
        len(matchup.columns),
        matchup["target_home_win"].mean() * 100,
    )

    return matchup


def main():
    """Build and save the matchup dataset."""
    setup_logging()

    games_path = PROCESSED_DIR / "games.parquet"
    logs_path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"

    logger.info("Loading data...")
    games = pd.read_parquet(games_path)
    team_logs = pd.read_parquet(logs_path)

    logger.info("Building matchup dataset...")
    matchup = build_matchup_dataset(games, team_logs)

    out_path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matchup.to_parquet(out_path, index=False)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
