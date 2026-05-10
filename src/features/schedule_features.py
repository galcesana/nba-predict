"""Compute schedule-based features: rest days, back-to-back, games in window.

All features are computed from prior game dates only — no leakage.

Usage:
    python -m src.features.schedule_features
"""

import logging

import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)


def compute_schedule_features(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute schedule features per team per game.

    Features:
        rest_days: days since last game (0 = back-to-back)
        back_to_back: 1 if rest_days == 0
        games_last_7: number of games in last 7 days (before this game)
        games_last_14: number of games in last 14 days (before this game)

    Args:
        team_logs: Team game logs with game_id, date, team_idx, season.

    Returns:
        DataFrame with game_id, team_idx, and schedule feature columns.
    """
    team_logs = team_logs.sort_values(["team_idx", "season", "date"]).copy()
    team_logs["date"] = pd.to_datetime(team_logs["date"])

    all_features = []

    for (team_idx, season), group in team_logs.groupby(["team_idx", "season"]):
        group = group.sort_values("date").copy()
        dates = group["date"].values

        rest_days = []
        games_last_7 = []
        games_last_14 = []

        for i in range(len(group)):
            current_date = dates[i]

            if i == 0:
                # First game of the season for this team — no prior data
                rest_days.append(7)  # default to 7 (full rest)
                games_last_7.append(0)
                games_last_14.append(0)
            else:
                prev_date = dates[i - 1]
                delta = (current_date - prev_date) / pd.Timedelta(days=1)
                rest_days.append(int(delta))

                # Count games in last 7/14 days (strictly before current game)
                prior_dates = dates[:i]
                g7 = int(((current_date - prior_dates) / pd.Timedelta(days=1) <= 7).sum())
                g14 = int(((current_date - prior_dates) / pd.Timedelta(days=1) <= 14).sum())
                games_last_7.append(g7)
                games_last_14.append(g14)

        features = pd.DataFrame({
            "game_id": group["game_id"].values,
            "team_idx": group["team_idx"].values,
            "rest_days": rest_days,
            "back_to_back": [1 if r <= 1 else 0 for r in rest_days],
            "games_last_7": games_last_7,
            "games_last_14": games_last_14,
        })

        all_features.append(features)

    result = pd.concat(all_features, ignore_index=True)
    logger.info("Computed schedule features: %d rows", len(result))
    return result


def main():
    """Build schedule features and save."""
    setup_logging()

    logs_path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
    logger.info("Loading team game logs from %s", logs_path)
    team_logs = pd.read_parquet(logs_path)

    logger.info("Computing schedule features...")
    features = compute_schedule_features(team_logs)

    out_path = PROCESSED_DIR / "team_schedule_features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Saved schedule features to %s (%d rows)", out_path, len(features))


if __name__ == "__main__":
    main()
