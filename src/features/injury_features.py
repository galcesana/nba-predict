"""Injury feature engineering — team injury impact vector builder.

Builds injury features for each team per game. In V1, we use a synthetic
approach based on team performance variance and roster availability signals,
since full player-level injury data requires extensive API fetching.

The infrastructure supports plugging in real injury data when available.

Features produced per team per game:
    players_out_count: Number of players listed as out
    players_questionable_count: Number of players listed as questionable
    starter_out_count: Number of starters out
    minutes_missing: Estimated minutes of missing players
    usage_missing: Estimated usage rate of missing players
    estimated_value_missing: Combined value metric of missing players
    injury_data_available: 1 if real data, 0 if using defaults

Usage:
    python -m src.features.injury_features
"""

import logging

import numpy as np
import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# Injury feature columns
INJURY_FEATURE_COLS = [
    "players_out_count",
    "players_questionable_count",
    "starter_out_count",
    "minutes_missing",
    "usage_missing",
    "estimated_value_missing",
    "injury_data_available",
]


def load_injury_reports() -> pd.DataFrame | None:
    """Load raw injury report data if available.

    Returns:
        DataFrame with columns: game_id, team_idx, player_id, status,
        player_minutes_avg, player_usage_avg, player_value
        or None if no injury data exists.
    """
    injury_dir = RAW_DIR / "injuries"
    if not injury_dir.exists():
        logger.info("No injuries directory found — using defaults")
        return None

    injury_files = [f for f in injury_dir.iterdir() if f.suffix == ".parquet"]

    if not injury_files:
        logger.info("No injury report data found — using defaults")
        return None

    dfs = [pd.read_parquet(f) for f in injury_files]
    return pd.concat(dfs, ignore_index=True)


def build_injury_features_from_reports(
    injury_reports: pd.DataFrame,
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build injury features from actual injury report data.

    Args:
        injury_reports: Raw injury reports with player statuses.
        games: Games table.
        team_logs: Team game logs.

    Returns:
        DataFrame with game_id, team_idx, and injury feature columns.
    """
    features = []

    for _, game in games.iterrows():
        game_id = game["game_id"]

        for team_col in ["home_team_idx", "away_team_idx"]:
            team_idx = int(game[team_col])

            # Get injuries for this game/team
            mask = (
                (injury_reports["game_id"] == game_id)
                & (injury_reports["team_idx"] == team_idx)
            )
            game_injuries = injury_reports[mask]

            if len(game_injuries) == 0:
                features.append({
                    "game_id": game_id,
                    "team_idx": team_idx,
                    **{c: 0 for c in INJURY_FEATURE_COLS[:-1]},
                    "injury_data_available": 1,  # Data exists, just no injuries
                })
                continue

            out_players = game_injuries[game_injuries["status"] == "Out"]
            questionable = game_injuries[game_injuries["status"].isin(["Questionable", "Doubtful"])]

            features.append({
                "game_id": game_id,
                "team_idx": team_idx,
                "players_out_count": len(out_players),
                "players_questionable_count": len(questionable),
                "starter_out_count": int(out_players.get("is_starter", pd.Series([0])).sum()),
                "minutes_missing": float(out_players.get("player_minutes_avg", pd.Series([0])).sum()),
                "usage_missing": float(out_players.get("player_usage_avg", pd.Series([0])).sum()),
                "estimated_value_missing": float(out_players.get("player_value", pd.Series([0])).sum()),
                "injury_data_available": 1,
            })

    return pd.DataFrame(features)


def build_injury_features_default(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build default injury features when no injury data is available.

    Uses team performance variance as a proxy — teams with high game-to-game
    variance in key stats may be experiencing roster instability.
    Vectorized implementation for speed.

    Args:
        games: Games table.
        team_logs: Team game logs.

    Returns:
        DataFrame with game_id, team_idx, and injury feature columns.
    """
    team_logs = team_logs.sort_values(["team_idx", "date"]).copy()
    team_logs["date"] = pd.to_datetime(team_logs["date"])

    # Pre-compute rolling stats per team: 10-game rolling std + 3-game mean
    all_team_stats = {}
    for team_idx, group in team_logs.groupby("team_idx"):
        group = group.sort_values("date").reset_index(drop=True)
        nr_col = "net_rating" if "net_rating" in group.columns else "point_diff"
        vals = group[nr_col]
        # Rolling 10-game std (shifted to avoid leakage)
        roll_std = vals.shift(1).rolling(10, min_periods=3).std().values
        # Rolling 3-game mean vs 10-game mean (shifted)
        mean_3 = vals.shift(1).rolling(3, min_periods=3).mean().values
        mean_10 = vals.shift(1).rolling(10, min_periods=3).mean().values
        drop = np.maximum(0, mean_10 - mean_3)  # Recent decline
        dates = group["date"].values
        game_ids_team = group["game_id"].values
        all_team_stats[team_idx] = {
            "dates": dates,
            "game_ids": game_ids_team,
            "roll_std": roll_std,
            "drop": drop,
        }

    # Build features for all (game, team) pairs
    rows = []
    games_sorted = games.sort_values("date").copy()

    for team_col in ["home_team_idx", "away_team_idx"]:
        for _, game in games_sorted.iterrows():
            game_id = game["game_id"]
            team_idx = int(game[team_col])
            game_date = pd.Timestamp(game["date"])

            stats = all_team_stats.get(team_idx)
            if stats is None:
                rows.append({
                    "game_id": game_id,
                    "team_idx": team_idx,
                    **{c: 0.0 for c in INJURY_FEATURE_COLS[:-1]},
                    "injury_data_available": 0,
                })
                continue

            # Find this team's games before current date
            prior_mask = stats["dates"] < game_date
            prior_indices = np.where(prior_mask)[0]

            if len(prior_indices) < 3:
                rows.append({
                    "game_id": game_id,
                    "team_idx": team_idx,
                    **{c: 0.0 for c in INJURY_FEATURE_COLS[:-1]},
                    "injury_data_available": 0,
                })
                continue

            last_idx = prior_indices[-1]
            drop_val = stats["drop"][last_idx]
            drop_val = 0.0 if np.isnan(drop_val) else float(drop_val)

            rows.append({
                "game_id": game_id,
                "team_idx": team_idx,
                "players_out_count": 0,
                "players_questionable_count": 0,
                "starter_out_count": 0,
                "minutes_missing": 0.0,
                "usage_missing": 0.0,
                "estimated_value_missing": drop_val,
                "injury_data_available": 0,
            })

    result = pd.DataFrame(rows)
    logger.info(
        "Built default injury features: %d rows, injury_data_available=0 for all",
        len(result),
    )
    return result


def build_injury_features(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build injury features, using real data if available, defaults otherwise.

    Args:
        games: Games table.
        team_logs: Team game logs.

    Returns:
        DataFrame with game_id, team_idx, and injury feature columns.
    """
    injury_reports = load_injury_reports()

    if injury_reports is not None:
        logger.info("Building injury features from %d injury reports", len(injury_reports))
        return build_injury_features_from_reports(injury_reports, games, team_logs)
    else:
        logger.info("No injury data — building default features")
        return build_injury_features_default(games, team_logs)


def main():
    """Build and save injury features."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    team_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")

    logger.info("Building injury features for %d games...", len(games))
    features = build_injury_features(games, team_logs)

    out_dir = PROCESSED_DIR / "injury_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "injury_features.parquet"
    features.to_parquet(out_path, index=False)

    logger.info(
        "Saved injury features: %d rows, %d columns to %s",
        len(features), len(features.columns), out_path,
    )
    logger.info(
        "Coverage: %.1f%% with real injury data",
        features["injury_data_available"].mean() * 100,
    )


if __name__ == "__main__":
    main()
