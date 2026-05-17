"""Build leakage-safe pregame player value features for each team-game."""

from __future__ import annotations

import logging

import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

PLAYER_VALUE_FEATURE_COLUMNS = [
    "game_id",
    "date",
    "season",
    "team_idx",
    "player_id",
    "player_idx",
    "player_name",
    "recent_games_played",
    "recent_minutes_avg",
    "recent_minutes_share",
    "recent_fantasy_points_avg",
    "recent_plus_minus_avg",
    "recent_points_avg",
    "recent_assists_avg",
    "recent_rebounds_avg",
    "recent_starter_rate",
    "recent_role_stability",
    "last_game_minutes",
    "last_game_date",
    "player_value_score",
    "rotation_rank",
]


def _empty_player_value_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PLAYER_VALUE_FEATURE_COLUMNS)


def _recent_player_pool(
    team_logs: pd.DataFrame,
    *,
    game_date: pd.Timestamp,
    recent_team_games: int,
    max_players: int,
    starter_size: int,
) -> pd.DataFrame:
    prior_logs = team_logs[team_logs["date"] < game_date].copy()
    if prior_logs.empty:
        return pd.DataFrame()

    recent_game_ids = (
        prior_logs.sort_values(["date", "game_id"])["game_id"]
        .drop_duplicates()
        .tail(recent_team_games)
    )
    if recent_game_ids.empty:
        return pd.DataFrame()

    recent_logs = prior_logs[prior_logs["game_id"].isin(recent_game_ids)].copy()
    if recent_logs.empty:
        return pd.DataFrame()

    for column in ["points", "rebounds", "assists", "plus_minus", "fantasy_points"]:
        if column not in recent_logs.columns:
            recent_logs[column] = 0.0

    recent_logs = recent_logs.sort_values(
        ["date", "game_id", "minutes", "player_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    recent_logs["minutes_rank"] = recent_logs.groupby("game_id")["minutes"].rank(
        method="first",
        ascending=False,
    )
    recent_logs["starter_proxy"] = (recent_logs["minutes_rank"] <= starter_size).astype(float)

    recency_weights = (
        recent_logs[["game_id", "date"]]
        .drop_duplicates()
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )
    recency_weights["recency_weight"] = (
        (recency_weights.index + 1) / len(recency_weights)
    ).astype(float)
    recent_logs = recent_logs.merge(
        recency_weights[["game_id", "recency_weight"]],
        on="game_id",
        how="left",
    )

    team_minutes_total = (
        recent_logs.groupby("game_id")["minutes"].sum().rename("team_minutes_total")
    )
    recent_logs = recent_logs.merge(team_minutes_total, on="game_id", how="left")
    total_recent_team_minutes = float(team_minutes_total.sum())
    total_recent_games = len(recency_weights)

    grouped = (
        recent_logs.groupby(["player_id", "player_idx", "player_name"], as_index=False)
        .agg(
            recent_games_played=("game_id", "nunique"),
            recent_minutes_avg=("minutes", "mean"),
            recent_fantasy_points_avg=("fantasy_points", "mean"),
            recent_plus_minus_avg=("plus_minus", "mean"),
            recent_points_avg=("points", "mean"),
            recent_assists_avg=("assists", "mean"),
            recent_rebounds_avg=("rebounds", "mean"),
            recent_minutes_total=("minutes", "sum"),
            recent_starter_count=("starter_proxy", "sum"),
            last_game_minutes=("minutes", "last"),
            last_game_date=("date", "max"),
        )
    )

    grouped["recent_minutes_share"] = (
        grouped["recent_minutes_total"] / total_recent_team_minutes
        if total_recent_team_minutes > 0
        else 0.0
    )
    grouped["recent_role_stability"] = grouped["recent_games_played"] / total_recent_games
    grouped["recent_starter_rate"] = grouped["recent_starter_count"] / total_recent_games
    grouped["player_value_score"] = (
        0.55 * grouped["recent_minutes_avg"]
        + 0.25 * grouped["recent_fantasy_points_avg"]
        + 20.0 * grouped["recent_minutes_share"]
        + 8.0 * grouped["recent_starter_rate"]
        + 6.0 * grouped["recent_role_stability"]
        + 0.15 * grouped["recent_plus_minus_avg"]
    ).round(4)

    grouped = grouped.sort_values(
        ["player_value_score", "recent_minutes_avg", "player_id"],
        ascending=[False, False, True],
    ).head(max_players)
    grouped["rotation_rank"] = range(1, len(grouped) + 1)

    return grouped[
        [
            "player_id",
            "player_idx",
            "player_name",
            "recent_games_played",
            "recent_minutes_avg",
            "recent_minutes_share",
            "recent_fantasy_points_avg",
            "recent_plus_minus_avg",
            "recent_points_avg",
            "recent_assists_avg",
            "recent_rebounds_avg",
            "recent_starter_rate",
            "recent_role_stability",
            "last_game_minutes",
            "last_game_date",
            "player_value_score",
            "rotation_rank",
        ]
    ].reset_index(drop=True)


def build_player_value_features(
    games: pd.DataFrame,
    player_logs: pd.DataFrame,
    *,
    recent_team_games: int = 10,
    max_players: int = 12,
    starter_size: int = 5,
) -> pd.DataFrame:
    """Build pregame player value features using only prior games for each target row."""
    if games.empty or player_logs.empty:
        return _empty_player_value_frame()

    logs = player_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])

    logs_by_team = {
        int(team_idx): group.sort_values(["date", "game_id"]).reset_index(drop=True)
        for team_idx, group in logs.groupby("team_idx", sort=False)
    }

    rows: list[dict[str, object]] = []
    for _, game in games.sort_values("date").iterrows():
        game_id = str(game["game_id"])
        game_date = pd.Timestamp(game["date"])
        season = str(game["season"])
        for team_idx in (int(game["home_team_idx"]), int(game["away_team_idx"])):
            team_logs = logs_by_team.get(team_idx)
            if team_logs is None or team_logs.empty:
                continue

            pool = _recent_player_pool(
                team_logs,
                game_date=game_date,
                recent_team_games=recent_team_games,
                max_players=max_players,
                starter_size=starter_size,
            )
            if pool.empty:
                continue

            for _, player in pool.iterrows():
                rows.append(
                    {
                        "game_id": game_id,
                        "date": game_date,
                        "season": season,
                        "team_idx": team_idx,
                        "player_id": int(player["player_id"]),
                        "player_idx": int(player["player_idx"]),
                        "player_name": str(player["player_name"]),
                        "recent_games_played": int(player["recent_games_played"]),
                        "recent_minutes_avg": float(player["recent_minutes_avg"]),
                        "recent_minutes_share": float(player["recent_minutes_share"]),
                        "recent_fantasy_points_avg": float(player["recent_fantasy_points_avg"]),
                        "recent_plus_minus_avg": float(player["recent_plus_minus_avg"]),
                        "recent_points_avg": float(player["recent_points_avg"]),
                        "recent_assists_avg": float(player["recent_assists_avg"]),
                        "recent_rebounds_avg": float(player["recent_rebounds_avg"]),
                        "recent_starter_rate": float(player["recent_starter_rate"]),
                        "recent_role_stability": float(player["recent_role_stability"]),
                        "last_game_minutes": float(player["last_game_minutes"]),
                        "last_game_date": pd.Timestamp(player["last_game_date"]),
                        "player_value_score": float(player["player_value_score"]),
                        "rotation_rank": int(player["rotation_rank"]),
                    }
                )

    if not rows:
        return _empty_player_value_frame()

    result = pd.DataFrame(rows, columns=PLAYER_VALUE_FEATURE_COLUMNS)
    return result.sort_values(
        ["date", "game_id", "team_idx", "rotation_rank", "player_id"]
    ).reset_index(drop=True)


def main() -> None:
    """Build and save historical player value features."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    player_logs = pd.read_parquet(PROCESSED_DIR / "player_game_logs" / "player_game_logs.parquet")
    features = build_player_value_features(games, player_logs)

    out_dir = PROCESSED_DIR / "player_value_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "player_value_features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Saved player value features rows=%d to %s", len(features), out_path)


if __name__ == "__main__":
    main()
