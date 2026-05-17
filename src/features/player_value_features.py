"""Build leakage-safe pregame player value features for each team-game."""

from __future__ import annotations

import logging

import numpy as np
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

STAT_COLUMNS = [
    "minutes",
    "fantasy_points",
    "plus_minus",
    "points",
    "assists",
    "rebounds",
]


def _empty_player_value_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PLAYER_VALUE_FEATURE_COLUMNS)


def _team_target_rows(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = games[["game_id", "date", "season", "home_team_idx"]].rename(
        columns={"home_team_idx": "team_idx"}
    )
    away_rows = games[["game_id", "date", "season", "away_team_idx"]].rename(
        columns={"away_team_idx": "team_idx"}
    )
    targets = pd.concat([home_rows, away_rows], ignore_index=True)
    targets["team_idx"] = targets["team_idx"].astype(int)
    return targets.sort_values(["team_idx", "date", "game_id"]).reset_index(drop=True)


def _rolling_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=1).sum().shift(1).fillna(0.0)


def _rolling_mean_from_sum(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
) -> pd.DataFrame:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def _build_team_player_value_rows(
    team_idx: int,
    team_targets: pd.DataFrame,
    team_logs: pd.DataFrame,
    *,
    recent_team_games: int,
    max_players: int,
    starter_size: int,
) -> pd.DataFrame:
    if team_logs.empty or team_targets.empty:
        return pd.DataFrame(columns=PLAYER_VALUE_FEATURE_COLUMNS)

    logs = team_logs.copy()
    logs["game_id"] = logs["game_id"].astype(str)
    logs = logs.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)
    for column in STAT_COLUMNS:
        if column not in logs.columns:
            logs[column] = 0.0
    team_games = (
        logs[["game_id", "date", "season"]]
        .drop_duplicates()
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )
    team_games["game_sequence"] = range(len(team_games))
    game_sequence_lookup = team_games.set_index("game_id")["game_sequence"]
    logs["game_sequence"] = logs["game_id"].map(game_sequence_lookup).astype(int)

    logs["minutes_rank"] = logs.groupby("game_sequence")["minutes"].rank(
        method="first",
        ascending=False,
    )
    logs["starter_proxy"] = (logs["minutes_rank"] <= starter_size).astype(float)

    target_game_ids = set(team_targets["game_id"].astype(str))

    player_meta = (
        logs.sort_values(["date", "game_id", "player_id"])
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")[["player_idx", "player_name"]]
    )

    expanded_index = pd.Index(range(len(team_games) + 1), name="history_index")
    game_dates = team_games["date"].reset_index(drop=True)
    prior_game_counts = pd.Series(
        np.minimum(expanded_index.to_numpy(), recent_team_games),
        index=expanded_index,
        dtype=float,
    )

    def _pivot_stat(column: str) -> pd.DataFrame:
        return (
            logs.pivot_table(
                index="game_sequence",
                columns="player_id",
                values=column,
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index()
            .sort_index(axis=1)
        )

    minutes = _pivot_stat("minutes").reindex(expanded_index, fill_value=0.0)
    player_ids = minutes.columns
    appearances = (minutes > 0).astype(float)
    fantasy_points = _pivot_stat("fantasy_points").reindex(expanded_index, fill_value=0.0)
    fantasy_points = fantasy_points.reindex(columns=player_ids, fill_value=0.0)
    plus_minus = _pivot_stat("plus_minus").reindex(expanded_index, fill_value=0.0)
    plus_minus = plus_minus.reindex(columns=player_ids, fill_value=0.0)
    points = _pivot_stat("points").reindex(expanded_index, fill_value=0.0)
    points = points.reindex(columns=player_ids, fill_value=0.0)
    assists = _pivot_stat("assists").reindex(expanded_index, fill_value=0.0)
    assists = assists.reindex(columns=player_ids, fill_value=0.0)
    rebounds = _pivot_stat("rebounds").reindex(expanded_index, fill_value=0.0)
    rebounds = rebounds.reindex(columns=player_ids, fill_value=0.0)
    starter_proxy = _pivot_stat("starter_proxy").reindex(expanded_index, fill_value=0.0)
    starter_proxy = starter_proxy.reindex(columns=player_ids, fill_value=0.0)

    recent_games_played = _rolling_sum(appearances, recent_team_games)
    recent_minutes_total = _rolling_sum(minutes, recent_team_games)
    recent_fantasy_points_total = _rolling_sum(fantasy_points, recent_team_games)
    recent_plus_minus_total = _rolling_sum(plus_minus, recent_team_games)
    recent_points_total = _rolling_sum(points, recent_team_games)
    recent_assists_total = _rolling_sum(assists, recent_team_games)
    recent_rebounds_total = _rolling_sum(rebounds, recent_team_games)
    recent_starter_count = _rolling_sum(starter_proxy, recent_team_games)

    recent_minutes_avg = _rolling_mean_from_sum(recent_minutes_total, recent_games_played)
    recent_fantasy_points_avg = _rolling_mean_from_sum(
        recent_fantasy_points_total,
        recent_games_played,
    )
    recent_plus_minus_avg = _rolling_mean_from_sum(recent_plus_minus_total, recent_games_played)
    recent_points_avg = _rolling_mean_from_sum(recent_points_total, recent_games_played)
    recent_assists_avg = _rolling_mean_from_sum(recent_assists_total, recent_games_played)
    recent_rebounds_avg = _rolling_mean_from_sum(recent_rebounds_total, recent_games_played)

    prior_game_counts_frame = pd.DataFrame(
        np.repeat(prior_game_counts.to_numpy()[:, None], len(player_ids), axis=1),
        index=expanded_index,
        columns=player_ids,
    )
    recent_role_stability = recent_games_played.div(prior_game_counts_frame).fillna(0.0)
    recent_starter_rate = recent_starter_count.div(prior_game_counts_frame).fillna(0.0)

    recent_team_minutes_total = (
        minutes.sum(axis=1).rolling(window=recent_team_games, min_periods=1).sum().shift(1)
    ).fillna(0.0)
    recent_minutes_share = recent_minutes_total.div(recent_team_minutes_total, axis=0).fillna(0.0)

    player_value_score = (
        0.55 * recent_minutes_avg
        + 0.25 * recent_fantasy_points_avg
        + 20.0 * recent_minutes_share
        + 8.0 * recent_starter_rate
        + 6.0 * recent_role_stability
        + 0.15 * recent_plus_minus_avg
    ).round(4)

    last_game_minutes = minutes.replace(0.0, np.nan).ffill().shift(1)
    expanded_dates = pd.Index([*game_dates.to_list(), pd.NaT], dtype="datetime64[ns]")
    date_frame = pd.DataFrame(
        np.repeat(expanded_dates.to_numpy()[:, None], len(player_ids), axis=1),
        index=expanded_index,
        columns=player_ids,
    )
    last_game_date = date_frame.where(appearances.astype(bool)).ffill().shift(1)

    long_frame = pd.concat(
        [
            recent_games_played.stack().rename("recent_games_played"),
            recent_minutes_avg.stack().rename("recent_minutes_avg"),
            recent_minutes_share.stack().rename("recent_minutes_share"),
            recent_fantasy_points_avg.stack().rename("recent_fantasy_points_avg"),
            recent_plus_minus_avg.stack().rename("recent_plus_minus_avg"),
            recent_points_avg.stack().rename("recent_points_avg"),
            recent_assists_avg.stack().rename("recent_assists_avg"),
            recent_rebounds_avg.stack().rename("recent_rebounds_avg"),
            recent_starter_rate.stack().rename("recent_starter_rate"),
            recent_role_stability.stack().rename("recent_role_stability"),
            last_game_minutes.stack(future_stack=True).rename("last_game_minutes"),
            last_game_date.stack(future_stack=True).rename("last_game_date"),
            player_value_score.stack().rename("player_value_score"),
        ],
        axis=1,
    ).reset_index(names=["history_index", "player_id"])

    long_frame = long_frame[long_frame["recent_games_played"] > 0].copy()
    if long_frame.empty:
        return pd.DataFrame(columns=PLAYER_VALUE_FEATURE_COLUMNS)

    target_frame = team_targets.copy()
    target_dates = pd.to_datetime(target_frame["date"]).to_numpy()
    history_dates = pd.to_datetime(team_games["date"]).to_numpy()
    target_frame["history_index"] = np.searchsorted(history_dates, target_dates, side="left")
    target_frame["game_id"] = target_frame["game_id"].astype(str)
    target_frame = target_frame[target_frame["game_id"].isin(target_game_ids)]
    if target_frame.empty:
        return pd.DataFrame(columns=PLAYER_VALUE_FEATURE_COLUMNS)

    long_frame = long_frame.merge(
        target_frame[["game_id", "date", "season", "history_index"]],
        on="history_index",
        how="inner",
    )
    if long_frame.empty:
        return pd.DataFrame(columns=PLAYER_VALUE_FEATURE_COLUMNS)

    long_frame["team_idx"] = int(team_idx)
    long_frame["player_idx"] = long_frame["player_id"].map(player_meta["player_idx"]).astype(int)
    long_frame["player_name"] = long_frame["player_id"].map(player_meta["player_name"]).astype(str)

    long_frame = long_frame.sort_values(
        ["game_id", "player_value_score", "recent_minutes_avg", "player_id"],
        ascending=[True, False, False, True],
    )
    long_frame["rotation_rank"] = long_frame.groupby("game_id").cumcount() + 1
    long_frame = long_frame[long_frame["rotation_rank"] <= max_players].copy()

    long_frame["last_game_date"] = pd.to_datetime(long_frame["last_game_date"])
    return long_frame[PLAYER_VALUE_FEATURE_COLUMNS].reset_index(drop=True)


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
    target_rows = _team_target_rows(games)

    logs_by_team = {
        int(team_idx): group.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)
        for team_idx, group in logs.groupby("team_idx", sort=False)
    }

    team_frames: list[pd.DataFrame] = []
    team_count = target_rows["team_idx"].nunique()
    for team_number, (team_idx, team_targets) in enumerate(
        target_rows.groupby("team_idx", sort=False),
        start=1,
    ):
        team_logs = logs_by_team.get(int(team_idx))
        if team_logs is None or team_logs.empty:
            continue

        logger.info(
            "Building player value features for team %d (%d/%d)",
            int(team_idx),
            team_number,
            team_count,
        )
        team_frame = _build_team_player_value_rows(
            int(team_idx),
            team_targets,
            team_logs,
            recent_team_games=recent_team_games,
            max_players=max_players,
            starter_size=starter_size,
        )
        if not team_frame.empty:
            team_frames.append(team_frame)

    if not team_frames:
        return _empty_player_value_frame()

    result = pd.concat(team_frames, ignore_index=True)
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
