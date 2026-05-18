"""Build lineup and rotation features from projected availability and player logs."""

from __future__ import annotations

import logging

import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

LINEUP_FEATURE_VERSION = "historical_absence_proxy_v1"

LINEUP_FEATURE_COLS = [
    "expected_starter_continuity",
    "expected_top8_continuity",
    "projected_minutes_concentration",
    "bench_depth_quality",
    "rotation_stability",
    "lineup_familiarity",
    "expected_missing_starter_value",
    "expected_missing_rotation_value",
    "projected_available_starter_value",
    "projected_available_rotation_value",
    "available_top8_players",
]


def _projection_value_column(projected_availability: pd.DataFrame) -> str:
    if "player_value_score" in projected_availability.columns:
        return "player_value_score"
    return "role_score"


def _team_target_rows(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = games[["game_id", "date", "home_team_idx"]].rename(
        columns={"home_team_idx": "team_idx"}
    )
    away_rows = games[["game_id", "date", "away_team_idx"]].rename(
        columns={"away_team_idx": "team_idx"}
    )
    targets = pd.concat([home_rows, away_rows], ignore_index=True)
    targets["team_idx"] = targets["team_idx"].astype(int)
    return targets.sort_values(["team_idx", "date", "game_id"]).reset_index(drop=True)


def _top_players_from_game_rows(game_rows: pd.DataFrame, *, top_n: int) -> set[int]:
    if game_rows.empty:
        return set()
    return set(
        game_rows.sort_values(["minutes", "player_id"], ascending=[False, True])
        .head(top_n)["player_id"]
        .astype(int)
    )


def build_lineup_features(
    games: pd.DataFrame,
    player_logs: pd.DataFrame,
    projected_availability: pd.DataFrame,
    *,
    rotation_size: int = 8,
    starter_size: int = 5,
    recent_team_games: int = 10,
) -> pd.DataFrame:
    """Build team-level lineup/rotation features for each team-game."""
    if games.empty or projected_availability.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_idx",
                "lineup_feature_stack_version",
                *LINEUP_FEATURE_COLS,
            ]
        )

    logs = player_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    availability = projected_availability.copy()
    if "date" in availability.columns:
        availability["date"] = pd.to_datetime(availability["date"])
    value_column = _projection_value_column(availability)
    target_rows = _team_target_rows(games)
    logs_by_team = {
        int(team_idx): group.sort_values(["date", "game_id"]).reset_index(drop=True)
        for team_idx, group in logs.groupby("team_idx", sort=False)
    }
    projection_groups = {
        (str(game_id), int(team_idx)): group.copy()
        for (game_id, team_idx), group in availability.groupby(["game_id", "team_idx"], sort=False)
    }

    rows: list[dict[str, object]] = []
    team_count = target_rows["team_idx"].nunique()
    for team_number, (team_idx, team_targets) in enumerate(
        target_rows.groupby("team_idx", sort=False),
        start=1,
    ):
        team_logs = logs_by_team.get(int(team_idx))
        if team_logs is None or team_logs.empty:
            continue

        logger.info(
            "Building lineup features for team %d (%d/%d)",
            int(team_idx),
            team_number,
            team_count,
        )
        team_games = (
            team_logs[["game_id", "date"]]
            .drop_duplicates()
            .sort_values(["date", "game_id"])
            .reset_index(drop=True)
        )
        game_log_lookup = {
            str(game_id): frame.copy()
            for game_id, frame in team_logs.groupby("game_id", sort=False)
        }
        history_pointer = 0

        for _, game in team_targets.iterrows():
            game_id = str(game["game_id"])
            game_date = pd.Timestamp(game["date"])
            team_projection = projection_groups.get((game_id, int(team_idx)))
            if team_projection is None:
                continue
            team_projection = team_projection.copy()
            if team_projection.empty:
                continue

            team_projection["effective_role_value"] = (
                team_projection[value_column] * team_projection["availability_score"]
            )
            team_projection["effective_minutes"] = (
                team_projection["expected_minutes"] * team_projection["availability_score"]
            )
            team_projection = team_projection.sort_values(
                ["effective_role_value", value_column, "player_id"],
                ascending=[False, False, True],
            )

            starters = team_projection.head(starter_size).copy()
            rotation = team_projection.head(rotation_size).copy()

            while (
                history_pointer < len(team_games)
                and pd.Timestamp(team_games.iloc[history_pointer]["date"]) < game_date
            ):
                history_pointer += 1

            recent_game_ids = (
                team_games.iloc[
                    max(0, history_pointer - recent_team_games) : history_pointer
                ]["game_id"]
                .astype(str)
                .tolist()
            )
            last_game_id = recent_game_ids[-1] if recent_game_ids else None
            recent_frames = [
                game_log_lookup[recent_game_id]
                for recent_game_id in recent_game_ids
                if recent_game_id in game_log_lookup
            ]
            recent_logs = (
                pd.concat(recent_frames, ignore_index=True)
                if recent_frames
                else pd.DataFrame(columns=team_logs.columns)
            )

            last_game_rows = (
                game_log_lookup.get(last_game_id, pd.DataFrame(columns=team_logs.columns))
                if last_game_id is not None
                else pd.DataFrame(columns=team_logs.columns)
            )
            prior_top5 = _top_players_from_game_rows(last_game_rows, top_n=starter_size)
            prior_top8 = _top_players_from_game_rows(last_game_rows, top_n=rotation_size)

            starter_ids = set(starters["player_id"].astype(int))
            rotation_ids = set(rotation["player_id"].astype(int))
            starter_continuity = len(starter_ids & prior_top5) / starter_size if prior_top5 else 0.0
            top8_continuity = len(rotation_ids & prior_top8) / rotation_size if prior_top8 else 0.0

            total_rotation_value = float(rotation["effective_role_value"].sum())
            starter_value = float(starters["effective_role_value"].sum())
            total_rotation_minutes = float(rotation["effective_minutes"].sum())
            starter_minutes = float(starters["effective_minutes"].sum())
            projected_minutes_concentration = (
                starter_minutes / total_rotation_minutes if total_rotation_minutes > 0 else 0.0
            )
            bench_depth_quality = float(rotation.iloc[starter_size:]["effective_role_value"].sum())

            if recent_game_ids:
                appearance_counts = (
                    recent_logs.groupby("player_id")["game_id"].nunique()
                )
                rotation_stability = float(
                    appearance_counts.reindex(rotation["player_id"]).fillna(0).mean()
                    / len(recent_game_ids)
                )
            else:
                rotation_stability = 0.0

            lineup_familiarity = float(
                rotation["recent_games_played"].mean() / recent_team_games
                if recent_team_games > 0
                else 0.0
            )
            expected_missing_starter_value = float(
                (starters[value_column] * (1.0 - starters["availability_score"])).sum()
            )
            expected_missing_rotation_value = float(
                (rotation[value_column] * (1.0 - rotation["availability_score"])).sum()
            )
            available_top8_players = int((rotation["availability_score"] >= 0.5).sum())

            rows.append(
                {
                    "game_id": game_id,
                    "team_idx": team_idx,
                    "lineup_feature_stack_version": LINEUP_FEATURE_VERSION,
                    "expected_starter_continuity": round(starter_continuity, 4),
                    "expected_top8_continuity": round(top8_continuity, 4),
                    "projected_minutes_concentration": round(projected_minutes_concentration, 4),
                    "bench_depth_quality": round(bench_depth_quality, 4),
                    "rotation_stability": round(rotation_stability, 4),
                    "lineup_familiarity": round(lineup_familiarity, 4),
                    "expected_missing_starter_value": round(expected_missing_starter_value, 4),
                    "expected_missing_rotation_value": round(expected_missing_rotation_value, 4),
                    "projected_available_starter_value": round(starter_value, 4),
                    "projected_available_rotation_value": round(total_rotation_value, 4),
                    "available_top8_players": available_top8_players,
                }
            )

    return pd.DataFrame(rows).sort_values(["game_id", "team_idx"]).reset_index(drop=True)


def main() -> None:
    """Build and save lineup/rotation features from processed projected availability."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    player_logs = pd.read_parquet(PROCESSED_DIR / "player_game_logs" / "player_game_logs.parquet")
    projected = pd.read_parquet(
        PROCESSED_DIR / "projected_availability" / "projected_availability.parquet"
    )
    features = build_lineup_features(games, player_logs, projected)

    out_dir = PROCESSED_DIR / "lineup_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "lineup_features.parquet", index=False)
    logger.info("Saved lineup features rows=%d", len(features))


if __name__ == "__main__":
    main()
