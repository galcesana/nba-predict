"""Build lineup and rotation features from projected availability and player logs."""

from __future__ import annotations

import logging

import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

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


def _top_players_from_game(
    player_logs: pd.DataFrame,
    *,
    team_idx: int,
    game_id: str,
    top_n: int,
) -> set[int]:
    game_rows = player_logs[
        (player_logs["team_idx"] == int(team_idx)) & (player_logs["game_id"] == str(game_id))
    ].copy()
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
        return pd.DataFrame(columns=["game_id", "team_idx", *LINEUP_FEATURE_COLS])

    logs = player_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    availability = projected_availability.copy()
    if "date" in availability.columns:
        availability["date"] = pd.to_datetime(availability["date"])

    rows: list[dict[str, object]] = []
    for _, game in games.sort_values("date").iterrows():
        game_id = str(game["game_id"])
        game_date = pd.Timestamp(game["date"])
        for team_idx in (int(game["home_team_idx"]), int(game["away_team_idx"])):
            team_projection = availability[
                (availability["game_id"] == game_id) & (availability["team_idx"] == team_idx)
            ].copy()
            if team_projection.empty:
                continue

            team_projection["effective_role_value"] = (
                team_projection["role_score"] * team_projection["availability_score"]
            )
            team_projection["effective_minutes"] = (
                team_projection["expected_minutes"] * team_projection["availability_score"]
            )
            team_projection = team_projection.sort_values(
                ["effective_role_value", "role_score", "player_id"],
                ascending=[False, False, True],
            )

            starters = team_projection.head(starter_size).copy()
            rotation = team_projection.head(rotation_size).copy()

            prior_logs = logs[(logs["team_idx"] == team_idx) & (logs["date"] < game_date)].copy()
            prior_game_ids = (
                prior_logs.sort_values(["date", "game_id"])["game_id"].drop_duplicates()
            )
            last_game_id = prior_game_ids.iloc[-1] if not prior_game_ids.empty else None
            recent_game_ids = list(prior_game_ids.tail(recent_team_games))

            prior_top5 = (
                _top_players_from_game(
                    logs,
                    team_idx=team_idx,
                    game_id=last_game_id,
                    top_n=starter_size,
                )
                if last_game_id is not None
                else set()
            )
            prior_top8 = (
                _top_players_from_game(
                    logs,
                    team_idx=team_idx,
                    game_id=last_game_id,
                    top_n=rotation_size,
                )
                if last_game_id is not None
                else set()
            )

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
                    prior_logs[prior_logs["game_id"].isin(recent_game_ids)]
                    .groupby("player_id")["game_id"]
                    .nunique()
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
                (starters["role_score"] * (1.0 - starters["availability_score"])).sum()
            )
            expected_missing_rotation_value = float(
                (rotation["role_score"] * (1.0 - rotation["availability_score"])).sum()
            )
            available_top8_players = int((rotation["availability_score"] >= 0.5).sum())

            rows.append(
                {
                    "game_id": game_id,
                    "team_idx": team_idx,
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
