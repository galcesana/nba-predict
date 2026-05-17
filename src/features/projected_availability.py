"""Build leakage-safe projected availability rows for each team-game."""

from __future__ import annotations

import logging

import pandas as pd

from src.data.player_metadata import resolve_player_name
from src.data.providers.nba_api_provider import NbaApiProvider
from src.features.player_value_features import build_player_value_features
from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

STATUS_TO_AVAILABILITY = {
    "AVAILABLE": 1.0,
    "ACTIVE": 1.0,
    "CLEAR": 1.0,
    "PROBABLE": 0.85,
    "QUESTIONABLE": 0.5,
    "GAME TIME DECISION": 0.5,
    "DOUBTFUL": 0.15,
    "OUT": 0.0,
    "NOT YET SUBMITTED": 0.75,
}

STATUS_TO_CONFIDENCE = {
    "AVAILABLE": 0.9,
    "ACTIVE": 0.9,
    "CLEAR": 0.9,
    "PROBABLE": 0.8,
    "QUESTIONABLE": 0.65,
    "GAME TIME DECISION": 0.65,
    "DOUBTFUL": 0.8,
    "OUT": 0.95,
    "NOT YET SUBMITTED": 0.2,
}

AVAILABILITY_COLUMNS = [
    "game_id",
    "date",
    "season",
    "team_idx",
    "player_id",
    "player_idx",
    "player_name",
    "status",
    "availability_score",
    "projection_confidence",
    "source_type",
    "source_timestamp",
    "report_reason",
    "recent_games_played",
    "expected_minutes",
    "player_value_score",
    "minutes_share_recent",
    "starter_rate_recent",
    "recent_role_stability",
    "role_score",
    "resolved_from_name",
]

UNRESOLVED_COLUMNS = [
    "game_id",
    "season",
    "team_idx",
    "player_name",
    "status",
    "resolution_status",
]


def _normalize_status(status: str | None) -> str:
    return str(status or "AVAILABLE").upper().strip()


def status_to_availability_score(status: str | None) -> float:
    """Map an injury-style status to a continuous availability score."""
    normalized = _normalize_status(status)
    return float(STATUS_TO_AVAILABILITY.get(normalized, 0.75))


def status_to_projection_confidence(status: str | None) -> float:
    """Map an injury-style status to a projection confidence score."""
    normalized = _normalize_status(status)
    return float(STATUS_TO_CONFIDENCE.get(normalized, 0.4))


def _season_metadata_lookup(games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    provider = NbaApiProvider()
    metadata_by_season: dict[str, pd.DataFrame] = {}
    for season in sorted(set(games["season"].astype(str))):
        from src.data.player_metadata import build_player_metadata_for_season

        metadata_by_season[season] = build_player_metadata_for_season(season, provider)
    return metadata_by_season


def _empty_availability_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=AVAILABILITY_COLUMNS)


def _empty_unresolved_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=UNRESOLVED_COLUMNS)


def _recent_team_pool(
    player_logs: pd.DataFrame,
    *,
    team_idx: int,
    game_date: pd.Timestamp,
    recent_team_games: int,
    max_players: int,
) -> pd.DataFrame:
    prior_logs = player_logs[
        (player_logs["team_idx"] == int(team_idx))
        & (pd.to_datetime(player_logs["date"]) < game_date)
    ].copy()
    if prior_logs.empty:
        return pd.DataFrame()

    recent_game_ids = (
        prior_logs.sort_values(["date", "game_id"])["game_id"]
        .drop_duplicates()
        .tail(recent_team_games)
    )
    recent_logs = prior_logs[prior_logs["game_id"].isin(recent_game_ids)].copy()
    recent_logs = recent_logs.sort_values(["date", "game_id"])
    recency_rank = recent_logs["date"].rank(method="dense", ascending=True)
    recent_logs["recency_weight"] = recency_rank / recency_rank.max()

    def _weighted_minutes(series: pd.Series) -> float:
        weights = recent_logs.loc[series.index, "recency_weight"]
        return float((series * weights).sum())

    grouped = (
        recent_logs.groupby(["player_id", "player_idx", "player_name"], as_index=False)
        .agg(
            recent_games_played=("game_id", "nunique"),
            avg_minutes=("minutes", "mean"),
            avg_fantasy_points=("fantasy_points", "mean"),
            avg_plus_minus=("plus_minus", "mean"),
            weighted_minutes=("minutes", _weighted_minutes),
            last_game_date=("date", "max"),
        )
    )
    grouped["role_score"] = grouped["weighted_minutes"] + 0.15 * grouped["avg_fantasy_points"]
    grouped["projection_confidence"] = (
        0.3
        + 0.65 * grouped["recent_games_played"].clip(upper=recent_team_games) / recent_team_games
    ).round(4)
    grouped["availability_score"] = 1.0
    grouped["status"] = "AVAILABLE"
    grouped["source_type"] = "historical_recent_role"
    grouped["source_timestamp"] = grouped["last_game_date"].apply(
        lambda value: pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return (
        grouped.sort_values("role_score", ascending=False)
        .head(max_players)
        .reset_index(drop=True)
    )


def _fallback_role_snapshot(
    player_logs: pd.DataFrame,
    *,
    team_idx: int,
    player_id: int,
    game_date: pd.Timestamp,
    recent_team_games: int,
) -> dict[str, float | int | str]:
    player_history = player_logs[
        (player_logs["team_idx"] == int(team_idx))
        & (player_logs["player_id"] == int(player_id))
        & (pd.to_datetime(player_logs["date"]) < game_date)
    ].copy()
    if player_history.empty:
        return {
            "recent_games_played": 0,
            "expected_minutes": 0.0,
            "player_value_score": 0.0,
            "minutes_share_recent": 0.0,
            "starter_rate_recent": 0.0,
            "recent_role_stability": 0.0,
            "role_score": 0.0,
            "projection_confidence": 0.35,
            "source_timestamp": None,
        }

    recent_history = (
        player_history.sort_values(["date", "game_id"])
        .tail(recent_team_games)
        .reset_index(drop=True)
    )
    recency_rank = recent_history["date"].rank(method="dense", ascending=True)
    recent_history["recency_weight"] = recency_rank / recency_rank.max()
    weighted_minutes = float(
        (recent_history["minutes"] * recent_history["recency_weight"]).sum()
    )
    avg_fantasy_points = float(recent_history["fantasy_points"].mean())
    recent_games_played = int(recent_history["game_id"].nunique())
    projection_confidence = round(
        0.25 + 0.65 * min(recent_games_played, recent_team_games) / recent_team_games,
        4,
    )
    return {
        "recent_games_played": recent_games_played,
        "expected_minutes": float(recent_history["minutes"].mean()),
        "player_value_score": weighted_minutes + 0.15 * avg_fantasy_points,
        "minutes_share_recent": float(
            recent_history["minutes"].sum() / max(recent_team_games * 240, 1)
        ),
        "starter_rate_recent": float(min(recent_games_played, 5) / max(recent_team_games, 1)),
        "recent_role_stability": float(recent_games_played / max(recent_team_games, 1)),
        "role_score": weighted_minutes + 0.15 * avg_fantasy_points,
        "projection_confidence": projection_confidence,
        "source_timestamp": pd.Timestamp(recent_history["date"].max()).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def _enrich_resolved_report_rows(
    resolved_reports: pd.DataFrame,
    *,
    games_by_id: pd.DataFrame,
    player_logs: pd.DataFrame,
    recent_team_games: int,
    player_value_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if resolved_reports.empty:
        return resolved_reports

    enriched = resolved_reports.copy()
    enriched["date"] = enriched["game_id"].map(games_by_id["date"])
    enriched["season"] = enriched["game_id"].map(games_by_id["season"])

    snapshots = []
    for _, row in enriched.iterrows():
        if player_value_lookup is not None:
            key = (row["game_id"], int(row["team_idx"]), int(row["player_id"]))
            if key in player_value_lookup.index:
                snapshot = player_value_lookup.loc[key]
                snapshots.append(
                    {
                        "recent_games_played": int(snapshot["recent_games_played"]),
                        "expected_minutes": float(snapshot["recent_minutes_avg"]),
                        "player_value_score": float(snapshot["player_value_score"]),
                        "minutes_share_recent": float(snapshot["recent_minutes_share"]),
                        "starter_rate_recent": float(snapshot["recent_starter_rate"]),
                        "recent_role_stability": float(snapshot["recent_role_stability"]),
                        "role_score": float(snapshot["player_value_score"]),
                        "projection_confidence": None,
                        "source_timestamp": None,
                    }
                )
                continue

        snapshots.append(
            _fallback_role_snapshot(
                player_logs,
                team_idx=int(row["team_idx"]),
                player_id=int(row["player_id"]),
                game_date=pd.Timestamp(row["date"]),
                recent_team_games=recent_team_games,
            )
        )

    snapshot_frame = pd.DataFrame(snapshots)
    for column in [
        "recent_games_played",
        "expected_minutes",
        "player_value_score",
        "minutes_share_recent",
        "starter_rate_recent",
        "recent_role_stability",
        "role_score",
        "projection_confidence",
        "source_timestamp",
    ]:
        if column in snapshot_frame.columns:
            enriched[column] = enriched[column].where(
                enriched[column].notna(),
                snapshot_frame[column],
            )
    return enriched


def resolve_injury_report_players(
    injury_reports: pd.DataFrame,
    metadata_by_season: dict[str, pd.DataFrame],
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve injury-report player names into player ids when possible."""
    if injury_reports.empty:
        return _empty_availability_frame(), _empty_unresolved_frame()

    game_lookup = games.set_index("game_id")[["season", "date"]].to_dict("index")
    resolved_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for _, row in injury_reports.iterrows():
        player_name = row.get("player_name")
        game_id = row.get("game_id")
        if not player_name or game_id not in game_lookup:
            continue

        season = str(game_lookup[game_id]["season"])
        metadata = metadata_by_season.get(season)
        if metadata is None or metadata.empty:
            unresolved_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "team_idx": int(row["team_idx"]),
                    "player_name": player_name,
                    "status": row.get("status"),
                    "resolution_status": "missing_season_metadata",
                }
            )
            continue

        player_id = resolve_player_name(player_name, metadata, team_idx=int(row["team_idx"]))
        if player_id is None:
            unresolved_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "team_idx": int(row["team_idx"]),
                    "player_name": player_name,
                    "status": row.get("status"),
                    "resolution_status": "unmatched_name",
                }
            )
            continue

        matched = metadata[metadata["player_id"] == int(player_id)].iloc[0]
        status = _normalize_status(str(row.get("status")))
        resolved_rows.append(
            {
                "game_id": game_id,
                "team_idx": int(row["team_idx"]),
                "player_id": int(player_id),
                "player_idx": int(matched["player_idx"]),
                "player_name": str(matched["player_name"]),
                "status": status,
                "availability_score": status_to_availability_score(status),
                "projection_confidence": status_to_projection_confidence(status),
                "source_type": "official_injury_report",
                "source_timestamp": row.get("report_generated_at"),
                "report_reason": row.get("reason"),
                "resolved_from_name": player_name,
                "recent_games_played": None,
                "role_score": None,
                "expected_minutes": None,
            }
        )

    return (
        pd.DataFrame(resolved_rows, columns=AVAILABILITY_COLUMNS),
        pd.DataFrame(unresolved_rows, columns=UNRESOLVED_COLUMNS),
    )


def build_projected_availability(
    games: pd.DataFrame,
    player_logs: pd.DataFrame,
    *,
    injury_reports: pd.DataFrame | None = None,
    metadata_by_season: dict[str, pd.DataFrame] | None = None,
    player_value_features: pd.DataFrame | None = None,
    recent_team_games: int = 10,
    max_players: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build projected availability rows and an unresolved injury-entity audit table."""
    if games.empty:
        return _empty_availability_frame(), _empty_unresolved_frame()

    logs = player_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games_by_id = games.set_index("game_id")[["date", "season"]]
    if player_value_features is None:
        player_value_features = build_player_value_features(
            games,
            logs,
            recent_team_games=recent_team_games,
            max_players=max_players,
        )
    player_value_lookup = None
    player_value_groups: dict[tuple[str, int], pd.DataFrame] = {}
    if player_value_features is not None and not player_value_features.empty:
        player_values = player_value_features.copy()
        player_values["date"] = pd.to_datetime(player_values["date"])
        player_value_lookup = player_values.set_index(["game_id", "team_idx", "player_id"])
        for (game_id, team_idx), group in player_values.groupby(
            ["game_id", "team_idx"],
            sort=False,
        ):
            player_value_groups[(str(game_id), int(team_idx))] = group.copy()

    availability_rows: list[dict[str, object]] = []
    for _, game in games.sort_values("date").iterrows():
        game_id = str(game["game_id"])
        game_date = pd.Timestamp(game["date"])
        for team_idx in (int(game["home_team_idx"]), int(game["away_team_idx"])):
            value_pool = player_value_groups.get((game_id, team_idx))
            if value_pool is not None and not value_pool.empty:
                baseline_rows = []
                for _, player in value_pool.sort_values(
                    ["rotation_rank", "player_value_score", "player_id"]
                ).iterrows():
                    baseline_rows.append(
                        {
                            "game_id": game_id,
                            "date": game_date,
                            "season": str(game["season"]),
                            "team_idx": team_idx,
                            "player_id": int(player["player_id"]),
                            "player_idx": int(player["player_idx"]),
                            "player_name": str(player["player_name"]),
                            "status": "AVAILABLE",
                            "availability_score": 1.0,
                            "projection_confidence": float(
                                0.3 + 0.6 * player["recent_role_stability"]
                            ),
                            "source_type": "historical_recent_role",
                            "source_timestamp": (
                                pd.Timestamp(player["last_game_date"]).strftime("%Y-%m-%dT%H:%M:%SZ")
                                if pd.notna(player["last_game_date"])
                                else None
                            ),
                            "report_reason": None,
                            "recent_games_played": int(player["recent_games_played"]),
                            "expected_minutes": float(player["recent_minutes_avg"]),
                            "player_value_score": float(player["player_value_score"]),
                            "minutes_share_recent": float(player["recent_minutes_share"]),
                            "starter_rate_recent": float(player["recent_starter_rate"]),
                            "recent_role_stability": float(player["recent_role_stability"]),
                            "role_score": float(player["player_value_score"]),
                            "resolved_from_name": None,
                        }
                    )
                availability_rows.extend(baseline_rows)
                continue

            recent_pool = _recent_team_pool(
                logs,
                team_idx=team_idx,
                game_date=game_date,
                recent_team_games=recent_team_games,
                max_players=max_players,
            )
            if recent_pool.empty:
                continue

            for _, player in recent_pool.iterrows():
                availability_rows.append(
                    {
                        "game_id": game_id,
                        "date": game_date,
                        "season": str(game["season"]),
                        "team_idx": team_idx,
                        "player_id": int(player["player_id"]),
                        "player_idx": int(player["player_idx"]),
                        "player_name": str(player["player_name"]),
                        "status": "AVAILABLE",
                        "availability_score": 1.0,
                        "projection_confidence": float(player["projection_confidence"]),
                        "source_type": "historical_recent_role",
                        "source_timestamp": player["source_timestamp"],
                        "report_reason": None,
                        "recent_games_played": int(player["recent_games_played"]),
                        "expected_minutes": float(player["avg_minutes"]),
                        "player_value_score": float(player["role_score"]),
                        "minutes_share_recent": 0.0,
                        "starter_rate_recent": 0.0,
                        "recent_role_stability": float(
                            player["recent_games_played"] / max(recent_team_games, 1)
                        ),
                        "role_score": float(player["role_score"]),
                        "resolved_from_name": None,
                    }
                )

    availability = pd.DataFrame(availability_rows, columns=AVAILABILITY_COLUMNS)
    if availability.empty:
        availability = _empty_availability_frame()

    reports = injury_reports.copy() if injury_reports is not None else pd.DataFrame()
    unresolved = _empty_unresolved_frame()
    if not reports.empty:
        metadata_lookup = metadata_by_season or _season_metadata_lookup(games)
        resolved_reports, unresolved = resolve_injury_report_players(
            reports,
            metadata_lookup,
            games,
        )
        resolved_reports = _enrich_resolved_report_rows(
            resolved_reports,
            games_by_id=games_by_id,
            player_logs=logs,
            recent_team_games=recent_team_games,
            player_value_lookup=player_value_lookup,
        )

        if not resolved_reports.empty:
            if availability.empty:
                availability = resolved_reports.copy()
            else:
                availability = availability.set_index(["game_id", "team_idx", "player_id"])
                resolved_reports = resolved_reports.set_index(["game_id", "team_idx", "player_id"])

                shared_index = availability.index.intersection(resolved_reports.index)
                for column in [
                    "status",
                    "availability_score",
                    "projection_confidence",
                    "source_type",
                    "source_timestamp",
                    "report_reason",
                    "resolved_from_name",
                ]:
                    availability.loc[shared_index, column] = resolved_reports.loc[
                        shared_index,
                        column,
                    ]

                new_rows = resolved_reports.loc[
                    resolved_reports.index.difference(availability.index)
                ].copy()
                if not new_rows.empty:
                    new_rows["expected_minutes"] = new_rows["expected_minutes"].fillna(0.0)
                    new_rows["player_value_score"] = new_rows["player_value_score"].fillna(0.0)
                    new_rows["minutes_share_recent"] = new_rows["minutes_share_recent"].fillna(0.0)
                    new_rows["starter_rate_recent"] = new_rows["starter_rate_recent"].fillna(0.0)
                    new_rows["recent_role_stability"] = new_rows[
                        "recent_role_stability"
                    ].fillna(0.0)
                    new_rows["role_score"] = new_rows["role_score"].fillna(0.0)
                    availability = pd.concat([availability, new_rows], axis=0)

                availability = availability.reset_index()

    if not availability.empty:
        availability["availability_score"] = pd.to_numeric(
            availability["availability_score"], errors="coerce"
        ).fillna(0.75)
        availability["projection_confidence"] = pd.to_numeric(
            availability["projection_confidence"], errors="coerce"
        ).fillna(0.4)
        availability["expected_minutes"] = pd.to_numeric(
            availability["expected_minutes"], errors="coerce"
        ).fillna(0.0)
        availability["player_value_score"] = pd.to_numeric(
            availability["player_value_score"], errors="coerce"
        ).fillna(0.0)
        availability["minutes_share_recent"] = pd.to_numeric(
            availability["minutes_share_recent"], errors="coerce"
        ).fillna(0.0)
        availability["starter_rate_recent"] = pd.to_numeric(
            availability["starter_rate_recent"], errors="coerce"
        ).fillna(0.0)
        availability["recent_role_stability"] = pd.to_numeric(
            availability["recent_role_stability"], errors="coerce"
        ).fillna(0.0)
        availability["role_score"] = pd.to_numeric(
            availability["role_score"],
            errors="coerce",
        ).fillna(0.0)
        availability["recent_games_played"] = pd.to_numeric(
            availability["recent_games_played"], errors="coerce"
        ).fillna(0).astype(int)
        availability = availability.sort_values(
            ["date", "game_id", "team_idx", "role_score", "player_id"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    return availability, unresolved


def main() -> None:
    """Build and save a historical projected-availability table from player logs."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    player_logs = pd.read_parquet(PROCESSED_DIR / "player_game_logs" / "player_game_logs.parquet")
    injury_dir = RAW_DIR / "injuries"
    injury_reports = pd.DataFrame()
    if injury_dir.exists():
        injury_files = sorted(path for path in injury_dir.iterdir() if path.suffix == ".parquet")
        if injury_files:
            injury_reports = pd.concat(
                [pd.read_parquet(path) for path in injury_files],
                ignore_index=True,
            )

    projected, unresolved = build_projected_availability(
        games,
        player_logs,
        player_value_features=build_player_value_features(games, player_logs),
        injury_reports=injury_reports,
    )

    out_dir = PROCESSED_DIR / "projected_availability"
    out_dir.mkdir(parents=True, exist_ok=True)
    projected.to_parquet(out_dir / "projected_availability.parquet", index=False)
    unresolved.to_parquet(out_dir / "unresolved_injury_entities.parquet", index=False)
    logger.info(
        "Saved projected availability rows=%d unresolved=%d",
        len(projected),
        len(unresolved),
    )


if __name__ == "__main__":
    main()
