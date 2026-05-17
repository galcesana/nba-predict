"""Fetch historical player game logs and save a processed player-game table.

Usage:
    python -m src.data.fetch_player_logs
    python -m src.data.fetch_player_logs --season 2024-25
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import yaml
from nba_api.stats.endpoints import leaguegamelog

from src.anonymization.player_mapping import ensure_player_id_mapping
from src.anonymization.team_mapping import team_abbr_to_idx
from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = RAW_DIR / "nba_api"
REQUEST_DELAY = 1.0


def _load_seasons_config() -> list[str]:
    with open(CONFIGS_DIR / "data_sources.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    start = config["seasons"]["start"]
    end = config["seasons"]["end"]

    start_year = int(start.split("-")[0])
    end_year = int(end.split("-")[0])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def fetch_season_player_logs(season: str) -> pd.DataFrame:
    """Fetch all regular-season player game logs for a season with Parquet caching."""
    cache_path = CACHE_DIR / f"player_logs_{season.replace('-', '_')}.parquet"
    if cache_path.exists():
        logger.info("Cache hit for player logs %s: %s", season, cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching player logs for season %s from nba_api...", season)
    time.sleep(REQUEST_DELAY)

    endpoint = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
    )
    frame = endpoint.get_data_frames()[0]
    if frame.empty:
        logger.warning("No player logs returned for season %s", season)
        return frame

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    logger.info("Cached %d player-game rows to %s", len(frame), cache_path)
    return frame


def build_player_game_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw player-game rows into the project's processed schema."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "date",
                "season",
                "team_idx",
                "opponent_team_idx",
                "player_id",
                "player_idx",
                "player_name",
                "is_home",
                "won",
                "minutes",
                "points",
                "rebounds",
                "assists",
                "steals",
                "blocks",
                "turnovers",
                "fg_pct",
                "fg3_pct",
                "ft_pct",
                "plus_minus",
                "fantasy_points",
                "available_for_game",
            ]
        )

    frame = raw_df.copy()
    frame["date"] = pd.to_datetime(frame["GAME_DATE"])
    frame["is_home"] = frame["MATCHUP"].str.contains("vs.").astype(int)
    frame["won"] = frame["WL"].eq("W").astype(int)
    frame["opponent_abbr"] = frame["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")

    frame["team_idx"] = frame["TEAM_ABBREVIATION"].map(team_abbr_to_idx)
    frame["opponent_team_idx"] = frame["opponent_abbr"].map(team_abbr_to_idx)
    frame["player_id"] = frame["PLAYER_ID"].astype(int)
    player_mapping = ensure_player_id_mapping(frame["player_id"].tolist())
    frame["player_idx"] = frame["player_id"].map(player_mapping)
    frame["season"] = frame["SEASON_ID"].str[-4:].astype(int).apply(
        lambda year: f"{year}-{str(year + 1)[-2:]}"
    )
    frame["available_for_game"] = 1

    result = pd.DataFrame(
        {
            "game_id": frame["GAME_ID"].astype(str),
            "date": frame["date"],
            "season": frame["season"],
            "team_idx": frame["team_idx"].astype(int),
            "opponent_team_idx": frame["opponent_team_idx"].astype(int),
            "player_id": frame["player_id"],
            "player_idx": frame["player_idx"].astype(int),
            "player_name": frame["PLAYER_NAME"].astype(str),
            "is_home": frame["is_home"].astype(int),
            "won": frame["won"].astype(int),
            "minutes": pd.to_numeric(frame["MIN"], errors="coerce").fillna(0.0),
            "points": pd.to_numeric(frame["PTS"], errors="coerce").fillna(0.0),
            "rebounds": pd.to_numeric(frame["REB"], errors="coerce").fillna(0.0),
            "assists": pd.to_numeric(frame["AST"], errors="coerce").fillna(0.0),
            "steals": pd.to_numeric(frame["STL"], errors="coerce").fillna(0.0),
            "blocks": pd.to_numeric(frame["BLK"], errors="coerce").fillna(0.0),
            "turnovers": pd.to_numeric(frame["TOV"], errors="coerce").fillna(0.0),
            "fg_pct": pd.to_numeric(frame["FG_PCT"], errors="coerce").fillna(0.0),
            "fg3_pct": pd.to_numeric(frame["FG3_PCT"], errors="coerce").fillna(0.0),
            "ft_pct": pd.to_numeric(frame["FT_PCT"], errors="coerce").fillna(0.0),
            "plus_minus": pd.to_numeric(frame["PLUS_MINUS"], errors="coerce").fillna(0.0),
            "fantasy_points": pd.to_numeric(frame["FANTASY_PTS"], errors="coerce").fillna(0.0),
            "available_for_game": frame["available_for_game"].astype(int),
        }
    )
    return result.sort_values(["date", "game_id", "team_idx", "player_id"]).reset_index(drop=True)


def main() -> None:
    """Fetch player logs for configured seasons and save a processed table."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Fetch NBA player game logs.")
    parser.add_argument("--season", type=str, default=None, help="Single season to fetch.")
    args = parser.parse_args()

    seasons = [args.season] if args.season else _load_seasons_config()
    raw_frames = [fetch_season_player_logs(season) for season in seasons]
    raw_frames = [frame for frame in raw_frames if not frame.empty]
    if not raw_frames:
        logger.warning("No player logs were fetched.")
        return

    player_logs = build_player_game_logs(pd.concat(raw_frames, ignore_index=True))
    out_dir = PROCESSED_DIR / "player_game_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "player_game_logs.parquet"
    player_logs.to_parquet(out_path, index=False)

    logger.info(
        "Saved player game logs: %d rows, %d columns to %s",
        len(player_logs),
        len(player_logs.columns),
        out_path,
    )


if __name__ == "__main__":
    main()
