"""Fetch NBA game data via nba_api and save as processed Parquet.

Usage:
    python -m src.data.fetch_games                  # fetch all configured seasons
    python -m src.data.fetch_games --season 2024-25  # fetch single season
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yaml
from nba_api.stats.endpoints import leaguegamefinder

from src.anonymization.team_mapping import team_abbr_to_idx
from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = RAW_DIR / "nba_api"
REQUEST_DELAY = 1.0
REGULAR_SEASON = "Regular Season"
PLAYOFFS = "Playoffs"


def _load_seasons_config() -> list[str]:
    """Load season list from data_sources.yaml."""
    with open(CONFIGS_DIR / "data_sources.yaml") as f:
        config = yaml.safe_load(f)
    start = config["seasons"]["start"]
    end = config["seasons"]["end"]

    # Generate season strings from start to end
    start_year = int(start.split("-")[0])
    end_year = int(end.split("-")[0])
    seasons = []
    for year in range(start_year, end_year + 1):
        seasons.append(f"{year}-{str(year + 1)[-2:]}")
    return seasons


def _load_season_types_config() -> list[str]:
    """Load configured season types, defaulting to regular season for old configs."""
    with open(CONFIGS_DIR / "data_sources.yaml") as f:
        config = yaml.safe_load(f)
    return list(config.get("season_types", [REGULAR_SEASON]))


def _season_type_slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "_")


def _season_cache_path(season: str, season_type: str) -> Path:
    season_slug = season.replace("-", "_")
    if season_type == REGULAR_SEASON:
        return CACHE_DIR / f"games_{season_slug}.parquet"
    return CACHE_DIR / f"games_{season_slug}_{_season_type_slug(season_type)}.parquet"


def _infer_season_type_from_game_id(game_id: object) -> str:
    return PLAYOFFS if str(game_id)[:3] == "004" else REGULAR_SEASON


def _ensure_season_type_column(
    frame: pd.DataFrame,
    *,
    season_type: str | None = None,
) -> pd.DataFrame:
    """Attach a normalized season_type column to raw nba_api rows."""
    result = frame.copy()
    if result.empty:
        return result
    if season_type is not None:
        result["season_type"] = season_type
    elif "season_type" not in result.columns:
        result["season_type"] = result["GAME_ID"].map(_infer_season_type_from_game_id)
    else:
        result["season_type"] = result["season_type"].fillna(REGULAR_SEASON).astype(str)
    return result


def fetch_season_games(season: str, season_type: str = REGULAR_SEASON) -> pd.DataFrame:
    """Fetch games for a single season and season type from nba_api.

    Uses caching — skips API call if Parquet cache exists.

    Args:
        season: Season string, e.g. "2024-25".

    Returns:
        DataFrame with one row per team per game.
    """
    cache_path = _season_cache_path(season, season_type)

    if cache_path.exists():
        logger.info("Cache hit for season %s %s: %s", season, season_type, cache_path)
        return _ensure_season_type_column(pd.read_parquet(cache_path), season_type=season_type)

    logger.info("Fetching season %s %s from nba_api...", season, season_type)
    time.sleep(REQUEST_DELAY)

    gf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    df = gf.get_data_frames()[0]

    if df.empty:
        logger.warning("No games returned for season %s %s", season, season_type)
        return df

    df = _ensure_season_type_column(df, season_type=season_type)

    logger.info("Fetched %d team-game rows for season %s %s", len(df), season, season_type)

    # Cache raw response
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached to %s", cache_path)

    return df


def build_games_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw nba_api output into the clean games table.

    The raw data has one row per team per game. We pivot to one row per game
    with home and away team info.

    Args:
        raw_df: Combined raw data from all seasons.

    Returns:
        Games table with columns: game_id, date, season, home_team_idx,
        away_team_idx, home_score, away_score, home_win.
    """
    df = _ensure_season_type_column(raw_df)

    # Parse home vs away from MATCHUP column
    # Home games: "BOS vs. NYK", Away games: "BOS @ NYK"
    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    # Merge home and away on GAME_ID
    games = home.merge(
        away[["GAME_ID", "TEAM_ABBREVIATION", "PTS", "season_type"]],
        on="GAME_ID",
        suffixes=("_home", "_away"),
    )

    # Map team abbreviations to anonymous indices
    games["home_team_idx"] = games["TEAM_ABBREVIATION_home"].map(
        lambda x: team_abbr_to_idx(x)
    )
    games["away_team_idx"] = games["TEAM_ABBREVIATION_away"].map(
        lambda x: team_abbr_to_idx(x)
    )

    # Build clean table
    result = pd.DataFrame({
        "game_id": games["GAME_ID"],
        "date": pd.to_datetime(games["GAME_DATE"]),
        "season": games["SEASON_ID"].str[-4:].astype(int).apply(
            lambda y: f"{y}-{str(y + 1)[-2:]}"
        ),
        "season_type": games["season_type_home"].fillna(games["season_type_away"]),
        "home_team_idx": games["home_team_idx"],
        "away_team_idx": games["away_team_idx"],
        "home_score": games["PTS_home"].astype(int),
        "away_score": games["PTS_away"].astype(int),
        "home_win": (games["PTS_home"] > games["PTS_away"]).astype(int),
    })

    result = result.sort_values("date").reset_index(drop=True)
    return result


def build_team_game_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw nba_api output into team game logs.

    One row per team per game with stats.

    Args:
        raw_df: Combined raw data from all seasons.

    Returns:
        Team game logs DataFrame.
    """
    df = _ensure_season_type_column(raw_df)

    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["won"] = (df["WL"] == "W").astype(int)
    df["date"] = pd.to_datetime(df["GAME_DATE"])

    # Extract opponent abbreviation from matchup
    df["opponent_abbr"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")

    # Map to anonymous indices
    df["team_idx"] = df["TEAM_ABBREVIATION"].map(lambda x: team_abbr_to_idx(x))
    df["opponent_team_idx"] = df["opponent_abbr"].map(lambda x: team_abbr_to_idx(x))

    # Season string
    df["season"] = df["SEASON_ID"].str[-4:].astype(int).apply(
        lambda y: f"{y}-{str(y + 1)[-2:]}"
    )

    # Compute derived stats
    # We need opponent points — get from the other row for the same game
    game_pts = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["points_for"] = df["PTS"].astype(int)
    df["points_against"] = (game_pts - df["PTS"]).astype(int)
    df["point_diff"] = df["points_for"] - df["points_against"]

    # Approximate advanced stats from available box score data
    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    fga = df["FGA"].astype(float)
    fta = df["FTA"].astype(float)
    pts = df["PTS"].astype(float)
    df["ts_pct"] = (pts / (2 * (fga + 0.44 * fta))).round(4)
    df["ts_pct"] = df["ts_pct"].fillna(0)

    # Effective FG% = (FGM + 0.5 * FG3M) / FGA
    fgm = df["FGM"].astype(float)
    fg3m = df["FG3M"].astype(float)
    df["efg_pct"] = ((fgm + 0.5 * fg3m) / fga).round(4)
    df["efg_pct"] = df["efg_pct"].fillna(0)

    # Turnover % approximation = TOV / (FGA + 0.44 * FTA + TOV)
    tov = df["TOV"].astype(float)
    df["turnover_pct"] = (tov / (fga + 0.44 * fta + tov)).round(4)
    df["turnover_pct"] = df["turnover_pct"].fillna(0)

    # Free Throw Rate = FTA / FGA
    df["free_throw_rate"] = (fta / fga).round(4)
    df["free_throw_rate"] = df["free_throw_rate"].fillna(0)

    # Assist % approximation = AST / FGM
    ast = df["AST"].astype(float)
    df["assist_pct"] = (ast / fgm).round(4)
    df["assist_pct"] = df["assist_pct"].fillna(0)

    # Rebound rates — offensive and defensive
    oreb = df["OREB"].astype(float)
    dreb = df["DREB"].astype(float)
    total_reb = df["REB"].astype(float)
    df["off_rebound_pct"] = (oreb / total_reb).round(4).fillna(0)
    df["def_rebound_pct"] = (dreb / total_reb).round(4).fillna(0)

    # Steal and block rates (per 100 possessions approximation)
    df["steal_pct"] = (df["STL"].astype(float) / (fga + 0.44 * fta + tov)).round(4).fillna(0)
    df["block_pct"] = (df["BLK"].astype(float) / fga).round(4).fillna(0)

    # --- Advanced stats: pace and ratings ---
    # Pace ≈ possessions per 48 minutes
    # Possessions (team) ≈ FGA + 0.44 * FTA - OREB + TOV
    possessions = fga + 0.44 * fta - oreb + tov
    possessions = possessions.clip(lower=1)  # avoid division by zero

    # Minutes played (team total, e.g. 240 for a normal game)
    minutes = df["MIN"].astype(float).clip(lower=1)
    df["pace"] = ((possessions / minutes) * 48.0).round(2)

    # Offensive rating = points scored per 100 possessions
    df["off_rating"] = ((pts / possessions) * 100.0).round(2)

    # Defensive rating = points allowed per 100 possessions
    # Need opponent possessions — approximate using opponent's stats from same game
    opp_stats = (
        df.groupby("GAME_ID")
        .agg({"FGA": "sum", "FTA": "sum", "OREB": "sum", "TOV": "sum"})
        .rename(
            columns={
                "FGA": "total_fga",
                "FTA": "total_fta",
                "OREB": "total_oreb",
                "TOV": "total_tov",
            }
        )
    )
    df = df.merge(opp_stats, on="GAME_ID", how="left")
    opp_possessions = (
        (df["total_fga"] - fga)
        + 0.44 * (df["total_fta"] - fta)
        - (df["total_oreb"] - oreb)
        + (df["total_tov"] - tov)
    ).clip(lower=1)
    df["def_rating"] = ((df["points_against"].astype(float) / opp_possessions) * 100.0).round(2)

    # Net rating = off_rating - def_rating
    df["net_rating"] = (df["off_rating"] - df["def_rating"]).round(2)

    # Rename raw columns before selection
    df = df.rename(columns={
        "GAME_ID": "game_id",
        "FG_PCT": "fg_pct",
        "PLUS_MINUS": "plus_minus",
    })

    # Select final columns
    result = df[[
        "game_id", "date", "season", "season_type", "team_idx", "opponent_team_idx",
        "is_home", "won", "points_for", "points_against", "point_diff",
        "fg_pct", "ts_pct", "efg_pct", "turnover_pct",
        "off_rebound_pct", "def_rebound_pct",
        "free_throw_rate", "assist_pct", "steal_pct", "block_pct",
        "plus_minus", "pace", "off_rating", "def_rating", "net_rating",
    ]].copy()

    result = result.sort_values(["date", "game_id"]).reset_index(drop=True)
    return result


def main():
    """Main entry point: fetch all seasons, build games table and team game logs."""
    parser = argparse.ArgumentParser(description="Fetch NBA game data")
    parser.add_argument("--season", type=str, help="Single season to fetch (e.g., 2024-25)")
    parser.add_argument(
        "--season-type",
        action="append",
        choices=[REGULAR_SEASON, PLAYOFFS],
        help=(
            "Season type to fetch. Repeat for multiple values. "
            "Defaults to configs/data_sources.yaml season_types."
        ),
    )
    args = parser.parse_args()

    setup_logging()

    if args.season:
        seasons = [args.season]
    else:
        seasons = _load_seasons_config()
    season_types = args.season_type if args.season_type else _load_season_types_config()

    logger.info(
        "Fetching %d seasons across season types %s: %s to %s",
        len(seasons),
        season_types,
        seasons[0],
        seasons[-1],
    )

    # Fetch all seasons
    all_raw = []
    for i, season in enumerate(seasons):
        logger.info("=== Season %d/%d: %s ===", i + 1, len(seasons), season)
        for season_type in season_types:
            try:
                df = fetch_season_games(season, season_type=season_type)
                if not df.empty:
                    all_raw.append(df)
            except Exception as e:
                logger.error("Failed to fetch season %s %s: %s", season, season_type, e)
                continue

    if not all_raw:
        logger.error("No data fetched. Aborting.")
        return

    raw_combined = pd.concat(all_raw, ignore_index=True)
    logger.info("Total raw rows: %d", len(raw_combined))

    # Build games table
    logger.info("Building games table...")
    games = build_games_table(raw_combined)
    games_path = PROCESSED_DIR / "games.parquet"
    games_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(games_path, index=False)
    logger.info("Games table: %d rows saved to %s", len(games), games_path)

    # Build team game logs
    logger.info("Building team game logs...")
    logs = build_team_game_logs(raw_combined)
    logs_path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(logs_path, index=False)
    logger.info("Team game logs: %d rows saved to %s", len(logs), logs_path)

    # Summary
    n_seasons = games["season"].nunique()
    n_games = len(games)
    home_win_pct = games["home_win"].mean()
    logger.info("Summary: %d seasons, %d games, home win rate: %.1f%%",
                n_seasons, n_games, home_win_pct * 100)


if __name__ == "__main__":
    main()
