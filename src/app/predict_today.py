"""Script to fetch today's NBA games and run the prediction pipeline.

Usage:
    python -m src.app.predict_today
    python -m src.app.predict_today --date 2026-05-16
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scoreboardv3

from src.models.predict import PredictionPipeline
from src.utils.logging import setup_logging
from src.utils.paths import DATA_DIR, PROCESSED_DIR, PREDICTIONS_DIR

logger = logging.getLogger(__name__)


def _load_team_mapping() -> dict[str, int]:
    with open(DATA_DIR / "mappings" / "team_to_idx.json") as f:
        return json.load(f)


def _season_from_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _fetch_schedule_v3(date_str: str, team_mapping: dict[str, int]) -> pd.DataFrame:
    sb = scoreboardv3.ScoreboardV3(game_date=date_str)
    line_scores = sb.line_score.get_data_frame()

    if line_scores.empty:
        logger.info("No games scheduled for %s.", date_str)
        return pd.DataFrame()

    games = []
    season_str = _season_from_date(date_str)

    for game_id, group in line_scores.groupby("gameId", sort=False):
        teams = group.reset_index(drop=True)
        if len(teams) < 2:
            continue

        home_abbr = str(teams.loc[0, "teamTricode"]).upper()
        away_abbr = str(teams.loc[1, "teamTricode"]).upper()

        if home_abbr in team_mapping and away_abbr in team_mapping:
            games.append({
                "game_id": str(game_id),
                "date": date_str,
                "season": season_str,
                "home_team_idx": team_mapping[home_abbr],
                "away_team_idx": team_mapping[away_abbr],
            })

    logger.info("Found %d scheduled games via ScoreboardV3.", len(games))
    return pd.DataFrame(games)


def _fetch_schedule_v2(date_str: str, team_mapping: dict[str, int]) -> pd.DataFrame:
    sb = scoreboardv2.ScoreboardV2(game_date=date_str)
    df = sb.get_data_frames()[0]

    if len(df) == 0:
        logger.info("No games scheduled for %s.", date_str)
        return pd.DataFrame()

    games = []
    for _, row in df.iterrows():
        game_code = str(row["GAMECODE"]).split("/")[-1]
        away_abbr = game_code[:3].upper()
        home_abbr = game_code[3:].upper()
        season_year = int(row["SEASON"])
        season_str = f"{season_year}-{(season_year + 1) % 100:02d}"

        if home_abbr in team_mapping and away_abbr in team_mapping:
            games.append({
                "game_id": str(row["GAME_ID"]),
                "date": date_str,
                "season": season_str,
                "home_team_idx": team_mapping[home_abbr],
                "away_team_idx": team_mapping[away_abbr],
            })

    logger.info("Found %d scheduled games via ScoreboardV2.", len(games))
    return pd.DataFrame(games)


def fetch_schedule(date_str: str) -> pd.DataFrame:
    """Fetch the schedule for a given date."""
    logger.info("Fetching schedule for %s...", date_str)

    team_mapping = _load_team_mapping()

    try:
        return _fetch_schedule_v3(date_str, team_mapping)
    except Exception as exc:
        logger.warning("ScoreboardV3 fetch failed for %s: %s", date_str, exc)

    return _fetch_schedule_v2(date_str, team_mapping)


def generate_predictions_for_date(
    date_str: str,
    output_dir: Path | None = None,
    schedule_fetcher: Callable[[str], pd.DataFrame] = fetch_schedule,
    pipeline: PredictionPipeline | None = None,
    historical_games: pd.DataFrame | None = None,
    historical_team_logs: pd.DataFrame | None = None,
) -> tuple[dict, Path] | None:
    """Generate and save daily predictions for a specific date."""
    target_games = schedule_fetcher(date_str)
    if target_games.empty:
        return None
    target_games = target_games.copy()
    target_games["date"] = pd.to_datetime(target_games["date"])

    logger.info("Loading historical games and team logs...")
    hist_games = historical_games
    if hist_games is None:
        hist_games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    hist_logs = historical_team_logs
    if hist_logs is None:
        hist_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")

    # Filter out same-day or future games to avoid leakage.
    cutoff = pd.Timestamp(date_str)
    hist_games = hist_games.copy()
    hist_logs = hist_logs.copy()
    hist_games["date"] = pd.to_datetime(hist_games["date"])
    hist_logs["date"] = pd.to_datetime(hist_logs["date"])
    hist_games = hist_games[hist_games["date"] < cutoff].copy()
    hist_logs = hist_logs[hist_logs["date"] < cutoff].copy()

    if pipeline is None:
        pipeline = PredictionPipeline()

    results = pipeline.predict_games(target_games, hist_games, hist_logs)

    output = {
        "date": date_str,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": "ensemble_v1",
        "predictions": results,
    }

    out_dir = output_dir or (PREDICTIONS_DIR / "daily")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved daily predictions to %s", out_file)

    for res in results:
        logger.info(
            "Game %s | Home Win Prob: %.1f%% | Confidence: %s",
            res["game_id"],
            res["home_win_probability"] * 100,
            res["confidence_bucket"],
        )

    return output, out_file


def main(argv: list[str] | None = None):
    setup_logging()

    parser = argparse.ArgumentParser(description="Predict today's NBA games.")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Date to predict for in YYYY-MM-DD format.",
    )
    args = parser.parse_args(argv)

    generate_predictions_for_date(args.date)


if __name__ == "__main__":
    main()
