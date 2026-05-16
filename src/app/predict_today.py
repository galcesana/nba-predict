"""Scripts to fetch live NBA schedules and run the prediction pipeline.

Usage:
    python -m src.app.predict_today
    python -m src.app.predict_today --date 2026-05-16
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scoreboardv3

from src.models.predict import PredictionPipeline
from src.utils.logging import setup_logging
from src.utils.paths import DATA_DIR, PREDICTIONS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)
DEFAULT_FORECAST_WINDOW_DAYS = 7


def _load_team_mapping() -> dict[str, int]:
    with open(DATA_DIR / "mappings" / "team_to_idx.json") as f:
        return json.load(f)


def _season_from_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _fetch_schedule_v3(date_str: str, team_mapping: dict[str, int]) -> pd.DataFrame:
    sb = scoreboardv3.ScoreboardV3(game_date=date_str)
    scoreboard = sb.get_dict().get("scoreboard", {})
    games = scoreboard.get("games", [])

    if not games:
        logger.info("No games scheduled for %s.", date_str)
        return pd.DataFrame()

    schedule_rows = []
    season_str = _season_from_date(date_str)

    for game in games:
        home_team = game.get("homeTeam", {})
        away_team = game.get("awayTeam", {})
        home_abbr = str(home_team.get("teamTricode", "")).upper()
        away_abbr = str(away_team.get("teamTricode", "")).upper()

        if home_abbr in team_mapping and away_abbr in team_mapping:
            schedule_rows.append(
                {
                    "game_id": str(game["gameId"]),
                    "date": date_str,
                    "season": season_str,
                    "home_team_idx": team_mapping[home_abbr],
                    "away_team_idx": team_mapping[away_abbr],
                    "game_label": game.get("gameLabel"),
                    "game_sub_label": game.get("gameSubLabel"),
                    "series_text": game.get("seriesText"),
                    "game_status_text": game.get("gameStatusText"),
                    "game_time_utc": game.get("gameTimeUTC"),
                    "game_time_et": game.get("gameEt"),
                    "game_code": game.get("gameCode"),
                }
            )

    logger.info("Found %d scheduled games via ScoreboardV3.", len(schedule_rows))
    return pd.DataFrame(schedule_rows)


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
            games.append(
                {
                    "game_id": str(row["GAME_ID"]),
                    "date": date_str,
                    "season": season_str,
                    "home_team_idx": team_mapping[home_abbr],
                    "away_team_idx": team_mapping[away_abbr],
                    "game_status_text": row.get("GAME_STATUS_TEXT"),
                    "game_code": row.get("GAMECODE"),
                }
            )

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


def forecast_window_dates(start_date: str, days: int = DEFAULT_FORECAST_WINDOW_DAYS) -> list[str]:
    """Return consecutive forecast dates starting from start_date."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days)]


def _load_historical_inputs(
    historical_games: pd.DataFrame | None = None,
    historical_team_logs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading historical games and team logs...")
    hist_games = historical_games
    if hist_games is None:
        hist_games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    hist_logs = historical_team_logs
    if hist_logs is None:
        hist_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")

    hist_games = hist_games.copy()
    hist_logs = hist_logs.copy()
    hist_games["date"] = pd.to_datetime(hist_games["date"])
    hist_logs["date"] = pd.to_datetime(hist_logs["date"])
    return hist_games, hist_logs


def _predict_for_schedule(
    target_games: pd.DataFrame,
    *,
    date_str: str,
    pipeline: PredictionPipeline,
    historical_games: pd.DataFrame,
    historical_team_logs: pd.DataFrame,
) -> list[dict]:
    target_games = target_games.copy()
    target_games["date"] = pd.to_datetime(target_games["date"])

    cutoff = pd.Timestamp(date_str)
    hist_games = historical_games[historical_games["date"] < cutoff].copy()
    hist_logs = historical_team_logs[historical_team_logs["date"] < cutoff].copy()

    results = pipeline.predict_games(target_games, hist_games, hist_logs)
    metadata_columns = [
        "game_id",
        "date",
        "game_label",
        "game_sub_label",
        "series_text",
        "game_status_text",
        "game_time_utc",
        "game_time_et",
        "game_code",
    ]
    metadata_lookup = (
        target_games[[col for col in metadata_columns if col in target_games.columns]]
        .drop_duplicates("game_id")
        .set_index("game_id")
        .to_dict("index")
    )

    enriched_results = []
    for result in results:
        game_id = result["game_id"]
        meta = metadata_lookup.get(game_id, {})
        enriched = dict(result)
        enriched["game_date"] = date_str
        if "game_time_utc" in meta:
            enriched["game_time_utc"] = meta.get("game_time_utc")
        if "game_time_et" in meta:
            enriched["game_time_et"] = meta.get("game_time_et")
        if "game_status_text" in meta:
            enriched["game_status_text"] = meta.get("game_status_text")
        if "game_label" in meta:
            enriched["game_label"] = meta.get("game_label")
        if "game_sub_label" in meta:
            enriched["game_sub_label"] = meta.get("game_sub_label")
        if "series_text" in meta:
            enriched["series_text"] = meta.get("series_text")
        if "game_code" in meta:
            enriched["game_code"] = meta.get("game_code")
        enriched_results.append(enriched)

    return enriched_results


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
    hist_games, hist_logs = _load_historical_inputs(
        historical_games=historical_games,
        historical_team_logs=historical_team_logs,
    )

    if pipeline is None:
        pipeline = PredictionPipeline()

    results = _predict_for_schedule(
        target_games,
        date_str=date_str,
        pipeline=pipeline,
        historical_games=hist_games,
        historical_team_logs=hist_logs,
    )

    output = {
        "date": date_str,
        "slate_type": "day",
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


def generate_predictions_for_window(
    start_date: str,
    *,
    days: int = DEFAULT_FORECAST_WINDOW_DAYS,
    output_dir: Path | None = None,
    schedule_fetcher: Callable[[str], pd.DataFrame] = fetch_schedule,
    pipeline: PredictionPipeline | None = None,
    historical_games: pd.DataFrame | None = None,
    historical_team_logs: pd.DataFrame | None = None,
) -> tuple[dict, Path] | None:
    """Generate and save predictions for an upcoming multi-day window."""
    forecast_dates = forecast_window_dates(start_date, days=days)
    hist_games, hist_logs = _load_historical_inputs(
        historical_games=historical_games,
        historical_team_logs=historical_team_logs,
    )

    if pipeline is None:
        pipeline = PredictionPipeline()

    all_predictions: list[dict] = []
    dates_with_games: list[dict[str, int | str]] = []

    for date_str in forecast_dates:
        target_games = schedule_fetcher(date_str)
        if target_games.empty:
            continue

        day_predictions = _predict_for_schedule(
            target_games,
            date_str=date_str,
            pipeline=pipeline,
            historical_games=hist_games,
            historical_team_logs=hist_logs,
        )
        if not day_predictions:
            continue

        all_predictions.extend(day_predictions)
        dates_with_games.append(
            {
                "date": date_str,
                "games_count": len(day_predictions),
            }
        )

    if not all_predictions:
        return None

    all_predictions.sort(
        key=lambda pred: (pred.get("game_date", start_date), pred.get("game_id", ""))
    )
    generated_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    output = {
        "date": start_date,
        "slate_type": "week",
        "window_start": forecast_dates[0],
        "window_end": forecast_dates[-1],
        "generated_at": generated_at,
        "model_version": "ensemble_v1",
        "dates_with_games": dates_with_games,
        "predictions": all_predictions,
    }

    out_dir = output_dir or (PREDICTIONS_DIR / "daily")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{start_date}.json"

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Saved %d predictions across %d forecast dates to %s",
        len(all_predictions),
        len(dates_with_games),
        out_file,
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
