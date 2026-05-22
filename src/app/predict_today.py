"""Scripts to fetch live NBA schedules and run the prediction pipeline.

Usage:
    python -m src.app.predict_today
    python -m src.app.predict_today --date 2026-05-16
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from nba_api.stats.endpoints import scoreboardv2, scoreboardv3

from src.anonymization.team_mapping import idx_to_team_abbr
from src.models.predict import NEXTGEN_SHADOW_MODEL_VERSION, PredictionPipeline
from src.utils.logging import setup_logging
from src.utils.paths import DATA_DIR, PREDICTIONS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)
DEFAULT_FORECAST_WINDOW_DAYS = 7
NBA_STATS_TIMEOUT_SECONDS = 12
NBA_CDN_TIMEOUT_SECONDS = 15
NBA_CDN_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
NBA_CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


def _context_summary_from_predictions(predictions: list[dict]) -> dict[str, object]:
    """Summarize live context coverage across a generated slate."""
    if not predictions:
        return {
            "injury_live_games": 0,
            "injury_partial_games": 0,
            "injury_pending_games": 0,
            "news_live_games": 0,
            "news_partial_games": 0,
            "injury_coverage_rate": 0.0,
            "news_coverage_rate": 0.0,
            "latest_injury_report_at": None,
            "latest_news_article_at": None,
            "latest_news_collection_at": None,
        }

    injury_live_games = 0
    injury_partial_games = 0
    injury_pending_games = 0
    news_live_games = 0
    news_partial_games = 0
    latest_injury_report_at = None
    latest_news_article_at = None
    latest_news_collection_at = None

    for prediction in predictions:
        details = prediction.get("context_details", {})
        injury_mode = details.get("injury_mode")
        news_mode = details.get("news_mode")
        if injury_mode == "live":
            injury_live_games += 1
        elif injury_mode == "partial":
            injury_partial_games += 1
        elif injury_mode == "pending":
            injury_pending_games += 1
        if news_mode == "live":
            news_live_games += 1
        elif news_mode == "partial":
            news_partial_games += 1

        latest_injury_report_at = (
            max(
                latest_injury_report_at or "",
                str(details.get("injury_report_generated_at") or ""),
            )
            or None
        )
        latest_news_article_at = (
            max(
                latest_news_article_at or "",
                str(details.get("latest_article_at") or ""),
            )
            or None
        )
        latest_news_collection_at = (
            max(
                latest_news_collection_at or "",
                str(details.get("news_collected_at") or ""),
            )
            or None
        )

    total_games = len(predictions)
    return {
        "injury_live_games": injury_live_games,
        "injury_partial_games": injury_partial_games,
        "injury_pending_games": injury_pending_games,
        "news_live_games": news_live_games,
        "news_partial_games": news_partial_games,
        "injury_coverage_rate": round(
            (injury_live_games + 0.5 * injury_partial_games) / total_games, 4
        ),
        "news_coverage_rate": round(
            (news_live_games + 0.5 * news_partial_games) / total_games,
            4,
        ),
        "latest_injury_report_at": latest_injury_report_at,
        "latest_news_article_at": latest_news_article_at,
        "latest_news_collection_at": latest_news_collection_at,
    }


def _shadow_model_version_from_predictions(predictions: list[dict]) -> str | None:
    """Return the shadow model version when candidate outputs are present."""
    for prediction in predictions:
        details = prediction.get("context_details", {})
        version = details.get("nextgen_shadow_model_version")
        if version:
            return str(version)
        outputs = prediction.get("component_outputs", {})
        if "nextgen_shadow_probability" in outputs:
            return NEXTGEN_SHADOW_MODEL_VERSION
    return None


def _model_version_from_pipeline(pipeline: PredictionPipeline | object) -> str:
    """Return the active production model version for a prediction payload."""
    return str(getattr(pipeline, "active_model_version", "ensemble_v1"))


def _load_team_mapping() -> dict[str, int]:
    with open(DATA_DIR / "mappings" / "team_to_idx.json") as f:
        return json.load(f)


def _team_label(team_idx: int | str) -> str:
    """Return the public team abbreviation for a model-internal team index."""
    idx = int(team_idx)
    try:
        return idx_to_team_abbr(idx)
    except KeyError:
        return f"TEAM_{idx}"


def _with_team_labels(prediction: dict) -> dict:
    """Attach public team labels while preserving anonymous model IDs."""
    enriched = dict(prediction)
    home_team = _team_label(enriched["home_team_idx"])
    away_team = _team_label(enriched["away_team_idx"])
    enriched["home_team"] = home_team
    enriched["away_team"] = away_team
    enriched["home_team_abbr"] = home_team
    enriched["away_team_abbr"] = away_team
    enriched["matchup"] = f"{away_team} at {home_team}"
    return enriched


def _season_from_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _filter_confirmed_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Keep only confirmed scheduled games.

    The NBA playoff schedule includes future `if necessary` placeholders that
    are not guaranteed to happen. For the live forecast board we only publish
    confirmed games.
    """
    if schedule.empty or "if_necessary" not in schedule.columns:
        return schedule

    confirmed = schedule[~schedule["if_necessary"].fillna(False)].copy()
    filtered_count = len(schedule) - len(confirmed)
    if filtered_count > 0:
        logger.info("Filtered out %d tentative if-necessary games.", filtered_count)
    return confirmed


def _as_bool(value: object) -> bool:
    """Parse NBA API boolean-ish fields that sometimes arrive as strings."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_time(value: object):
    """Extract a clock time from NBA schedule fields that may omit the real date."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    normalized = (
        text.replace("(ET)", "")
        .replace("ET", "")
        .replace("EDT", "")
        .replace("EST", "")
        .strip()
    )
    parsed = pd.to_datetime(normalized, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().time().replace(tzinfo=None)


def _normalize_game_time_utc(game: dict, *, date_str: str) -> str | None:
    """Return a real UTC tipoff datetime even when CDN fields are time-only."""
    for key in ("gameDateTimeUTC", "gameTimeUTC"):
        value = game.get(key)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed) and parsed.year >= 2000:
            return _format_utc(parsed.to_pydatetime())

    et_time = _extract_time(
        game.get("gameDateTimeEst") or game.get("gameEt") or game.get("gameStatusText")
    )
    if et_time is not None:
        local_tipoff = datetime.combine(
            datetime.strptime(date_str, "%Y-%m-%d").date(),
            et_time,
            tzinfo=ZoneInfo("America/New_York"),
        )
        return _format_utc(local_tipoff)

    utc_time = _extract_time(game.get("gameTimeUTC"))
    if utc_time is None:
        return None
    utc_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    if utc_time.hour < 12:
        utc_date += timedelta(days=1)
    return _format_utc(datetime.combine(utc_date, utc_time, tzinfo=timezone.utc))


def _playoff_series_key(row: pd.Series) -> str | None:
    """Return a stable series key for playoff games, otherwise None."""
    game_label = str(row.get("game_label") or "")
    game_sub_label = str(row.get("game_sub_label") or "")
    series_text = str(row.get("series_text") or "")

    is_playoff_series = bool(series_text) or game_sub_label.startswith("Game ")
    if not is_playoff_series:
        return None

    teams = sorted([int(row["home_team_idx"]), int(row["away_team_idx"])])
    return f"{game_label}|{teams[0]}|{teams[1]}"


def _schedule_row_from_nba_game(
    game: dict,
    *,
    date_str: str,
    season_str: str,
    team_mapping: dict[str, int],
) -> dict | None:
    home_team = game.get("homeTeam", {})
    away_team = game.get("awayTeam", {})
    home_abbr = str(home_team.get("teamTricode", "")).upper()
    away_abbr = str(away_team.get("teamTricode", "")).upper()

    if home_abbr not in team_mapping or away_abbr not in team_mapping:
        return None

    return {
        "game_id": str(game["gameId"]),
        "date": date_str,
        "season": season_str,
        "home_team_idx": team_mapping[home_abbr],
        "away_team_idx": team_mapping[away_abbr],
        "game_label": game.get("gameLabel"),
        "game_sub_label": game.get("gameSubLabel") or game.get("seriesGameNumber"),
        "series_text": game.get("seriesText"),
        "game_status_text": game.get("gameStatusText"),
        "game_time_utc": _normalize_game_time_utc(game, date_str=date_str),
        "game_time_et": game.get("gameEt") or game.get("gameDateTimeEst"),
        "game_code": game.get("gameCode"),
        "if_necessary": _as_bool(game.get("ifNecessary", False)),
    }


def _filter_to_next_playoff_games(
    target_games: pd.DataFrame,
    seen_series_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Keep only the next scheduled game for each playoff series."""
    if target_games.empty:
        return target_games, seen_series_keys or set()

    seen = set(seen_series_keys or set())
    keep_indices: list[int] = []
    skipped = 0

    ordered = target_games.sort_values(["date", "game_id"]).copy()
    for index, row in ordered.iterrows():
        series_key = _playoff_series_key(row)
        if series_key and series_key in seen:
            skipped += 1
            continue

        keep_indices.append(index)
        if series_key:
            seen.add(series_key)

    if skipped > 0:
        logger.info("Filtered out %d later playoff games from already-listed series.", skipped)

    return target_games.loc[keep_indices].copy(), seen


def _fetch_schedule_v3(date_str: str, team_mapping: dict[str, int]) -> pd.DataFrame:
    sb = scoreboardv3.ScoreboardV3(
        game_date=date_str,
        timeout=NBA_STATS_TIMEOUT_SECONDS,
    )
    scoreboard = sb.get_dict().get("scoreboard", {})
    games = scoreboard.get("games", [])

    if not games:
        logger.info("No games scheduled for %s.", date_str)
        return pd.DataFrame()

    schedule_rows = []
    season_str = _season_from_date(date_str)

    for game in games:
        row = _schedule_row_from_nba_game(
            game,
            date_str=date_str,
            season_str=season_str,
            team_mapping=team_mapping,
        )
        if row is not None:
            schedule_rows.append(row)

    logger.info("Found %d scheduled games via ScoreboardV3.", len(schedule_rows))
    return _filter_confirmed_schedule(pd.DataFrame(schedule_rows))


def _schedule_day_matches(raw_date: object, date_str: str) -> bool:
    if raw_date is None:
        return False

    raw = str(raw_date)
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat() == date_str
        except ValueError:
            continue

    parsed = pd.to_datetime(raw, errors="coerce")
    return bool(not pd.isna(parsed) and parsed.date().isoformat() == date_str)


def _fetch_schedule_cdn(date_str: str, team_mapping: dict[str, int]) -> pd.DataFrame:
    response = requests.get(
        NBA_CDN_SCHEDULE_URL,
        headers=NBA_CDN_HEADERS,
        timeout=NBA_CDN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    schedule = response.json().get("leagueSchedule", {})
    game_dates = schedule.get("gameDates", [])

    target_day = next(
        (
            day
            for day in game_dates
            if _schedule_day_matches(day.get("gameDate"), date_str)
        ),
        None,
    )
    if target_day is None:
        logger.info("No games scheduled for %s via NBA CDN schedule.", date_str)
        return pd.DataFrame()

    season_str = _season_from_date(date_str)
    schedule_rows = []
    for game in target_day.get("games", []):
        row = _schedule_row_from_nba_game(
            game,
            date_str=date_str,
            season_str=season_str,
            team_mapping=team_mapping,
        )
        if row is not None:
            schedule_rows.append(row)

    logger.info("Found %d scheduled games via NBA CDN schedule.", len(schedule_rows))
    return _filter_confirmed_schedule(pd.DataFrame(schedule_rows))


def _fetch_schedule_v2(date_str: str, team_mapping: dict[str, int]) -> pd.DataFrame:
    sb = scoreboardv2.ScoreboardV2(
        game_date=date_str,
        timeout=NBA_STATS_TIMEOUT_SECONDS,
    )
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
                    "if_necessary": False,
                }
            )

    logger.info("Found %d scheduled games via ScoreboardV2.", len(games))
    return _filter_confirmed_schedule(pd.DataFrame(games))


def _fetch_schedule_with_retries(
    source_name: str,
    date_str: str,
    fetcher: Callable[[], pd.DataFrame],
    *,
    attempts: int = 1,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fetcher()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    "%s fetch attempt %d/%d failed for %s: %s",
                    source_name,
                    attempt,
                    attempts,
                    date_str,
                    exc,
                )
                time.sleep(retry_delay_seconds)
            else:
                logger.warning("%s fetch failed for %s: %s", source_name, date_str, exc)

    if last_error is None:
        msg = f"{source_name} fetch failed for {date_str}"
        raise RuntimeError(msg)
    raise last_error


def fetch_schedule(date_str: str) -> pd.DataFrame:
    """Fetch the schedule for a given date."""
    logger.info("Fetching schedule for %s...", date_str)

    team_mapping = _load_team_mapping()
    sources: list[tuple[str, Callable[[], pd.DataFrame], int]] = [
        (
            "ScoreboardV3",
            lambda: _fetch_schedule_v3(date_str, team_mapping),
            1,
        ),
        (
            "NBA CDN schedule",
            lambda: _fetch_schedule_cdn(date_str, team_mapping),
            2,
        ),
        (
            "ScoreboardV2",
            lambda: _fetch_schedule_v2(date_str, team_mapping),
            1,
        ),
    ]

    errors = []
    for source_name, fetcher, attempts in sources:
        try:
            return _fetch_schedule_with_retries(
                source_name,
                date_str,
                fetcher,
                attempts=attempts,
            )
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")

    msg = f"All schedule sources failed for {date_str}: {' | '.join(errors)}"
    raise RuntimeError(msg)


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
        enriched = _with_team_labels(result)
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
    enable_nextgen_shadow: bool = False,
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
        pipeline = PredictionPipeline(enable_nextgen_shadow=enable_nextgen_shadow)

    results = _predict_for_schedule(
        target_games,
        date_str=date_str,
        pipeline=pipeline,
        historical_games=hist_games,
        historical_team_logs=hist_logs,
    )

    shadow_model_version = _shadow_model_version_from_predictions(results)
    output = {
        "date": date_str,
        "slate_type": "day",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": _model_version_from_pipeline(pipeline),
        "shadow_model_version": shadow_model_version,
        "context_summary": _context_summary_from_predictions(results),
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
    enable_nextgen_shadow: bool = False,
) -> tuple[dict, Path] | None:
    """Generate and save predictions for an upcoming multi-day window."""
    forecast_dates = forecast_window_dates(start_date, days=days)
    hist_games, hist_logs = _load_historical_inputs(
        historical_games=historical_games,
        historical_team_logs=historical_team_logs,
    )

    if pipeline is None:
        pipeline = PredictionPipeline(enable_nextgen_shadow=enable_nextgen_shadow)

    all_predictions: list[dict] = []
    dates_with_games: list[dict[str, int | str]] = []
    seen_series_keys: set[str] = set()

    for date_str in forecast_dates:
        target_games = schedule_fetcher(date_str)
        if target_games.empty:
            continue

        target_games, seen_series_keys = _filter_to_next_playoff_games(
            target_games,
            seen_series_keys,
        )
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
    shadow_model_version = _shadow_model_version_from_predictions(all_predictions)
    output = {
        "date": start_date,
        "slate_type": "week",
        "window_start": forecast_dates[0],
        "window_end": forecast_dates[-1],
        "generated_at": generated_at,
        "model_version": _model_version_from_pipeline(pipeline),
        "shadow_model_version": shadow_model_version,
        "dates_with_games": dates_with_games,
        "playoff_filtering_mode": "next_game_per_series",
        "context_summary": _context_summary_from_predictions(all_predictions),
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
    parser.add_argument(
        "--nextgen-shadow",
        action="store_true",
        help="Emit next-gen comparison fields alongside the active production output.",
    )
    args = parser.parse_args(argv)

    generate_predictions_for_date(args.date, enable_nextgen_shadow=args.nextgen_shadow)


if __name__ == "__main__":
    main()
