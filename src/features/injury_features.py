"""Team injury context features with live-report support for current slates."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.fetch_injuries import DEFAULT_TIMEZONE, fetch_injury_reports_for_games
from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

INJURY_FEATURE_COLS = [
    "players_out_count",
    "players_questionable_count",
    "starter_out_count",
    "minutes_missing",
    "usage_missing",
    "estimated_value_missing",
    "injury_data_available",
]

_OUT_STATUSES = {"OUT"}
_QUESTIONABLE_STATUSES = {"QUESTIONABLE", "DOUBTFUL", "GAME TIME DECISION"}
_LOW_IMPACT_REASONS = {"G LEAGUE - TWO-WAY", "G LEAGUE - ON ASSIGNMENT"}


def _status_impact_weight(status: str, reason: str | None) -> float:
    normalized_status = status.upper().strip()
    normalized_reason = (reason or "").upper()
    if normalized_status in _OUT_STATUSES:
        if any(tag in normalized_reason for tag in _LOW_IMPACT_REASONS):
            return 0.2
        return 1.0
    if normalized_status == "DOUBTFUL":
        return 0.75
    if normalized_status in {"QUESTIONABLE", "GAME TIME DECISION"}:
        return 0.45
    return 0.0


def _is_live_forecast_window(games: pd.DataFrame, timezone_name: str = DEFAULT_TIMEZONE) -> bool:
    if games.empty or "date" not in games.columns:
        return False
    game_dates = pd.to_datetime(games["date"]).dt.date
    today = datetime.now(ZoneInfo(timezone_name)).date()
    return bool((game_dates >= today).all() and (game_dates <= today + timedelta(days=1)).all())


def load_injury_reports(
    games: pd.DataFrame | None = None,
    *,
    allow_live_fetch: bool = False,
) -> pd.DataFrame | None:
    """Load cached injury report rows and optionally fetch a live snapshot."""
    injury_dir = RAW_DIR / "injuries"
    cached_frames = []
    if injury_dir.exists():
        injury_files = [path for path in injury_dir.iterdir() if path.suffix == ".parquet"]
        if injury_files:
            cached_frames = [pd.read_parquet(path) for path in injury_files]

    if cached_frames:
        reports = pd.concat(cached_frames, ignore_index=True)
        if games is not None and "game_id" in reports.columns:
            reports = reports[reports["game_id"].isin(games["game_id"])]
        if not reports.empty:
            return reports.reset_index(drop=True)

    if allow_live_fetch and games is not None and _is_live_forecast_window(games):
        try:
            reports = fetch_injury_reports_for_games(games)
            if not reports.empty:
                return reports.reset_index(drop=True)
        except Exception as exc:
            logger.warning("Live injury report fetch failed: %s", exc)

    logger.info("No injury report rows available - using fallback features")
    return None


def build_injury_features_from_reports(
    injury_reports: pd.DataFrame,
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate submitted injury report rows into team-level feature vectors."""
    if injury_reports.empty:
        return pd.DataFrame(columns=["game_id", "team_idx", *INJURY_FEATURE_COLS])

    features: list[dict[str, object]] = []
    submitted = injury_reports[injury_reports["report_submitted"]].copy()
    pending = injury_reports[~injury_reports["report_submitted"]].copy()

    for (game_id, team_idx), group in pending.groupby(["game_id", "team_idx"], sort=False):
        features.append(
            {
                "game_id": game_id,
                "team_idx": int(team_idx),
                "injury_data_available": 0,
                "report_generated_at": group["report_generated_at"].max(),
                "report_source_url": group["source_url"].iloc[0],
                "report_status": "not_submitted",
            }
        )

    for (game_id, team_idx), group in submitted.groupby(["game_id", "team_idx"], sort=False):
        if group["status"].eq("CLEAR").all():
            features.append(
                {
                    "game_id": game_id,
                    "team_idx": int(team_idx),
                    "players_out_count": 0,
                    "players_questionable_count": 0,
                    "starter_out_count": 0,
                    "minutes_missing": 0.0,
                    "usage_missing": 0.0,
                    "estimated_value_missing": 0.0,
                    "injury_data_available": 1,
                    "report_generated_at": group["report_generated_at"].max(),
                    "report_source_url": group["source_url"].iloc[0],
                    "report_status": "submitted",
                }
            )
            continue

        out_count = 0
        questionable_count = 0
        weighted_outs = []
        for _, row in group.iterrows():
            weight = _status_impact_weight(str(row["status"]), row.get("reason"))
            if str(row["status"]).upper() in _OUT_STATUSES:
                out_count += 1
            elif str(row["status"]).upper() in _QUESTIONABLE_STATUSES:
                questionable_count += 1
            if weight > 0:
                weighted_outs.append(weight)

        total_weight = float(sum(weighted_outs))
        features.append(
            {
                "game_id": game_id,
                "team_idx": int(team_idx),
                "players_out_count": out_count,
                "players_questionable_count": questionable_count,
                "starter_out_count": min(out_count, 3),
                "minutes_missing": round(total_weight * 24.0, 4),
                "usage_missing": round(total_weight * 0.18, 4),
                "estimated_value_missing": round(total_weight, 4),
                "injury_data_available": 1,
                "report_generated_at": group["report_generated_at"].max(),
                "report_source_url": group["source_url"].iloc[0],
                "report_status": "submitted",
            }
        )

    if not features:
        return pd.DataFrame(columns=["game_id", "team_idx", *INJURY_FEATURE_COLS])

    return pd.DataFrame(features)


def build_injury_features_default(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build fallback injury features from team-level performance instability."""
    team_logs = team_logs.sort_values(["team_idx", "date"]).copy()
    team_logs["date"] = pd.to_datetime(team_logs["date"])

    all_team_stats = {}
    for team_idx, group in team_logs.groupby("team_idx"):
        group = group.sort_values("date").reset_index(drop=True)
        rating_col = "net_rating" if "net_rating" in group.columns else "point_diff"
        values = group[rating_col]
        roll_std = values.shift(1).rolling(10, min_periods=3).std().values
        mean_3 = values.shift(1).rolling(3, min_periods=3).mean().values
        mean_10 = values.shift(1).rolling(10, min_periods=3).mean().values
        drop = np.maximum(0, mean_10 - mean_3)
        all_team_stats[int(team_idx)] = {
            "dates": group["date"].values,
            "roll_std": roll_std,
            "drop": drop,
        }

    rows = []
    games_sorted = games.sort_values("date").copy()
    for team_col in ["home_team_idx", "away_team_idx"]:
        for _, game in games_sorted.iterrows():
            team_idx = int(game[team_col])
            game_date = pd.Timestamp(game["date"])
            stats = all_team_stats.get(team_idx)

            if stats is None:
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "team_idx": team_idx,
                        **{column: 0.0 for column in INJURY_FEATURE_COLS[:-1]},
                        "injury_data_available": 0,
                        "report_status": "fallback",
                    }
                )
                continue

            prior_indices = np.where(stats["dates"] < game_date)[0]
            if len(prior_indices) < 3:
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "team_idx": team_idx,
                        **{column: 0.0 for column in INJURY_FEATURE_COLS[:-1]},
                        "injury_data_available": 0,
                        "report_status": "fallback",
                    }
                )
                continue

            last_idx = prior_indices[-1]
            drop_val = stats["drop"][last_idx]
            drop_val = 0.0 if np.isnan(drop_val) else float(drop_val)
            rows.append(
                {
                    "game_id": game["game_id"],
                    "team_idx": team_idx,
                    "players_out_count": 0,
                    "players_questionable_count": 0,
                    "starter_out_count": 0,
                    "minutes_missing": 0.0,
                    "usage_missing": 0.0,
                    "estimated_value_missing": drop_val,
                    "injury_data_available": 0,
                    "report_status": "fallback",
                }
            )

    result = pd.DataFrame(rows)
    logger.info(
        "Built fallback injury features: %d rows with injury_data_available=0",
        len(result),
    )
    return result


def build_injury_features(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
    *,
    allow_live_fetch: bool = False,
) -> pd.DataFrame:
    """Build injury features, overlaying live reports onto fallback features when available."""
    fallback_features = build_injury_features_default(games, team_logs)
    injury_reports = load_injury_reports(games, allow_live_fetch=allow_live_fetch)
    if injury_reports is None or injury_reports.empty:
        return fallback_features

    live_features = build_injury_features_from_reports(injury_reports, games, team_logs)
    if live_features.empty:
        return fallback_features

    merged = (
        live_features.set_index(["game_id", "team_idx"])
        .combine_first(fallback_features.set_index(["game_id", "team_idx"]))
        .reset_index()
    )
    logger.info(
        "Built injury features with %.1f%% live report coverage",
        merged["injury_data_available"].mean() * 100,
    )
    return merged


def main() -> None:
    """Build and save processed injury features."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    team_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")

    logger.info("Building injury features for %d games...", len(games))
    features = build_injury_features(games, team_logs, allow_live_fetch=False)

    out_dir = PROCESSED_DIR / "injury_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "injury_features.parquet"
    features.to_parquet(out_path, index=False)

    logger.info(
        "Saved injury features: %d rows, %d columns to %s",
        len(features),
        len(features.columns),
        out_path,
    )
    logger.info(
        "Coverage: %.1f%% with submitted live injury reports",
        features["injury_data_available"].mean() * 100,
    )


if __name__ == "__main__":
    main()
