"""Team-level news sentiment features with live article collection support."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.fetch_news import DEFAULT_TIMEZONE, fetch_news_scores_for_games
from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

NEWS_FEATURE_COLS = [
    "weighted_sentiment_24h",
    "weighted_sentiment_72h",
    "article_volume_24h",
    "negative_ratio_72h",
    "sentiment_volatility_72h",
    "avg_llm_confidence",
    "news_available",
]


def _is_live_forecast_window(games: pd.DataFrame, timezone_name: str = DEFAULT_TIMEZONE) -> bool:
    if games.empty or "date" not in games.columns:
        return False
    game_dates = pd.to_datetime(games["date"]).dt.date
    today = datetime.now(ZoneInfo(timezone_name)).date()
    return bool((game_dates >= today).all() and (game_dates <= today + timedelta(days=7)).all())


def load_news_scores(
    games: pd.DataFrame | None = None,
    *,
    allow_live_fetch: bool = False,
) -> pd.DataFrame | None:
    """Load cached or live-scored team news rows."""
    news_dir = RAW_DIR / "news"
    cached_frames = []
    if news_dir.exists():
        news_files = [path for path in news_dir.iterdir() if path.suffix == ".parquet"]
        if news_files:
            cached_frames = [pd.read_parquet(path) for path in news_files]

    if cached_frames:
        scores = pd.concat(cached_frames, ignore_index=True)
        if games is not None and "team_idx" in scores.columns:
            target_teams = set(games["home_team_idx"]).union(set(games["away_team_idx"]))
            scores = scores[scores["team_idx"].isin(target_teams)]
        if not scores.empty:
            return scores.reset_index(drop=True)

    if allow_live_fetch and games is not None and _is_live_forecast_window(games):
        try:
            scores = fetch_news_scores_for_games(games)
            if not scores.empty:
                return scores.reset_index(drop=True)
        except Exception as exc:
            logger.warning("Live news fetch failed: %s", exc)

    logger.info("No news score rows available - using fallback vectors")
    return None


def build_news_features_from_scores(
    news_scores: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate article scores into per-team, per-game news features."""
    if news_scores.empty:
        return pd.DataFrame(columns=["game_id", "team_idx", *NEWS_FEATURE_COLS])

    features = []
    games = games.sort_values("date").copy()
    news_scores = news_scores.copy()
    news_scores["published_at"] = pd.to_datetime(
        news_scores["published_at"], utc=True, errors="coerce"
    )
    news_scores = news_scores.dropna(subset=["published_at"])
    news_scores["published_at"] = (
        news_scores["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    )

    for _, game in games.iterrows():
        game_id = game["game_id"]
        raw_game_time = game.get("game_time_utc")
        parsed_game_time = pd.to_datetime(raw_game_time, utc=True, errors="coerce")
        if pd.notna(parsed_game_time):
            game_date = parsed_game_time.tz_convert("UTC").tz_localize(None)
        else:
            game_date = pd.Timestamp(game["date"])

        for team_col in ["home_team_idx", "away_team_idx"]:
            team_idx = int(game[team_col])
            team_articles = news_scores[
                (news_scores["team_idx"] == team_idx) & (news_scores["published_at"] < game_date)
            ].copy()

            if team_articles.empty:
                continue

            hours_before = (game_date - team_articles["published_at"]).dt.total_seconds() / 3600
            recency = np.where(
                hours_before <= 12,
                1.0,
                np.where(
                    hours_before <= 24,
                    0.8,
                    np.where(hours_before <= 72, 0.5, np.where(hours_before <= 168, 0.2, 0.0)),
                ),
            )

            relevance = team_articles.get(
                "article_relevance",
                pd.Series(np.ones(len(team_articles)), index=team_articles.index),
            ).to_numpy(dtype=float)
            confidence = team_articles.get(
                "llm_confidence",
                pd.Series(np.ones(len(team_articles)), index=team_articles.index),
            ).to_numpy(dtype=float)
            weights = recency * relevance * confidence

            mask_24h = hours_before <= 24
            mask_72h = hours_before <= 72
            sentiment = team_articles["overall_sentiment"].to_numpy(dtype=float)

            w24 = weights[mask_24h]
            s24 = sentiment[mask_24h]
            w72 = weights[mask_72h]
            s72 = sentiment[mask_72h]

            latest_article_at = (
                team_articles.loc[mask_72h, "published_at"].max() if mask_72h.any() else pd.NaT
            )
            features.append(
                {
                    "game_id": game_id,
                    "team_idx": team_idx,
                    "weighted_sentiment_24h": float(np.average(s24, weights=w24))
                    if w24.sum() > 0
                    else 0.0,
                    "weighted_sentiment_72h": float(np.average(s72, weights=w72))
                    if w72.sum() > 0
                    else 0.0,
                    "article_volume_24h": int(mask_24h.sum()),
                    "negative_ratio_72h": float((s72 < 0).mean()) if len(s72) > 0 else 0.0,
                    "sentiment_volatility_72h": float(s72.std()) if len(s72) > 1 else 0.0,
                    "avg_llm_confidence": float(team_articles["llm_confidence"].mean()),
                    "news_available": 1,
                    "latest_article_at": latest_article_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if pd.notna(latest_article_at)
                    else None,
                    "article_count_72h": int(mask_72h.sum()),
                    "news_collected_at": team_articles.get(
                        "collected_at", pd.Series(dtype=str)
                    ).max()
                    if "collected_at" in team_articles.columns
                    else None,
                }
            )

    return pd.DataFrame(features)


def build_news_features_default(games: pd.DataFrame) -> pd.DataFrame:
    """Build default zero-vector news features when no live articles are available."""
    home_df = (
        games[["game_id", "home_team_idx"]]
        .rename(columns={"home_team_idx": "team_idx"})
        .copy()
    )
    away_df = (
        games[["game_id", "away_team_idx"]]
        .rename(columns={"away_team_idx": "team_idx"})
        .copy()
    )

    result = pd.concat([home_df, away_df], ignore_index=True)
    for column in NEWS_FEATURE_COLS[:-1]:
        result[column] = 0.0
    result["news_available"] = 0
    result["latest_article_at"] = None
    result["article_count_72h"] = 0
    result["news_collected_at"] = None

    logger.info(
        "Built default news features: %d rows (zero vectors, news_available=0)",
        len(result),
    )
    return result


def build_news_features(
    games: pd.DataFrame,
    *,
    allow_live_fetch: bool = False,
) -> pd.DataFrame:
    """Build news features, overlaying live article aggregates when available."""
    default_features = build_news_features_default(games)
    news_scores = load_news_scores(games, allow_live_fetch=allow_live_fetch)
    if news_scores is None or news_scores.empty:
        return default_features

    live_features = build_news_features_from_scores(news_scores, games)
    if live_features.empty:
        return default_features

    merged = (
        live_features.set_index(["game_id", "team_idx"])
        .combine_first(default_features.set_index(["game_id", "team_idx"]))
        .reset_index()
    )
    logger.info(
        "Built news features with %.1f%% live article coverage",
        merged["news_available"].mean() * 100,
    )
    return merged


def main() -> None:
    """Build and save processed news features."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    logger.info("Building news features for %d games...", len(games))
    features = build_news_features(games, allow_live_fetch=False)

    out_dir = PROCESSED_DIR / "news_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "news_features.parquet"
    features.to_parquet(out_path, index=False)

    logger.info(
        "Saved news features: %d rows, %d columns to %s",
        len(features),
        len(features.columns),
        out_path,
    )
    logger.info(
        "Coverage: %.1f%% with live article scores",
        features["news_available"].mean() * 100,
    )


if __name__ == "__main__":
    main()
