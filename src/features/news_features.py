"""News/sentiment feature engineering — team-level news signal aggregation.

Builds news feature vectors per team per game. In V1, we use zero vectors
with news_available=0 for all historical games. The infrastructure supports
plugging in real LLM-extracted sentiment when available.

Features produced per team per game:
    weighted_sentiment_24h: Weighted average sentiment (last 24h articles)
    weighted_sentiment_72h: Weighted average sentiment (last 72h articles)
    article_volume_24h: Number of articles in last 24h
    negative_ratio_72h: Fraction of negative articles in last 72h
    sentiment_volatility_72h: Std of sentiment scores in last 72h
    avg_llm_confidence: Average LLM confidence on extracted scores
    news_available: 1 if real news data, 0 if using defaults

Usage:
    python -m src.features.news_features
"""

import logging

import numpy as np
import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# News feature columns
NEWS_FEATURE_COLS = [
    "weighted_sentiment_24h",
    "weighted_sentiment_72h",
    "article_volume_24h",
    "negative_ratio_72h",
    "sentiment_volatility_72h",
    "avg_llm_confidence",
    "news_available",
]


def load_news_scores() -> pd.DataFrame | None:
    """Load LLM-extracted article scores if available.

    Returns:
        DataFrame with columns: game_id, team_idx, published_at,
        overall_sentiment, injury_concern, pressure, team_cohesion,
        motivation, llm_confidence, article_relevance
        or None if no news data exists.
    """
    news_dir = RAW_DIR / "news"
    if not news_dir.exists():
        logger.info("No news directory found — using zero vectors")
        return None

    news_files = [f for f in news_dir.iterdir() if f.suffix == ".parquet"]

    if not news_files:
        logger.info("No news/sentiment data found — using zero vectors")
        return None

    dfs = [pd.read_parquet(f) for f in news_files]
    return pd.concat(dfs, ignore_index=True)


def build_news_features_from_scores(
    news_scores: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Build news features from LLM-extracted article scores.

    Aggregates article-level scores into per-team per-game vectors
    using recency-weighted averaging.
    """
    features = []
    games = games.sort_values("date").copy()

    for _, game in games.iterrows():
        game_id = game["game_id"]
        game_date = pd.Timestamp(game["date"])

        for team_col in ["home_team_idx", "away_team_idx"]:
            team_idx = int(game[team_col])

            # Get articles for this team before game date
            mask = (
                (news_scores["team_idx"] == team_idx)
                & (pd.to_datetime(news_scores["published_at"]) < game_date)
            )
            team_articles = news_scores[mask].copy()

            if len(team_articles) == 0:
                features.append({
                    "game_id": game_id,
                    "team_idx": team_idx,
                    **{c: 0.0 for c in NEWS_FEATURE_COLS[:-1]},
                    "news_available": 0,
                })
                continue

            team_articles["published_at"] = pd.to_datetime(team_articles["published_at"])
            hours_before = (game_date - team_articles["published_at"]).dt.total_seconds() / 3600

            # Recency weights
            recency = np.where(hours_before <= 12, 1.0,
                     np.where(hours_before <= 24, 0.8,
                     np.where(hours_before <= 72, 0.5,
                     np.where(hours_before <= 168, 0.2, 0.0))))

            weights = (
                recency
                * team_articles.get("article_relevance", pd.Series(np.ones(len(team_articles)))).values
                * team_articles.get("llm_confidence", pd.Series(np.ones(len(team_articles)))).values
            )

            # 24h and 72h windows
            mask_24h = hours_before <= 24
            mask_72h = hours_before <= 72

            sentiment = team_articles["overall_sentiment"].values

            w24 = weights[mask_24h]
            s24 = sentiment[mask_24h]
            w72 = weights[mask_72h]
            s72 = sentiment[mask_72h]

            features.append({
                "game_id": game_id,
                "team_idx": team_idx,
                "weighted_sentiment_24h": float(np.average(s24, weights=w24)) if w24.sum() > 0 else 0.0,
                "weighted_sentiment_72h": float(np.average(s72, weights=w72)) if w72.sum() > 0 else 0.0,
                "article_volume_24h": int(mask_24h.sum()),
                "negative_ratio_72h": float((s72 < 0).mean()) if len(s72) > 0 else 0.0,
                "sentiment_volatility_72h": float(s72.std()) if len(s72) > 1 else 0.0,
                "avg_llm_confidence": float(team_articles.get("llm_confidence", pd.Series([0.5])).mean()),
                "news_available": 1,
            })

    return pd.DataFrame(features)


def build_news_features_default(games: pd.DataFrame) -> pd.DataFrame:
    """Build default zero-vector news features when no news data exists.

    As per project plan: zero vector + news_available=0 for all games.
    The model learns to handle missing news info gracefully.

    Vectorized: builds all rows at once instead of iterating.
    """
    # Build rows for both home and away teams
    home_df = games[["game_id", "home_team_idx"]].copy()
    home_df = home_df.rename(columns={"home_team_idx": "team_idx"})
    away_df = games[["game_id", "away_team_idx"]].copy()
    away_df = away_df.rename(columns={"away_team_idx": "team_idx"})

    result = pd.concat([home_df, away_df], ignore_index=True)

    for col in NEWS_FEATURE_COLS[:-1]:
        result[col] = 0.0
    result["news_available"] = 0

    logger.info(
        "Built default news features: %d rows (zero vectors, news_available=0)",
        len(result),
    )
    return result


def build_news_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build news features, using real data if available, defaults otherwise."""
    news_scores = load_news_scores()

    if news_scores is not None:
        logger.info("Building news features from %d article scores", len(news_scores))
        return build_news_features_from_scores(news_scores, games)
    else:
        logger.info("No news data — building zero-vector defaults")
        return build_news_features_default(games)


def main():
    """Build and save news features."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    logger.info("Building news features for %d games...", len(games))
    features = build_news_features(games)

    out_dir = PROCESSED_DIR / "news_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "news_features.parquet"
    features.to_parquet(out_path, index=False)

    logger.info(
        "Saved news features: %d rows, %d columns to %s",
        len(features), len(features.columns), out_path,
    )
    logger.info(
        "Coverage: %.1f%% with real news data",
        features["news_available"].mean() * 100,
    )


if __name__ == "__main__":
    main()
