"""Fetch live team-news articles and score them into structured sentiment rows."""

from __future__ import annotations

import argparse
import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.data.team_metadata import load_team_metadata_by_idx
from src.nlp.article_filtering import (
    clean_html_text,
    compute_article_relevance,
    is_betting_or_promo_article,
)
from src.nlp.extract_sentiment import score_articles
from src.utils.logging import setup_logging
from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_MAX_ITEMS = 20
GOOGLE_NEWS_ENDPOINT = "https://news.google.com/rss/search"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-predict/1.0)"}
MIN_ARTICLE_RELEVANCE = 0.18


def _rss_query_for_team(team_metadata: dict[str, object]) -> str:
    full_name = str(team_metadata["full_name"])
    nickname = str(team_metadata["nickname"])
    return f'"{full_name}" NBA OR "{nickname}" NBA'


def _google_news_rss_url(query: str, lookback_days: int) -> str:
    encoded = quote_plus(query)
    return f"{GOOGLE_NEWS_ENDPOINT}?q={encoded}+when:{lookback_days}d&hl=en-US&gl=US&ceid=US:en"


def _parse_rss_items(xml_text: str, *, team_idx: int) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        title = clean_html_text(item.findtext("title"))
        link = clean_html_text(item.findtext("link"))
        description = clean_html_text(item.findtext("description"))
        pub_date = clean_html_text(item.findtext("pubDate"))
        source = clean_html_text(item.findtext("source"))
        rows.append(
            {
                "team_idx": team_idx,
                "published_at": pub_date,
                "title": title,
                "summary": description,
                "source": source or None,
                "link": link,
            }
        )
    return pd.DataFrame(rows)


def _article_id(row: pd.Series) -> str:
    title = str(row.get("title", ""))
    return hashlib.sha1(
        f"{row.get('link', '')}|{row.get('published_at', '')}|{title}".encode("utf-8")
    ).hexdigest()


def annotate_team_news_articles(
    articles: pd.DataFrame,
    *,
    collected_at: str,
) -> pd.DataFrame:
    """Annotate fetched articles with model inclusion/exclusion metadata."""
    if articles.empty:
        return articles

    team_metadata = load_team_metadata_by_idx()
    annotated = articles.copy()
    annotated["article_id"] = annotated.apply(_article_id, axis=1)

    relevance_scores = []
    relevance_reasons = []
    excluded_reasons = []
    collected_ts = pd.to_datetime(collected_at, utc=True, errors="coerce")
    for _, row in annotated.iterrows():
        aliases = list(team_metadata[int(row["team_idx"])]["aliases"])
        relevance, reason = compute_article_relevance(
            str(row.get("title", "")),
            str(row.get("summary", "")),
            aliases,
        )
        published_ts = pd.to_datetime(row.get("published_at"), utc=True, errors="coerce")
        stale_article = (
            pd.notna(collected_ts)
            and pd.notna(published_ts)
            and published_ts < collected_ts - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS)
        )
        relevance_scores.append(relevance)
        relevance_reasons.append(reason)
        if is_betting_or_promo_article(
            str(row.get("title", "")),
            str(row.get("summary", "")),
            str(row.get("source") or ""),
        ):
            excluded_reasons.append("betting_or_promo")
        elif stale_article:
            excluded_reasons.append("stale")
        elif relevance < MIN_ARTICLE_RELEVANCE:
            excluded_reasons.append("low_relevance")
        else:
            excluded_reasons.append("")

    annotated["article_relevance"] = relevance_scores
    annotated["article_relevance_reason"] = relevance_reasons
    annotated["excluded_reason"] = excluded_reasons
    duplicate_mask = annotated.duplicated(subset=["team_idx", "link", "title"], keep="first")
    annotated.loc[duplicate_mask, "excluded_reason"] = "duplicate"
    annotated["included_in_model"] = annotated["excluded_reason"] == ""
    annotated["collected_at"] = collected_at
    return annotated.reset_index(drop=True)


def fetch_team_news_articles(
    team_idx: int,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_items: int = DEFAULT_MAX_ITEMS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch recent news articles for one team from Google News RSS."""
    metadata = load_team_metadata_by_idx()[int(team_idx)]
    query = _rss_query_for_team(metadata)
    url = _google_news_rss_url(query, lookback_days)

    client = session or requests.Session()
    response = client.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()

    articles = _parse_rss_items(response.text, team_idx=int(team_idx))
    if articles.empty:
        return articles

    articles["published_at"] = pd.to_datetime(articles["published_at"], utc=True, errors="coerce")
    articles = articles.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
    articles = articles.head(max_items).copy()
    articles["published_at"] = articles["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return articles.reset_index(drop=True)


def fetch_news_scores_for_games(
    games: pd.DataFrame,
    *,
    cache_dir: Path | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Fetch and score live team news for the teams appearing in `games`."""
    if games.empty:
        return pd.DataFrame()

    storage_dir = cache_dir or (RAW_DIR / "news")
    storage_dir.mkdir(parents=True, exist_ok=True)

    as_of_date = datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date().isoformat()
    cache_path = storage_dir / f"team_news_scores_{as_of_date}.parquet"
    article_cache_path = storage_dir / f"team_news_articles_{as_of_date}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        target_teams = set(games["home_team_idx"]).union(set(games["away_team_idx"]))
        return cached[cached["team_idx"].isin(target_teams)].reset_index(drop=True)

    target_teams = sorted(set(games["home_team_idx"]).union(set(games["away_team_idx"])))
    session = requests.Session()
    article_frames = []
    for team_idx in target_teams:
        try:
            article_frames.append(
                fetch_team_news_articles(
                    int(team_idx),
                    lookback_days=lookback_days,
                    session=session,
                )
            )
        except Exception as exc:
            logger.warning("News fetch failed for team_idx=%s: %s", team_idx, exc)

    if not article_frames:
        return pd.DataFrame()

    articles = pd.concat(article_frames, ignore_index=True)
    if articles.empty:
        return pd.DataFrame()

    collected_at = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    annotated_articles = annotate_team_news_articles(articles, collected_at=collected_at)
    annotated_articles.to_parquet(article_cache_path, index=False)
    model_articles = annotated_articles[annotated_articles["included_in_model"]].copy()
    if model_articles.empty:
        return pd.DataFrame()

    scores = score_articles(model_articles)
    if scores.empty:
        return scores

    scores["collected_at"] = collected_at
    scores.to_parquet(cache_path, index=False)
    return scores.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Fetch live NBA team news scores.")
    parser.add_argument(
        "--team-idx",
        type=int,
        action="append",
        default=[],
        help="Anonymous team index to fetch. Can be provided multiple times.",
    )
    args = parser.parse_args(argv)

    if not args.team_idx:
        logger.error("Provide at least one --team-idx value.")
        return 1

    games = pd.DataFrame(
        [{"home_team_idx": team_idx, "away_team_idx": team_idx} for team_idx in args.team_idx]
    )
    scores = fetch_news_scores_for_games(games)
    logger.info("Fetched %d scored team-news rows.", len(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
