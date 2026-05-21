"""Utilities for filtering and scoring team-news article relevance."""

from __future__ import annotations

import re

import pandas as pd

_EXCLUDED_NEWS_TERMS = [
    "BEST BET",
    "BETTING",
    "BETMGM",
    "BONUS",
    "COMPUTER PICKS",
    "DRAFTKINGS",
    "FANDUEL",
    "ODDS",
    "PARLAY",
    "PLAYER PROP",
    "POLYMARKET",
    "PROMO CODE",
    "PROP BET",
    "PROP PROJECTION",
    "SPORTSLINE",
    "SAME GAME PARLAY",
    "DFS",
]


def clean_html_text(value: str | None) -> str:
    """Strip simple HTML tags and whitespace from RSS content."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def compute_article_relevance(
    title: str,
    summary: str,
    team_aliases: list[str],
) -> tuple[float, str]:
    """Estimate how directly an article speaks to a team's current context."""
    haystack = f"{title} {summary}".upper()
    alias_hits = sum(1 for alias in team_aliases if alias in haystack)

    recency_terms = [
        "INJURY",
        "OUT",
        "QUESTIONABLE",
        "DOUBTFUL",
        "PROBABLE",
        "PREVIEW",
        "MATCHUP",
        "PLAYOFF",
        "FINAL",
        "COACH",
        "RETURN",
        "AVAILABLE",
        "LINEUP",
        "START",
    ]
    recency_hits = sum(1 for term in recency_terms if term in haystack)

    score = min(1.0, 0.2 + 0.18 * alias_hits + 0.08 * recency_hits)
    if alias_hits == 0:
        score = 0.0

    if alias_hits and recency_hits:
        reason = "team mention plus matchup/injury context"
    elif alias_hits:
        reason = "team-specific mention"
    else:
        reason = "team mention not detected"
    return round(score, 4), reason


def is_betting_or_promo_article(title: str, summary: str, source: str | None = None) -> bool:
    """Return whether an article is odds/prop/promo content, not team context."""
    haystack = f"{title} {summary} {source or ''}".upper()
    return any(term in haystack for term in _EXCLUDED_NEWS_TERMS)


def filter_team_articles(
    articles: pd.DataFrame,
    team_aliases: list[str],
    *,
    min_relevance: float = 0.18,
) -> pd.DataFrame:
    """Keep only articles that directly mention the target team."""
    if articles.empty:
        return articles

    scored = articles.copy()
    relevance = scored.apply(
        lambda row: compute_article_relevance(
            str(row.get("title", "")),
            str(row.get("summary", "")),
            team_aliases,
        ),
        axis=1,
    )
    scored["article_relevance"] = [item[0] for item in relevance]
    scored["article_relevance_reason"] = [item[1] for item in relevance]
    scored["excluded_news_reason"] = scored.apply(
        lambda row: "betting_or_promo"
        if is_betting_or_promo_article(
            str(row.get("title", "")),
            str(row.get("summary", "")),
            str(row.get("source") or ""),
        )
        else "",
        axis=1,
    )
    return scored[
        (scored["article_relevance"] >= min_relevance)
        & (scored["excluded_news_reason"] == "")
    ].reset_index(drop=True)
