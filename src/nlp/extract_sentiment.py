"""Deterministic live article scorer used when LLM outputs are unavailable."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

import pandas as pd

from src.data.team_metadata import load_team_metadata_by_idx
from src.nlp.article_filtering import filter_team_articles
from src.nlp.sentiment_schema import ArticleSentimentScore

_POSITIVE_TERMS = {
    "CLEARED",
    "ACTIVE",
    "AVAILABLE",
    "RETURN",
    "RETURNS",
    "BACK",
    "HEALTHY",
    "PRACTICED",
    "WIN",
    "WINS",
    "STRONG",
    "CONFIDENT",
    "MOMENTUM",
    "SURGE",
    "HOT",
}
_NEGATIVE_TERMS = {
    "OUT",
    "DOUBTFUL",
    "QUESTIONABLE",
    "INJURY",
    "INJURED",
    "MISS",
    "MISSES",
    "SIDELINED",
    "SETBACK",
    "STRAIN",
    "SPRAIN",
    "DOUBT",
    "LIMITED",
}
_PRESSURE_TERMS = {
    "PLAYOFF",
    "FINAL",
    "ELIMINATION",
    "MUST-WIN",
    "DO-OR-DIE",
    "GAME 7",
    "DECIDER",
    "WIN OR GO HOME",
}
_COHESION_POSITIVE_TERMS = {"TOGETHER", "CHEMISTRY", "CONNECTED", "LOCKED IN", "UNITED"}
_COHESION_NEGATIVE_TERMS = {"TENSION", "DRAMA", "FEUD", "DISPUTE", "FRUSTRATION"}
_MOTIVATION_TERMS = {"MOTIVATED", "FOCUSED", "DETERMINED", "READY", "LOCKED IN", "REVENGE"}


def _token_hits(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def score_article(
    row: Mapping[str, object], *, team_aliases: list[str]
) -> ArticleSentimentScore | None:
    """Score one team-linked article into a structured feature row."""
    filtered = filter_team_articles(pd.DataFrame([row]), team_aliases)
    if filtered.empty:
        return None

    record = filtered.iloc[0].to_dict()
    title = str(record.get("title", ""))
    summary = str(record.get("summary", ""))
    text = re.sub(r"\s+", " ", f"{title} {summary}").upper()

    positive_hits = _token_hits(text, _POSITIVE_TERMS)
    negative_hits = _token_hits(text, _NEGATIVE_TERMS)
    pressure_hits = _token_hits(text, _PRESSURE_TERMS)
    cohesion_pos_hits = _token_hits(text, _COHESION_POSITIVE_TERMS)
    cohesion_neg_hits = _token_hits(text, _COHESION_NEGATIVE_TERMS)
    motivation_hits = _token_hits(text, _MOTIVATION_TERMS)

    polarity_total = positive_hits + negative_hits
    overall_sentiment = 0.0
    if polarity_total > 0:
        overall_sentiment = _bounded(
            (positive_hits - negative_hits) / max(polarity_total, 1),
            -1.0,
            1.0,
        )

    injury_concern = _bounded(0.25 * negative_hits, 0.0, 1.0)
    pressure = _bounded(0.33 * pressure_hits, 0.0, 1.0)
    team_cohesion = _bounded(
        0.5 + 0.18 * cohesion_pos_hits - 0.22 * cohesion_neg_hits,
        0.0,
        1.0,
    )
    motivation = _bounded(
        0.22 * motivation_hits + max(0.0, overall_sentiment) * 0.25,
        0.0,
        1.0,
    )

    relevance = float(record.get("article_relevance", 0.0))
    signal_hits = (
        positive_hits
        + negative_hits
        + pressure_hits
        + cohesion_pos_hits
        + cohesion_neg_hits
        + motivation_hits
    )
    llm_confidence = _bounded(
        0.35 + 0.08 * min(signal_hits, 5) + 0.3 * relevance,
        0.0,
        0.97,
    )

    article_id = hashlib.sha1(
        f"{record.get('link', '')}|{record.get('published_at', '')}|{title}".encode("utf-8")
    ).hexdigest()

    return ArticleSentimentScore(
        article_id=article_id,
        team_idx=int(record["team_idx"]),
        published_at=str(record["published_at"]),
        title=title,
        source=str(record.get("source") or "") or None,
        article_relevance=relevance,
        overall_sentiment=round(overall_sentiment, 4),
        injury_concern=round(injury_concern, 4),
        pressure=round(pressure, 4),
        team_cohesion=round(team_cohesion, 4),
        motivation=round(motivation, 4),
        llm_confidence=round(llm_confidence, 4),
        article_relevance_reason=str(record.get("article_relevance_reason") or "") or None,
        link=str(record.get("link") or "") or None,
    )


def score_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Score a frame of fetched news articles into structured sentiment rows."""
    if articles.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                "team_idx",
                "published_at",
                "title",
                "source",
                "article_relevance",
                "overall_sentiment",
                "injury_concern",
                "pressure",
                "team_cohesion",
                "motivation",
                "llm_confidence",
                "article_relevance_reason",
                "link",
            ]
        )

    team_metadata = load_team_metadata_by_idx()
    rows: list[dict[str, object]] = []
    for _, article in articles.iterrows():
        team_idx = int(article["team_idx"])
        scored = score_article(
            article.to_dict(),
            team_aliases=list(team_metadata[team_idx]["aliases"]),
        )
        if scored is not None:
            rows.append(scored.model_dump())

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).drop_duplicates(subset=["team_idx", "link", "title"])
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return frame.reset_index(drop=True)
