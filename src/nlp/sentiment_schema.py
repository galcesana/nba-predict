"""Structured article sentiment schema for live team-news scoring."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArticleSentimentScore(BaseModel):
    """Normalized article-level pregame context signals."""

    article_id: str
    team_idx: int
    published_at: str
    title: str
    source: str | None = None
    article_relevance: float = Field(ge=0.0, le=1.0)
    overall_sentiment: float = Field(ge=-1.0, le=1.0)
    injury_concern: float = Field(ge=0.0, le=1.0)
    pressure: float = Field(ge=0.0, le=1.0)
    team_cohesion: float = Field(ge=0.0, le=1.0)
    motivation: float = Field(ge=0.0, le=1.0)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    article_relevance_reason: str | None = None
    link: str | None = None
