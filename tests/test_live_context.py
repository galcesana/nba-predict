"""Tests for Phase 11 live context ingestion and scoring."""

from __future__ import annotations

import pandas as pd

from src.app.predict_today import _context_summary_from_predictions
from src.data import fetch_injuries
from src.features.injury_features import build_injury_features
from src.features.news_features import build_news_features
from src.nlp.extract_sentiment import score_articles


def _sample_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "date": "2026-05-17",
                "season": "2025-26",
                "home_team_idx": 8,
                "away_team_idx": 5,
                "game_time_utc": "2026-05-18T00:00:00Z",
            }
        ]
    )


def _sample_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "hist-1",
                "team_idx": 5,
                "date": "2026-05-10",
                "season": "2025-26",
                "net_rating": 2.0,
                "point_diff": 3.0,
            },
            {
                "game_id": "hist-2",
                "team_idx": 5,
                "date": "2026-05-11",
                "season": "2025-26",
                "net_rating": 1.0,
                "point_diff": 2.0,
            },
            {
                "game_id": "hist-3",
                "team_idx": 5,
                "date": "2026-05-12",
                "season": "2025-26",
                "net_rating": -1.0,
                "point_diff": -1.0,
            },
            {
                "game_id": "hist-4",
                "team_idx": 8,
                "date": "2026-05-10",
                "season": "2025-26",
                "net_rating": 2.5,
                "point_diff": 4.0,
            },
            {
                "game_id": "hist-5",
                "team_idx": 8,
                "date": "2026-05-11",
                "season": "2025-26",
                "net_rating": 2.0,
                "point_diff": 3.0,
            },
            {
                "game_id": "hist-6",
                "team_idx": 8,
                "date": "2026-05-12",
                "season": "2025-26",
                "net_rating": 1.5,
                "point_diff": 2.0,
            },
        ]
    )


def test_parse_injury_report_pdf_filters_to_requested_games(monkeypatch):
    """Official report parsing maps only requested matchups back onto game ids."""
    lines = [
        "05/17/2026",
        "08:00 (ET)",
        "CLE@DET",
        "Cleveland Cavaliers",
        "Mitchell, Donovan",
        "Questionable",
        "Injury/Illness - Left Ankle; Sprain",
        "Detroit Pistons",
        "NOT YET SUBMITTED",
        "05/17/2026",
        "10:00 (ET)",
        "SAS@OKC",
        "San Antonio Spurs",
        "NOT YET SUBMITTED",
        "Oklahoma City Thunder",
        "NOT YET SUBMITTED",
    ]
    monkeypatch.setattr(fetch_injuries, "_normalize_lines", lambda _: lines)

    games = _sample_games()
    parsed = fetch_injuries.parse_injury_report_pdf(
        b"ignored",
        games=games,
        source_url="https://example.com/report.pdf",
        report_generated_at="2026-05-16T11:00:00-0400",
    )

    assert parsed["game_id"].unique().tolist() == ["game-1"]
    assert set(parsed["team_idx"]) == {5, 8}
    assert "Mitchell, Donovan" in parsed["player_name"].fillna("").tolist()


def test_build_injury_features_overlays_live_reports(monkeypatch):
    """Submitted live reports override fallback features team by team."""
    reports = pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "team_idx": 5,
                "player_name": "Mitchell, Donovan",
                "status": "Questionable",
                "reason": "Injury/Illness - Left Ankle; Sprain",
                "report_submitted": True,
                "report_generated_at": "2026-05-16T11:00:00-0400",
                "source_url": "https://example.com/report.pdf",
            }
        ]
    )
    monkeypatch.setattr(
        "src.features.injury_features.load_injury_reports",
        lambda games=None, allow_live_fetch=False: reports,
    )

    features = build_injury_features(_sample_games(), _sample_logs(), allow_live_fetch=True)
    home_row = features[features["team_idx"] == 8].iloc[0]
    away_row = features[features["team_idx"] == 5].iloc[0]

    assert away_row["injury_data_available"] == 1
    assert away_row["players_questionable_count"] == 1
    assert home_row["injury_data_available"] == 0


def test_score_articles_extracts_negative_injury_signal():
    """Rule-based scoring produces structured sentiment rows for live articles."""
    articles = pd.DataFrame(
        [
            {
                "team_idx": 20,
                "published_at": "2026-05-16T15:30:00Z",
                "title": "Thunder star questionable after ankle injury",
                "summary": "Oklahoma City may be without a key scorer after a late ankle sprain.",
                "source": "Example",
                "link": "https://example.com/thunder-injury",
            }
        ]
    )

    scores = score_articles(articles)

    assert len(scores) == 1
    assert scores.iloc[0]["article_relevance"] > 0
    assert scores.iloc[0]["injury_concern"] > 0
    assert scores.iloc[0]["overall_sentiment"] < 0


def test_build_news_features_uses_live_scores(monkeypatch):
    """Live-scored articles are aggregated into non-zero news features."""
    scores = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "team_idx": 5,
                "published_at": "2026-05-17T12:00:00Z",
                "title": "Cavaliers getting healthier before Game 7",
                "source": "Example",
                "article_relevance": 0.8,
                "overall_sentiment": 0.6,
                "injury_concern": 0.1,
                "pressure": 0.8,
                "team_cohesion": 0.7,
                "motivation": 0.9,
                "llm_confidence": 0.85,
                "link": "https://example.com/cavs",
                "collected_at": "2026-05-17T13:00:00Z",
            }
        ]
    )
    monkeypatch.setattr(
        "src.features.news_features.load_news_scores",
        lambda games=None, allow_live_fetch=False: scores,
    )

    features = build_news_features(_sample_games(), allow_live_fetch=True)
    away_row = features[features["team_idx"] == 5].iloc[0]
    home_row = features[features["team_idx"] == 8].iloc[0]

    assert away_row["news_available"] == 1
    assert away_row["article_volume_24h"] >= 1
    assert away_row["weighted_sentiment_72h"] > 0
    assert home_row["news_available"] == 0


def test_context_summary_counts_live_and_partial_games():
    """Slate summaries capture mixed injury/news coverage across predictions."""
    predictions = [
        {
            "context_details": {
                "injury_mode": "live",
                "news_mode": "partial",
                "injury_report_generated_at": "2026-05-16T11:00:00-0400",
                "latest_article_at": "2026-05-16T15:30:00Z",
                "news_collected_at": "2026-05-16T16:00:00Z",
            }
        },
        {
            "context_details": {
                "injury_mode": "fallback",
                "news_mode": "live",
                "injury_report_generated_at": None,
                "latest_article_at": "2026-05-16T18:00:00Z",
                "news_collected_at": "2026-05-16T18:30:00Z",
            }
        },
    ]

    summary = _context_summary_from_predictions(predictions)

    assert summary["injury_live_games"] == 1
    assert summary["news_live_games"] == 1
    assert summary["news_partial_games"] == 1
    assert summary["latest_news_article_at"] == "2026-05-16T18:00:00Z"
