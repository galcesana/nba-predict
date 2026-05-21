"""Tests for Phase 13 Context Store V1."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.app import publish_today
from src.context_store.schema import (
    PREGAME_TABLES,
    TABLE_SCHEMAS,
    initialize_context_store,
    table_columns,
)
from src.context_store.writer import save_forecast_context


def _payload() -> dict:
    return {
        "date": "2026-05-21",
        "slate_type": "week",
        "window_start": "2026-05-21",
        "window_end": "2026-05-27",
        "generated_at": "2026-05-21T19:57:27Z",
        "model_version": "nextgen_full_value_tuned_v2",
        "shadow_model_version": "nextgen_full_value_tuned_v2",
        "predictions": [
            {
                "game_id": "0042500302",
                "game_date": "2026-05-21",
                "game_time_utc": "2026-05-22T00:00:00Z",
                "season": "2025-26",
                "home_team_idx": 19,
                "away_team_idx": 5,
                "home_team_abbr": "NYK",
                "away_team_abbr": "CLE",
                "game_label": "East Conf. Finals",
                "game_sub_label": "Game 2",
                "series_text": "NYK leads 1-0",
                "game_status_text": "8:00 pm ET",
                "home_win_probability": 0.6893,
                "away_win_probability": 0.3107,
                "predicted_winner": "home",
                "confidence_bucket": "medium",
                "component_outputs": {
                    "elo_probability": 0.57,
                    "tabular_probability": 0.61,
                    "sequence_probability": 0.7714,
                    "ensemble_v1_probability": 0.66,
                    "nextgen_shadow_probability": 0.6893,
                    "final_probability": 0.6893,
                },
                "context_details": {
                    "injury_mode": "partial",
                    "news_mode": "live",
                    "home_injury_data_available": True,
                    "away_injury_data_available": False,
                    "home_players_out": 1,
                    "away_players_out": 0,
                    "home_questionable": 0,
                    "away_questionable": 1,
                    "home_estimated_value_missing": 1.4,
                    "away_estimated_value_missing": 0.0,
                    "home_injury_report_status": "available",
                    "away_injury_report_status": "not_submitted",
                    "injury_report_generated_at": "2026-05-21T16:30:00-0400",
                    "injury_report_source_url": "https://example.com/injury.pdf",
                    "home_news_available": True,
                    "away_news_available": True,
                    "home_article_volume_24h": 8,
                    "away_article_volume_24h": 6,
                    "home_weighted_sentiment_72h": 0.2624,
                    "away_weighted_sentiment_72h": -0.0379,
                    "home_article_count_72h": 8,
                    "away_article_count_72h": 6,
                    "latest_article_at": "2026-05-21T19:48:05Z",
                    "news_collected_at": "2026-05-21T19:57:27Z",
                },
                "top_model_factors": [
                    "Home team has significantly better recent net rating",
                    "Home team has more positive recent news sentiment",
                ],
            }
        ],
    }


def _manifest() -> dict:
    return {
        "status": "published",
        "target_date": "2026-05-21",
        "attempted_at": "2026-05-21T19:57:20Z",
        "latest_available_date": "2026-05-21",
        "games_count": 1,
        "model_version": "nextgen_full_value_tuned_v2",
        "shadow_model_version": "nextgen_full_value_tuned_v2",
        "slate_type": "week",
        "window_start": "2026-05-21",
        "window_end": "2026-05-27",
        "publish_observability": {
            "timezone": "America/New_York",
            "api_status": {
                "schedule": "ok",
                "injury": "partial",
                "news": "live",
            },
        },
    }


def _write_news_cache(news_dir: Path) -> None:
    news_dir.mkdir(parents=True, exist_ok=True)
    articles = pd.DataFrame(
        [
            {
                "article_id": "included-article",
                "team_idx": 19,
                "published_at": "2026-05-21T19:48:05Z",
                "collected_at": "2026-05-21T19:57:27Z",
                "source": "Newsday",
                "title": "Landry Shamet etches himself in Knicks playoff lore",
                "summary": "Rotation note before Game 2.",
                "link": "https://example.com/included",
                "included_in_model": True,
                "excluded_reason": "",
                "article_relevance": 0.46,
                "article_relevance_reason": "team mention plus matchup/injury context",
            },
            {
                "article_id": "excluded-article",
                "team_idx": 5,
                "published_at": "2026-05-21T19:40:00Z",
                "collected_at": "2026-05-21T19:57:27Z",
                "source": "DraftKings Network",
                "title": "Best Donovan Mitchell prop bet for Cavaliers vs Knicks",
                "summary": "Odds and prop bets.",
                "link": "https://example.com/excluded",
                "included_in_model": False,
                "excluded_reason": "betting_or_promo",
                "article_relevance": 0.38,
                "article_relevance_reason": "team-specific mention",
            },
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "article_id": "included-article",
                "team_idx": 19,
                "published_at": "2026-05-21T19:48:05Z",
                "collected_at": "2026-05-21T19:57:27Z",
                "overall_sentiment": 1.0,
                "injury_concern": 0.0,
                "pressure": 0.33,
                "team_cohesion": 0.5,
                "motivation": 0.25,
                "llm_confidence": 0.6,
            }
        ]
    )
    articles.to_parquet(news_dir / "team_news_articles_2026-05-21.parquet", index=False)
    scores.to_parquet(news_dir / "team_news_scores_2026-05-21.parquet", index=False)


def _table_count(db_path: Path, table_name: str) -> int:
    with duckdb.connect(str(db_path)) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def test_context_store_schema_initializes(tmp_path):
    """Context store creates all tables and Parquet directories."""
    db_path, parquet_root = initialize_context_store(
        db_path=tmp_path / "context.duckdb",
        parquet_root=tmp_path / "parquet",
    )

    assert db_path.exists()
    for table_name in TABLE_SCHEMAS:
        assert (parquet_root / table_name).is_dir()

    with duckdb.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

    assert set(TABLE_SCHEMAS) == tables


def test_pregame_tables_do_not_contain_outcomes():
    """Outcome fields must stay isolated from pregame context tables."""
    forbidden = {"final_home_score", "final_away_score", "home_win", "completed_at"}
    for table_name in PREGAME_TABLES:
        assert forbidden.isdisjoint(table_columns(table_name))


def test_save_forecast_context_appends_publish_snapshot(tmp_path):
    """A successful publish payload is written into the context store."""
    news_dir = tmp_path / "raw_news"
    _write_news_cache(news_dir)
    db_path = tmp_path / "context.duckdb"
    parquet_root = tmp_path / "parquet"

    run_id = save_forecast_context(
        _payload(),
        _manifest(),
        db_path=db_path,
        parquet_root=parquet_root,
        run_id="run-1",
        news_dir=news_dir,
    )

    assert run_id == "run-1"
    assert _table_count(db_path, "forecast_runs") == 1
    assert _table_count(db_path, "game_snapshots") == 1
    assert _table_count(db_path, "prediction_outputs") == 1
    assert _table_count(db_path, "injury_context") == 2
    assert _table_count(db_path, "news_articles") == 2
    assert _table_count(db_path, "news_scores") == 1
    assert _table_count(db_path, "model_features") > 0
    assert (parquet_root / "forecast_runs" / "run-1.parquet").exists()


def test_news_article_exclusions_are_persisted(tmp_path):
    """Betting/promo rows are stored as excluded, not lost."""
    news_dir = tmp_path / "raw_news"
    _write_news_cache(news_dir)
    db_path = tmp_path / "context.duckdb"

    save_forecast_context(
        _payload(),
        _manifest(),
        db_path=db_path,
        parquet_root=tmp_path / "parquet",
        run_id="run-exclusions",
        news_dir=news_dir,
    )

    with duckdb.connect(str(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT article_id, included_in_model, excluded_reason
            FROM news_articles
            ORDER BY article_id
            """
        ).fetchall()

    assert ("excluded-article", False, "betting_or_promo") in rows
    assert ("included-article", True, None) in rows


def test_duplicate_run_id_is_rejected(tmp_path):
    """Context store should not silently append the same run twice."""
    db_path = tmp_path / "context.duckdb"
    kwargs = {
        "db_path": db_path,
        "parquet_root": tmp_path / "parquet",
        "run_id": "run-duplicate",
        "news_dir": tmp_path / "missing_news",
    }

    save_forecast_context(_payload(), _manifest(), **kwargs)
    with pytest.raises(ValueError, match="already exists"):
        save_forecast_context(_payload(), _manifest(), **kwargs)


def test_publish_integration_writes_context_when_explicitly_configured(tmp_path):
    """Publishing can append context without changing the published JSON contract."""

    def fake_generator(date_str: str, output_dir: Path, **_: object):
        payload = _payload()
        payload["date"] = date_str
        payload["window_start"] = date_str
        out_path = output_dir / f"{date_str}.json"
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, out_path

    db_path = tmp_path / "context.duckdb"
    parquet_root = tmp_path / "parquet"

    manifest, paths = publish_today.publish_predictions_for_date(
        "2026-05-21",
        published_root=tmp_path / "published",
        prediction_generator=fake_generator,
        context_store_db_path=db_path,
        context_store_parquet_root=parquet_root,
    )

    assert manifest["status"] == "published"
    assert {path.name for path in paths} == {
        "2026-05-21.json",
        "latest.json",
        "manifest.json",
    }
    assert _table_count(db_path, "forecast_runs") == 1
