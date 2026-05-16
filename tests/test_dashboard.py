"""Tests for the Streamlit dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.app import dashboard_data, streamlit_app
from src.utils.paths import PROJECT_ROOT


def _payload_for_date(date_str: str, game_id: str = "game-1") -> dict:
    return {
        "date": date_str,
        "slate_type": "day",
        "generated_at": "2026-05-16T12:00:00Z",
        "model_version": "ensemble_v1",
        "context_summary": {
            "injury_coverage_rate": 0.5,
            "news_coverage_rate": 1.0,
        },
        "predictions": [
            {
                "game_id": game_id,
                "game_date": date_str,
                "home_team_idx": 0,
                "away_team_idx": 1,
                "home_win_probability": 0.61,
                "away_win_probability": 0.39,
                "predicted_winner": "home",
                "confidence_bucket": "medium",
                "context_details": {
                    "injury_mode": "partial",
                    "news_mode": "live",
                    "home_injury_data_available": True,
                    "away_injury_data_available": False,
                    "home_players_out": 2,
                    "away_players_out": 0,
                    "home_estimated_value_missing": 1.5,
                    "away_estimated_value_missing": 0.0,
                    "home_news_available": True,
                    "away_news_available": True,
                    "home_article_volume_24h": 3,
                    "away_article_volume_24h": 2,
                    "home_weighted_sentiment_72h": 0.2,
                    "away_weighted_sentiment_72h": -0.1,
                },
                "top_model_factors": ["recent net rating"],
                "component_outputs": {"ensemble_probability": 0.61},
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_streamlit_app_imports():
    """Streamlit app module imports with expected page definitions."""
    assert "This Week's Games" in streamlit_app.PAGES
    assert "Calibration" in streamlit_app.PAGES


def test_nested_streamlit_entry_imports():
    """Import succeeds when the nested app file is executed like Streamlit Cloud."""
    entry = PROJECT_ROOT / "src" / "app" / "streamlit_app.py"
    entry_arg = json.dumps(str(entry))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (f"import runpy; runpy.run_path({entry_arg}, run_name='__streamlit_cloud__')"),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_predictions_page_renders():
    """Latest daily predictions payload loads with at least one game."""
    payload = dashboard_data.load_latest_daily_predictions()
    assert payload is not None
    assert len(payload["predictions"]) > 0


def test_game_detail_page_renders():
    """Game detail helper returns a valid prediction entry."""
    payload = dashboard_data.load_latest_daily_predictions()
    prediction = payload["predictions"][0]
    detail = dashboard_data.get_prediction_detail(payload, prediction["game_id"])
    assert detail is not None
    assert detail["game_id"] == prediction["game_id"]
    frame = dashboard_data.build_component_output_frame(detail)
    assert not frame.empty


def test_calibration_chart_generates():
    """Calibration bins are built with probabilities in range."""
    calibration = dashboard_data.build_calibration_frame()
    assert not calibration.empty
    assert calibration["avg_pred"].between(0.0, 1.0).all()
    assert calibration["actual_rate"].between(0.0, 1.0).all()


def test_team_form_page_renders():
    """Recent team form data is available for a known team."""
    payload = dashboard_data.load_latest_daily_predictions()
    team_idx = int(payload["predictions"][0]["home_team_idx"])
    form = dashboard_data.build_team_form_frame(team_idx)
    assert not form.empty
    assert {"date", "net_rating", "point_diff", "result"}.issubset(form.columns)


def test_archive_page_loads():
    """Archive frame combines saved daily predictions and backtests."""
    archive = dashboard_data.build_archive_dataframe()
    assert not archive.empty
    assert {"source", "matchup", "home_win_probability"}.issubset(archive.columns)


def test_no_crashes_with_empty_data(tmp_path):
    """Empty directories and missing files degrade gracefully."""
    empty_daily = tmp_path / "daily"
    empty_backtests = tmp_path / "historical_backtests"
    empty_daily.mkdir()
    empty_backtests.mkdir()

    assert dashboard_data.load_latest_daily_predictions(empty_daily) is None

    archive = dashboard_data.build_archive_dataframe(
        daily_dir=empty_daily,
        backtest_dir=empty_backtests,
        games_path=tmp_path / "missing_games.parquet",
    )
    assert archive.empty

    calibration = dashboard_data.build_calibration_frame(
        path=tmp_path / "missing_predictions.parquet"
    )
    assert calibration.empty


def test_probability_display_correct():
    """Home and away win probabilities sum to one within tolerance."""
    payload = dashboard_data.load_latest_daily_predictions()
    for prediction in payload["predictions"]:
        total = prediction["home_win_probability"] + prediction["away_win_probability"]
        assert abs(total - 1.0) < 1e-4


def test_news_debug_view():
    """News summary view returns the expected columns."""
    summary = dashboard_data.build_news_summary()
    assert not summary.empty
    assert {
        "team",
        "avg_sentiment_72h",
        "avg_article_volume",
        "coverage_rate",
    }.issubset(summary.columns)


def test_live_context_summary_prefers_payload():
    """Context summary prefers the loaded payload over the manifest fallback."""
    payload = _payload_for_date("2026-05-16")
    manifest = {"context_summary": {"injury_coverage_rate": 0.0}}

    summary = dashboard_data.latest_context_summary(payload, manifest)

    assert summary["injury_coverage_rate"] == 0.5
    assert summary["news_coverage_rate"] == 1.0


def test_live_summaries_use_current_payload(monkeypatch):
    """Injury and news summary pages can render from live payload context details."""
    payload = _payload_for_date("2026-05-16")
    monkeypatch.setattr(
        dashboard_data, "load_latest_daily_predictions", lambda directory=None: payload
    )

    injury_summary = dashboard_data.build_injury_summary()
    news_summary = dashboard_data.build_news_summary()

    assert not injury_summary.empty
    assert not news_summary.empty
    assert injury_summary["avg_players_out"].max() == 2
    assert news_summary["avg_article_volume"].max() == 3


def test_prediction_source_precedence(monkeypatch, tmp_path):
    """The freshest available forecast wins across local, published, and bundled data."""
    local_dir = tmp_path / "predictions" / "daily"
    published_dir = tmp_path / "published" / "daily"
    bundled_dir = tmp_path / "bundled"

    monkeypatch.setattr(dashboard_data, "DAILY_PREDICTIONS_DIR", local_dir)
    monkeypatch.setattr(dashboard_data, "PUBLISHED_DAILY_DIR", published_dir)
    monkeypatch.setattr(dashboard_data, "BUNDLED_DATA_DIR", bundled_dir)

    _write_json(bundled_dir / "latest_daily_predictions.json", _payload_for_date("2024-01-10"))
    payload = dashboard_data.load_latest_daily_predictions()
    assert payload is not None
    assert payload["data_mode"] == "bundled"

    _write_json(published_dir / "2024-01-11.json", _payload_for_date("2024-01-11"))
    _write_json(published_dir / "latest.json", _payload_for_date("2024-01-11"))
    payload = dashboard_data.load_latest_daily_predictions()
    assert payload is not None
    assert payload["data_mode"] == "published"
    assert payload["date"] == "2024-01-11"

    _write_json(local_dir / "2024-01-10.json", _payload_for_date("2024-01-10"))
    payload = dashboard_data.load_latest_daily_predictions()
    assert payload is not None
    assert payload["data_mode"] == "published"
    assert payload["date"] == "2024-01-11"

    _write_json(local_dir / "2024-01-12.json", _payload_for_date("2024-01-12"))
    payload = dashboard_data.load_latest_daily_predictions()
    assert payload is not None
    assert payload["data_mode"] == "local"
    assert payload["date"] == "2024-01-12"


def test_archive_includes_published_daily(tmp_path):
    """Published daily slates are included in the archive view."""
    local_dir = tmp_path / "local_daily"
    published_dir = tmp_path / "published_daily"
    backtests_dir = tmp_path / "historical_backtests"
    local_dir.mkdir()
    published_dir.mkdir()
    backtests_dir.mkdir()

    _write_json(published_dir / "2024-01-15.json", _payload_for_date("2024-01-15"))
    _write_json(published_dir / "latest.json", _payload_for_date("2024-01-15"))

    archive = dashboard_data.build_archive_dataframe(
        daily_dir=local_dir,
        published_daily_dir=published_dir,
        backtest_dir=backtests_dir,
        games_path=tmp_path / "missing_games.parquet",
    )

    assert not archive.empty
    assert "published" in archive["source"].unique()


def test_forecast_status_descriptions():
    """Forecast status text reflects publish state and staleness."""
    published_payload = _payload_for_date("2026-05-16")
    published_payload["data_mode"] = "published"
    published_payload["slate_type"] = "week"
    published_payload["window_start"] = "2026-05-16"
    published_payload["window_end"] = "2026-05-22"
    published_payload["dates_with_games"] = [{"date": "2026-05-16", "games_count": 1}]

    status = dashboard_data.describe_forecast_status(
        published_payload,
        {
            "status": "published",
            "target_date": "2026-05-16",
            "latest_available_date": "2026-05-16",
            "window_start": "2026-05-16",
            "window_end": "2026-05-22",
        },
        as_of_date="2026-05-16",
    )
    assert status["state"] == "published_this_week"
    assert "Published this week" in status["message"]

    no_games = dashboard_data.describe_forecast_status(
        published_payload,
        {"status": "no_games", "target_date": "2026-05-23", "latest_available_date": "2026-05-16"},
        as_of_date="2026-05-23",
    )
    assert no_games["state"] == "no_games"
    assert "No games scheduled in this forecast window" in no_games["message"]

    stale_payload = _payload_for_date("2026-05-08")
    stale_payload["data_mode"] = "published"
    stale_payload["slate_type"] = "week"
    stale_payload["window_start"] = "2026-05-08"
    stale_payload["window_end"] = "2026-05-14"

    stale = dashboard_data.describe_forecast_status(
        stale_payload,
        {
            "status": "published",
            "target_date": "2026-05-15",
            "latest_available_date": "2026-05-15",
            "window_start": "2026-05-08",
            "window_end": "2026-05-14",
        },
        as_of_date="2026-05-16",
    )
    assert stale["state"] == "stale"
    assert "Showing previous published slate" in stale["message"]

    bundled_payload = _payload_for_date("2024-01-15")
    bundled_payload["data_mode"] = "bundled"
    bundled = dashboard_data.describe_forecast_status(
        bundled_payload,
        None,
        as_of_date="2026-05-16",
    )
    assert bundled["state"] == "bundled"
    assert "bundled example slate" in bundled["message"]


def test_dashboard_startup():
    """Streamlit server starts without import/runtime errors."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--server.port=8513",
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
