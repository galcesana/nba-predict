"""Tests for the Phase 9 Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.app import dashboard_data, streamlit_app
from src.utils.paths import PROJECT_ROOT


def test_streamlit_app_imports():
    """Streamlit app module imports with expected page definitions."""
    assert "Today's Games" in streamlit_app.PAGES
    assert "Calibration" in streamlit_app.PAGES


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


def test_dashboard_startup():
    """Streamlit server starts without import/runtime errors."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/app/streamlit_app.py",
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
