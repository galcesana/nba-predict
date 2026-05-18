"""Tests for the FastAPI service layer."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from src.app import api, dashboard_data


def _payload(predictions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "date": "2026-05-16",
        "slate_type": "week",
        "window_start": "2026-05-16",
        "window_end": "2026-05-22",
        "generated_at": "2026-05-16T21:05:15Z",
        "model_version": "ensemble_v1",
        "data_mode": "published",
        "context_summary": {
            "injury_coverage_rate": 0.5,
            "news_coverage_rate": 1.0,
        },
        "predictions": predictions
        or [
            {
                "game_id": "game-1",
                "game_date": "2026-05-17",
                "home_team_idx": 1,
                "away_team_idx": 0,
                "home_win_probability": 0.61,
                "away_win_probability": 0.39,
                "predicted_winner": "home",
                "confidence_bucket": "medium",
                "context_details": {"injury_mode": "partial", "news_mode": "live"},
                "top_model_factors": ["recent net rating"],
                "component_outputs": {"final_probability": 0.61},
            },
            {
                "game_id": "game-2",
                "game_date": "2026-05-18",
                "home_team_idx": 3,
                "away_team_idx": 2,
                "home_win_probability": 0.54,
                "away_win_probability": 0.46,
                "predicted_winner": "home",
                "confidence_bucket": "low",
                "context_details": {"injury_mode": "fallback", "news_mode": "fallback"},
                "top_model_factors": ["rest advantage"],
                "component_outputs": {"final_probability": 0.54},
            },
        ],
    }


def _manifest(status: str = "published") -> dict[str, Any]:
    return {
        "status": status,
        "target_date": "2026-05-16",
        "attempted_at": "2026-05-16T21:05:15Z",
        "latest_available_date": "2026-05-16",
        "published_file": "published/daily/2026-05-16.json",
        "games_count": 2,
        "model_version": "ensemble_v1",
        "slate_type": "week",
        "window_start": "2026-05-16",
        "window_end": "2026-05-22",
        "context_summary": {
            "injury_coverage_rate": 0.5,
            "news_coverage_rate": 1.0,
        },
    }


def _client(
    monkeypatch,
    *,
    payload: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> TestClient:
    monkeypatch.setattr(
        dashboard_data,
        "load_latest_daily_predictions",
        lambda directory=None: payload,
    )
    monkeypatch.setattr(dashboard_data, "load_publish_manifest", lambda path=None: manifest)
    return TestClient(api.create_app())


def test_health_endpoint_reports_ok(monkeypatch):
    """Health should report a live published forecast as healthy."""
    client = _client(monkeypatch, payload=_payload(), manifest=_manifest())

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["service_status"] == "ok"
    assert body["forecast_status"]["state"] == "published_this_week"
    assert body["games_count"] == 2


def test_health_endpoint_exposes_shadow_model_version(monkeypatch):
    """Health metadata should label active shadow candidates when present."""
    payload = _payload()
    payload["shadow_model_version"] = "nextgen_full_raw_v1"
    manifest = _manifest()
    manifest["shadow_model_version"] = "nextgen_full_raw_v1"
    client = _client(monkeypatch, payload=payload, manifest=manifest)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["shadow_model_version"] == "nextgen_full_raw_v1"


def test_health_endpoint_returns_503_when_forecast_is_missing(monkeypatch):
    """Missing payloads and manifests should surface as unavailable health."""
    client = _client(monkeypatch, payload=None, manifest=None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["service_status"] == "unavailable"


def test_manifest_endpoint_returns_publish_manifest(monkeypatch):
    """Manifest endpoint should expose the tracked publish manifest."""
    client = _client(monkeypatch, payload=_payload(), manifest=_manifest())

    response = client.get("/manifest")

    assert response.status_code == 200
    body = response.json()
    assert body["manifest"]["status"] == "published"
    assert body["latest_forecast_date"] == "2026-05-16"


def test_forecast_week_groups_predictions_by_date(monkeypatch):
    """Weekly forecast endpoint should bucket games by game_date."""
    client = _client(monkeypatch, payload=_payload(), manifest=_manifest())

    response = client.get("/forecast/week")

    assert response.status_code == 200
    body = response.json()
    assert body["forecast"]["games_count"] == 2
    assert "source_file" not in body["forecast"]
    assert [group["date"] for group in body["games_by_date"]] == ["2026-05-17", "2026-05-18"]


def test_forecast_week_allows_no_games_manifest(monkeypatch):
    """No-games windows should return an empty forecast response instead of failing."""
    client = _client(monkeypatch, payload=None, manifest=_manifest(status="no_games"))

    response = client.get("/forecast/week")

    assert response.status_code == 200
    body = response.json()
    assert body["forecast"] is None
    assert body["games_by_date"] == []
    assert body["forecast_status"]["state"] == "no_games"


def test_forecast_game_returns_derived_team_labels(monkeypatch):
    """Single-game endpoint should include matchup display metadata."""
    client = _client(monkeypatch, payload=_payload(), manifest=_manifest())

    response = client.get("/forecast/game/game-1")

    assert response.status_code == 200
    game = response.json()["game"]
    assert game["matchup"] == "ATL at BOS"
    assert game["home_team"] == "BOS"
    assert game["away_team"] == "ATL"
    assert game["predicted_team"] == "BOS"


def test_forecast_game_returns_404_for_unknown_game(monkeypatch):
    """Unknown game ids should return a not-found error."""
    client = _client(monkeypatch, payload=_payload(), manifest=_manifest())

    response = client.get("/forecast/game/does-not-exist")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_metrics_endpoint_serializes_metric_frames(monkeypatch):
    """Metrics endpoint should serialize model tables and calibration data."""
    monkeypatch.setattr(
        dashboard_data,
        "build_model_performance_table",
        lambda: pd.DataFrame(
            [
                {
                    "family": "ensemble",
                    "model": "ensemble_calibrated",
                    "accuracy": 0.656,
                    "log_loss": 0.615,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        dashboard_data,
        "build_calibration_frame",
        lambda: pd.DataFrame(
            [
                {
                    "bin_mid": 0.55,
                    "avg_pred": 0.58,
                    "actual_rate": 0.6,
                    "count": 10,
                    "ideal": 0.55,
                    "abs_gap": 0.02,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        dashboard_data,
        "build_rolling_validation_frame",
        lambda: pd.DataFrame(
            [{"season": "2025-26", "log_loss": 0.62, "accuracy": 0.64}]
        ),
    )
    monkeypatch.setattr(
        dashboard_data,
        "build_ensemble_weights_frame",
        lambda: pd.DataFrame([{"model": "sequence", "weight": 0.7}]),
    )
    client = _client(monkeypatch, payload=_payload(), manifest=_manifest())

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["best_models"]["lowest_log_loss"]["model"] == "ensemble_calibrated"
    assert body["calibration"]["expected_calibration_error"] == 0.02
    assert body["ensemble_weights"][0]["model"] == "sequence"
    assert body["rolling_validation"][0]["season"] == "2025-26"
