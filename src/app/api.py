"""FastAPI service for published forecasts and model diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.app import dashboard_data
from src.utils.logging import setup_logging

API_TITLE = "NBA Predict API"
API_VERSION = "0.1.0"


def _to_builtin(value: Any) -> Any:
    """Convert pandas/numpy-rich objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {
            str(key): _to_builtin(item)
            for key, item in value.items()
            if key != "source_file"
        }
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, float):
        return None if pd.isna(value) else value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a dataframe into API-friendly record dictionaries."""
    if frame.empty:
        return []
    records = frame.to_dict(orient="records")
    return [_to_builtin(record) for record in records]


def _decorate_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Add display-friendly team labels to a prediction record."""
    enriched = dict(prediction)
    home_team = dashboard_data.team_abbr(enriched.get("home_team_idx"))
    away_team = dashboard_data.team_abbr(enriched.get("away_team_idx"))
    enriched["home_team"] = home_team
    enriched["away_team"] = away_team
    enriched["matchup"] = dashboard_data.matchup_label(
        enriched.get("home_team_idx"),
        enriched.get("away_team_idx"),
    )

    predicted_winner = str(enriched.get("predicted_winner") or "")
    if predicted_winner == "home":
        enriched["predicted_team"] = home_team
    elif predicted_winner == "away":
        enriched["predicted_team"] = away_team
    else:
        enriched["predicted_team"] = None

    return _to_builtin(enriched)


def _group_predictions_by_date(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Group the current forecast into date buckets for easier API clients."""
    if payload is None:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction in payload.get("predictions", []):
        game_date = str(prediction.get("game_date") or payload.get("date") or "")
        grouped.setdefault(game_date, []).append(_decorate_prediction(prediction))

    return [
        {
            "date": game_date,
            "games_count": len(games),
            "games": games,
        }
        for game_date, games in sorted(grouped.items())
    ]


def _current_forecast_bundle(
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, str]]:
    """Load the latest forecast payload, publish manifest, and derived status."""
    payload = dashboard_data.load_latest_daily_predictions()
    manifest = dashboard_data.load_publish_manifest()
    forecast_status = dashboard_data.describe_forecast_status(payload, manifest)
    return payload, manifest, forecast_status


def _public_forecast_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Prepare the current forecast payload for API delivery."""
    if payload is None:
        return None

    public_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"source_file", "predictions"}
    }
    public_payload["games_count"] = len(payload.get("predictions", []))
    public_payload["predictions"] = [
        _decorate_prediction(prediction) for prediction in payload.get("predictions", [])
    ]
    return _to_builtin(public_payload)


def _best_model(frame: pd.DataFrame, metric: str, *, ascending: bool) -> dict[str, Any] | None:
    """Return the best model row for a given metric when available."""
    if frame.empty or metric not in frame.columns:
        return None
    candidate = frame.dropna(subset=[metric])
    if candidate.empty:
        return None
    best = candidate.sort_values(metric, ascending=ascending).iloc[0].to_dict()
    return _to_builtin(best)


def _service_status_for(forecast_state: str) -> str:
    """Map forecast freshness into a coarse service health state."""
    if forecast_state == "unavailable":
        return "unavailable"
    if forecast_state in {"bundled", "stale"}:
        return "degraded"
    return "ok"


def create_app() -> FastAPI:
    """Create the API application."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        summary="Serve published NBA forecasts, manifest status, and model metrics.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "nba-predict-api",
            "version": API_VERSION,
            "docs_url": "/docs",
            "endpoints": [
                "/health",
                "/manifest",
                "/forecast/week",
                "/forecast/game/{game_id}",
                "/metrics",
            ],
        }

    @app.get("/health")
    def health(response: Response) -> dict[str, Any]:
        payload, manifest, forecast_status = _current_forecast_bundle()
        service_status = _service_status_for(forecast_status["state"])
        if service_status == "unavailable":
            response.status_code = 503

        return {
            "service": "nba-predict-api",
            "service_status": service_status,
            "forecast_status": forecast_status,
            "data_mode": payload.get("data_mode") if payload else None,
            "target_date": (manifest or {}).get("target_date"),
            "latest_available_date": (manifest or {}).get("latest_available_date"),
            "window_start": (
                (payload or {}).get("window_start")
                or (manifest or {}).get("window_start")
            ),
            "window_end": (
                (payload or {}).get("window_end")
                or (manifest or {}).get("window_end")
            ),
            "games_count": len((payload or {}).get("predictions", [])),
            "model_version": (payload or {}).get("model_version") or (manifest or {}).get(
                "model_version"
            ),
            "shadow_model_version": (payload or {}).get("shadow_model_version")
            or (manifest or {}).get("shadow_model_version"),
            "attempted_at": (manifest or {}).get("attempted_at"),
            "publish_observability": _to_builtin(
                (manifest or {}).get("publish_observability")
            ),
            "context_summary": _to_builtin(
                dashboard_data.latest_context_summary(payload, manifest)
            ),
        }

    @app.get("/manifest")
    def manifest() -> dict[str, Any]:
        payload, publish_manifest, forecast_status = _current_forecast_bundle()
        if publish_manifest is None:
            raise HTTPException(status_code=404, detail="Publish manifest not found.")

        return {
            "forecast_status": forecast_status,
            "manifest": _to_builtin(publish_manifest),
            "latest_forecast_date": (payload or {}).get("date"),
        }

    @app.get("/forecast/week")
    def forecast_week() -> dict[str, Any]:
        payload, publish_manifest, forecast_status = _current_forecast_bundle()
        if payload is None and (publish_manifest or {}).get("status") != "no_games":
            raise HTTPException(status_code=503, detail="No forecast payload is available.")

        return {
            "forecast_status": forecast_status,
            "manifest": _to_builtin(publish_manifest),
            "forecast": _public_forecast_payload(payload),
            "games_by_date": _group_predictions_by_date(payload),
        }

    @app.get("/forecast/game/{game_id}")
    def forecast_game(game_id: str) -> dict[str, Any]:
        payload, publish_manifest, forecast_status = _current_forecast_bundle()
        if payload is None:
            if (publish_manifest or {}).get("status") == "no_games":
                raise HTTPException(
                    status_code=404,
                    detail="No forecasted games are available in the current publish window.",
                )
            raise HTTPException(status_code=503, detail="No forecast payload is available.")

        prediction = dashboard_data.get_prediction_detail(payload, game_id)
        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game '{game_id}' was not found in the current forecast.",
            )

        return {
            "forecast_status": forecast_status,
            "manifest": _to_builtin(publish_manifest),
            "forecast_date": payload.get("date"),
            "window_start": payload.get("window_start"),
            "window_end": payload.get("window_end"),
            "game": _decorate_prediction(prediction),
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        payload, publish_manifest, _ = _current_forecast_bundle()
        performance = dashboard_data.build_model_performance_table()
        calibration = dashboard_data.build_calibration_frame()
        rolling = dashboard_data.build_rolling_validation_frame()
        weights = dashboard_data.build_ensemble_weights_frame()

        return {
            "model_performance": _frame_records(performance),
            "best_models": {
                "lowest_log_loss": _best_model(performance, "log_loss", ascending=True),
                "highest_accuracy": _best_model(performance, "accuracy", ascending=False),
            },
            "calibration": {
                "expected_calibration_error": _to_builtin(
                    dashboard_data.compute_expected_calibration_error(calibration)
                ),
                "bins": _frame_records(calibration),
            },
            "ensemble_weights": _frame_records(weights),
            "rolling_validation": _frame_records(rolling),
            "live_context": _to_builtin(
                dashboard_data.latest_context_summary(payload, publish_manifest)
            ),
        }

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    """Run the local API service."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run the NBA Predict API service.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    uvicorn.run(
        "src.app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
