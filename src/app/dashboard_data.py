"""Data loaders and view-model helpers for the dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.anonymization.team_mapping import load_idx_to_team
from src.utils.paths import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    PUBLISHED_DIR,
)

DAILY_PREDICTIONS_DIR = PREDICTIONS_DIR / "daily"
BACKTEST_DIR = PREDICTIONS_DIR / "historical_backtests"
PUBLISHED_DAILY_DIR = PUBLISHED_DIR / "daily"
PUBLISHED_MANIFEST_PATH = PUBLISHED_DIR / "manifest.json"
BUNDLED_DATA_DIR = PROJECT_ROOT / "src" / "app" / "bundled_data"
DEFAULT_PUBLISH_TIMEZONE = "America/New_York"


def _bundled_path(filename: str) -> Path:
    return BUNDLED_DATA_DIR / filename


def team_abbr(team_idx: Any) -> str:
    """Return a team abbreviation for a team index."""
    try:
        idx = int(team_idx)
    except (TypeError, ValueError):
        return "UNK"
    return load_idx_to_team().get(idx, f"TEAM_{idx}")


def matchup_label(home_team_idx: Any, away_team_idx: Any) -> str:
    """Return a display label for a matchup."""
    return f"{team_abbr(away_team_idx)} at {team_abbr(home_team_idx)}"


def _json_files(directory: Path, *, exclude_names: set[str] | None = None) -> list[Path]:
    if not directory.exists():
        return []
    excluded = exclude_names or set()
    return sorted(
        path for path in directory.glob("*.json") if path.is_file() and path.name not in excluded
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_payload_from_directory(
    directory: Path,
    *,
    data_mode: str,
    exclude_names: set[str] | None = None,
    preferred_filename: str | None = None,
) -> dict[str, Any] | None:
    files = _json_files(directory, exclude_names=exclude_names)
    if not files and not preferred_filename:
        return None

    preferred_path = directory / preferred_filename if preferred_filename else None
    if preferred_path and preferred_path.exists():
        source_path = preferred_path
    elif files:
        source_path = files[-1]
    else:
        return None

    payload = _load_json(source_path)
    payload["source_file"] = str(source_path)
    payload["data_mode"] = data_mode
    return payload


def _payload_recency_key(payload: dict[str, Any]) -> tuple[str, str]:
    window_end = str(payload.get("window_end") or payload.get("date") or "")
    generated_at = str(payload.get("generated_at") or "")
    return window_end, generated_at


def _load_frame_from_json(path: Path, *, date_columns: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def _load_frame(
    primary: Path,
    *,
    bundled_filename: str | None = None,
    date_columns: tuple[str, ...] = ("date",),
    allow_bundled: bool = True,
) -> pd.DataFrame:
    if primary.exists():
        if primary.suffix == ".json":
            return _load_frame_from_json(primary, date_columns=date_columns)
        frame = pd.read_parquet(primary)
        for column in date_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        return frame

    if allow_bundled and bundled_filename:
        return _load_frame_from_json(_bundled_path(bundled_filename), date_columns=date_columns)

    return pd.DataFrame()


def load_games_table(path: Path | None = None) -> pd.DataFrame:
    """Load the processed games table."""
    source = path or (PROCESSED_DIR / "games.parquet")
    return _load_frame(
        source,
        bundled_filename="games_snapshot.json",
        date_columns=("date",),
        allow_bundled=path is None,
    )


def load_team_logs(path: Path | None = None) -> pd.DataFrame:
    """Load processed team game logs."""
    source = path or (PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    return _load_frame(
        source,
        bundled_filename="team_game_logs_snapshot.json",
        date_columns=("date",),
        allow_bundled=path is None,
    )


def load_injury_features(path: Path | None = None) -> pd.DataFrame:
    """Load processed injury features."""
    source = path or (PROCESSED_DIR / "injury_features" / "injury_features.parquet")
    return _load_frame(
        source,
        bundled_filename="injury_features_snapshot.json",
        date_columns=(),
        allow_bundled=path is None,
    )


def load_news_features(path: Path | None = None) -> pd.DataFrame:
    """Load processed news features."""
    source = path or (PROCESSED_DIR / "news_features" / "news_features.parquet")
    return _load_frame(
        source,
        bundled_filename="news_features_snapshot.json",
        date_columns=(),
        allow_bundled=path is None,
    )


def load_latest_daily_predictions(directory: Path | None = None) -> dict[str, Any] | None:
    """Load the most recent daily prediction payload."""
    local_payload = _latest_payload_from_directory(
        directory or DAILY_PREDICTIONS_DIR,
        data_mode="local",
    )

    if directory is not None:
        return local_payload

    published_payload = _latest_payload_from_directory(
        PUBLISHED_DAILY_DIR,
        data_mode="published",
        exclude_names={"latest.json"},
        preferred_filename="latest.json",
    )
    if local_payload and published_payload:
        if _payload_recency_key(local_payload) >= _payload_recency_key(published_payload):
            return local_payload
        return published_payload
    if local_payload:
        return local_payload
    if published_payload:
        return published_payload

    bundled = _bundled_path("latest_daily_predictions.json")
    if not bundled.exists():
        return None

    payload = _load_json(bundled)
    payload["source_file"] = str(bundled)
    payload["data_mode"] = "bundled"
    return payload


def load_backtest_reports(directory: Path | None = None) -> list[dict[str, Any]]:
    """Load all saved backtest reports."""
    reports = []
    for path in _json_files(directory or BACKTEST_DIR):
        report = _load_json(path)
        report["source_file"] = str(path)
        report["data_mode"] = "published"
        reports.append(report)
    if reports or directory is not None:
        return reports

    bundled = _bundled_path("backtest_snapshot.json")
    if bundled.exists():
        report = _load_json(bundled)
        report["source_file"] = str(bundled)
        report["data_mode"] = "bundled"
        reports.append(report)
    return reports


def load_publish_manifest(path: Path | None = None) -> dict[str, Any] | None:
    """Load the latest publish manifest when available."""
    source = path or PUBLISHED_MANIFEST_PATH
    if not source.exists():
        return None
    return _load_json(source)


def latest_context_summary(
    payload: dict[str, Any] | None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the best available live-context summary for the current slate."""
    payload_summary = payload.get("context_summary") if payload else None
    if isinstance(payload_summary, dict):
        return payload_summary

    manifest_summary = (manifest or {}).get("context_summary")
    if isinstance(manifest_summary, dict):
        return manifest_summary

    return {}


def current_publish_date(timezone_name: str = DEFAULT_PUBLISH_TIMEZONE) -> str:
    """Return today's date in the publishing timezone."""
    return datetime.now(ZoneInfo(timezone_name)).date().isoformat()


def describe_forecast_status(
    payload: dict[str, Any] | None,
    manifest: dict[str, Any] | None = None,
    *,
    as_of_date: str | None = None,
    timezone_name: str = DEFAULT_PUBLISH_TIMEZONE,
) -> dict[str, str]:
    """Describe the currently loaded forecast source for the dashboard."""
    if payload is None:
        if manifest and manifest.get("status") == "no_games":
            return {"state": "no_games", "message": "No games scheduled in this forecast window."}
        return {"state": "unavailable", "message": "Waiting for the next published forecast."}

    mode = payload.get("data_mode")
    if mode == "bundled":
        return {"state": "bundled", "message": "Showing bundled example slate."}
    if mode == "local":
        return {"state": "local", "message": "Showing local forecast preview."}

    publish_date = as_of_date or current_publish_date(timezone_name)
    payload_date = str(payload.get("date", "unknown"))
    manifest = manifest or {}
    status = str(manifest.get("status", ""))
    latest_available = manifest.get("latest_available_date")
    window_start = str(payload.get("window_start") or manifest.get("window_start") or payload_date)
    window_end = str(payload.get("window_end") or manifest.get("window_end") or payload_date)

    if status == "published" and window_start <= publish_date <= window_end:
        return {
            "state": "published_this_week",
            "message": f"Published this week. Window: {window_start} to {window_end}.",
        }

    if status == "no_games":
        if latest_available:
            return {
                "state": "no_games",
                "message": (
                    "No games scheduled in this forecast window. "
                    f"Showing previous published slate from {latest_available}."
                ),
            }
        return {"state": "no_games", "message": "No games scheduled in this forecast window."}

    if window_end < publish_date:
        return {
            "state": "stale",
            "message": (f"Showing previous published slate from {window_start} to {window_end}."),
        }

    return {
        "state": "published",
        "message": f"Published slate for {window_start} to {window_end}.",
    }


def get_prediction_detail(payload: dict[str, Any] | None, game_id: str) -> dict[str, Any] | None:
    """Return a single prediction entry from a payload."""
    if not payload:
        return None
    for prediction in payload.get("predictions", []):
        if prediction.get("game_id") == game_id:
            return prediction
    return None


def build_component_output_frame(prediction: dict[str, Any]) -> pd.DataFrame:
    """Convert component model outputs into a display table."""
    outputs = prediction.get("component_outputs", {})
    rows = []
    for model_name, probability in outputs.items():
        label = model_name.replace("_", " ").replace("probability", "prob").title()
        rows.append({"model": label, "probability": float(probability)})
    return pd.DataFrame(rows)


def _prediction_record(
    prediction: dict[str, Any],
    *,
    date: Any,
    source: str,
    generated_at: str | None = None,
    actual_home_win: Any | None = None,
) -> dict[str, Any]:
    home_prob = float(prediction.get("home_win_probability", np.nan))
    away_prob = float(prediction.get("away_win_probability", 1.0 - home_prob))
    record = {
        "date": pd.to_datetime(date),
        "source": source,
        "game_id": prediction.get("game_id"),
        "matchup": matchup_label(prediction.get("home_team_idx"), prediction.get("away_team_idx")),
        "home_team": team_abbr(prediction.get("home_team_idx")),
        "away_team": team_abbr(prediction.get("away_team_idx")),
        "home_team_idx": int(prediction.get("home_team_idx")),
        "away_team_idx": int(prediction.get("away_team_idx")),
        "home_win_probability": home_prob,
        "away_win_probability": away_prob,
        "predicted_winner": prediction.get("predicted_winner"),
        "confidence_bucket": prediction.get("confidence_bucket"),
        "generated_at": generated_at,
        "actual_home_win": actual_home_win,
        "top_model_factors": ", ".join(prediction.get("top_model_factors", [])),
    }
    if actual_home_win in (0, 1):
        record["actual_winner"] = "home" if int(actual_home_win) == 1 else "away"
    else:
        record["actual_winner"] = None
    return record


def build_archive_dataframe(
    daily_dir: Path | None = None,
    published_daily_dir: Path | None = None,
    backtest_dir: Path | None = None,
    games_path: Path | None = None,
) -> pd.DataFrame:
    """Build a combined archive frame from daily predictions and backtests."""
    rows: list[dict[str, Any]] = []

    if daily_dir is None and published_daily_dir is None:
        daily_sources = [
            ("local", DAILY_PREDICTIONS_DIR, set()),
            ("published", PUBLISHED_DAILY_DIR, {"latest.json"}),
        ]
    else:
        daily_sources = []
        if daily_dir is not None:
            daily_sources.append(("local", Path(daily_dir), set()))
        if published_daily_dir is not None:
            daily_sources.append(("published", Path(published_daily_dir), {"latest.json"}))

    for source_label, directory, excluded in daily_sources:
        for path in _json_files(directory, exclude_names=excluded):
            payload = _load_json(path)
            for prediction in payload.get("predictions", []):
                rows.append(
                    _prediction_record(
                        prediction,
                        date=prediction.get("game_date", payload.get("date")),
                        source=source_label,
                        generated_at=payload.get("generated_at"),
                    )
                )

    games = load_games_table(games_path)
    if {"game_id", "date"}.issubset(games.columns):
        game_dates = games[["game_id", "date"]]
        date_lookup = dict(zip(game_dates["game_id"], game_dates["date"]))
    else:
        date_lookup = {}

    for report in load_backtest_reports(backtest_dir or BACKTEST_DIR):
        for prediction in report.get("predictions", []):
            rows.append(
                _prediction_record(
                    prediction,
                    date=date_lookup.get(prediction.get("game_id"), report.get("start_date")),
                    source="backtest",
                    actual_home_win=prediction.get("actual_home_win"),
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "source",
                "game_id",
                "matchup",
                "home_win_probability",
                "away_win_probability",
                "predicted_winner",
                "actual_winner",
                "confidence_bucket",
            ]
        )

    archive = pd.DataFrame(rows).sort_values(
        ["date", "game_id", "source"],
        ascending=[False, True, True],
    )
    archive["date"] = pd.to_datetime(archive["date"])
    return archive.reset_index(drop=True)


def build_model_performance_table(
    baseline_path: Path | None = None,
    ensemble_path: Path | None = None,
    neural_path: Path | None = None,
    ablation_path: Path | None = None,
) -> pd.DataFrame:
    """Build a unified model comparison table for dashboard display."""
    rows: list[dict[str, Any]] = []

    baseline_source = baseline_path or (MODELS_DIR / "baselines" / "evaluation_results.json")
    if baseline_source.exists():
        baseline_results = _load_json(baseline_source)
        for name, metrics in baseline_results.items():
            if not name.endswith("_test") or not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "family": "baseline",
                    "model": name.replace("_test", ""),
                    **metrics,
                }
            )

    ensemble_source = ensemble_path or (MODELS_DIR / "ensembles" / "ensemble_results.json")
    if ensemble_source.exists():
        ensemble_results = _load_json(ensemble_source)
        for name in ("ensemble_raw", "ensemble_calibrated"):
            if name in ensemble_results:
                rows.append(
                    {
                        "family": "ensemble",
                        "model": name,
                        **ensemble_results[name],
                    }
                )

    neural_source = neural_path or (MODELS_DIR / "neural" / "test_results.json")
    if neural_source.exists():
        neural_results = _load_json(neural_source)
        rows.append(
            {
                "family": "neural",
                "model": "full_fusion",
                "accuracy": neural_results.get("test_accuracy"),
                "log_loss": neural_results.get("test_loss"),
                "brier_score": np.nan,
                "roc_auc": np.nan,
                "calibration_error": np.nan,
            }
        )

    ablation_source = ablation_path or (MODELS_DIR / "neural" / "ablation_results.json")
    if ablation_source.exists():
        ablations = _load_json(ablation_source)
        for name, metrics in ablations.items():
            rows.append(
                {
                    "family": "ablation",
                    "model": name,
                    "accuracy": metrics.get("test_accuracy"),
                    "log_loss": metrics.get("test_loss"),
                    "brier_score": np.nan,
                    "roc_auc": np.nan,
                    "calibration_error": np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    numeric_cols = ["accuracy", "log_loss", "brier_score", "roc_auc", "calibration_error"]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["log_loss", "accuracy"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_rolling_validation_frame(path: Path | None = None) -> pd.DataFrame:
    """Load rolling validation metrics from the baseline results file."""
    source = path or (MODELS_DIR / "baselines" / "evaluation_results.json")
    if not source.exists():
        return pd.DataFrame()
    payload = _load_json(source)
    rolling = pd.DataFrame(payload.get("rolling_validation", []))
    return rolling


def build_ensemble_weights_frame(path: Path | None = None) -> pd.DataFrame:
    """Build a display frame for ensemble weights."""
    source = path or (MODELS_DIR / "ensembles" / "ensemble_results.json")
    if not source.exists():
        return pd.DataFrame()
    payload = _load_json(source)
    weights = payload.get("model_weights", {})
    if not weights:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [{"model": name, "weight": float(weight)} for name, weight in weights.items()]
    )
    return frame.sort_values("weight", ascending=False).reset_index(drop=True)


def build_calibration_frame(path: Path | None = None, n_bins: int = 10) -> pd.DataFrame:
    """Build calibration bins from ensemble test predictions."""
    source = path or (MODELS_DIR / "ensembles" / "ensemble_predictions_test.parquet")
    if not source.exists():
        if path is None:
            return _load_frame_from_json(
                _bundled_path("calibration_snapshot.json"),
                date_columns=(),
            )
        return pd.DataFrame(
            columns=["bin_mid", "avg_pred", "actual_rate", "count", "ideal", "abs_gap"]
        )

    predictions = pd.read_parquet(source)
    if predictions.empty:
        return pd.DataFrame(
            columns=["bin_mid", "avg_pred", "actual_rate", "count", "ideal", "abs_gap"]
        )

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    calibration = predictions.copy()
    calibration["bin"] = pd.cut(
        calibration["ensemble_prob"],
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    )
    grouped = (
        calibration.groupby("bin", observed=True)
        .agg(
            avg_pred=("ensemble_prob", "mean"),
            actual_rate=("actual_home_win", "mean"),
            count=("game_id", "size"),
        )
        .reset_index()
    )
    grouped["bin_mid"] = grouped["bin"].apply(
        lambda interval: float((interval.left + interval.right) / 2)
    )
    grouped["ideal"] = grouped["bin_mid"]
    grouped["abs_gap"] = (grouped["avg_pred"] - grouped["actual_rate"]).abs()
    return grouped[["bin_mid", "avg_pred", "actual_rate", "count", "ideal", "abs_gap"]]


def compute_expected_calibration_error(calibration_frame: pd.DataFrame) -> float:
    """Compute expected calibration error from a calibration frame."""
    if calibration_frame.empty:
        return float("nan")
    total = calibration_frame["count"].sum()
    if total == 0:
        return float("nan")
    return float((calibration_frame["abs_gap"] * calibration_frame["count"]).sum() / total)


def build_team_form_frame(
    team_idx: int,
    logs_path: Path | None = None,
    window: int = 10,
) -> pd.DataFrame:
    """Build a recent form frame for a single team."""
    logs = load_team_logs(logs_path)
    if logs.empty:
        return pd.DataFrame()
    team_logs = logs[logs["team_idx"] == int(team_idx)].sort_values("date").tail(window).copy()
    if team_logs.empty:
        return pd.DataFrame()
    team_logs["opponent"] = team_logs["opponent_team_idx"].map(team_abbr)
    team_logs["result"] = np.where(team_logs["won"] == 1, "W", "L")
    team_logs["location"] = np.where(team_logs["is_home"] == 1, "Home", "Away")
    return team_logs.reset_index(drop=True)


def _join_feature_dates(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach date/season columns from team logs to team-level feature frames."""
    if feature_frame.empty:
        return feature_frame
    logs = load_team_logs()[["game_id", "team_idx", "date", "season"]].drop_duplicates()
    merged = feature_frame.merge(logs, on=["game_id", "team_idx"], how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def build_injury_summary(window: int = 15, path: Path | None = None) -> pd.DataFrame:
    """Build a team-level injury summary from the most recent games."""
    live_payload = load_latest_daily_predictions() if path is None else None
    live_rows = []
    if live_payload:
        for prediction in live_payload.get("predictions", []):
            details = prediction.get("context_details", {})
            if not details:
                continue
            live_rows.append(
                {
                    "team_idx": int(prediction["home_team_idx"]),
                    "avg_players_out": float(details.get("home_players_out", 0)),
                    "avg_questionable": float(details.get("home_questionable", 0)),
                    "avg_estimated_value_missing": float(
                        details.get("home_estimated_value_missing", 0.0)
                    ),
                    "avg_minutes_missing": float(
                        details.get("home_estimated_value_missing", 0.0) * 24.0
                    ),
                    "data_available_rate": 1.0
                    if details.get("home_injury_data_available")
                    else 0.0,
                }
            )
            live_rows.append(
                {
                    "team_idx": int(prediction["away_team_idx"]),
                    "avg_players_out": float(details.get("away_players_out", 0)),
                    "avg_questionable": float(details.get("away_questionable", 0)),
                    "avg_estimated_value_missing": float(
                        details.get("away_estimated_value_missing", 0.0)
                    ),
                    "avg_minutes_missing": float(
                        details.get("away_estimated_value_missing", 0.0) * 24.0
                    ),
                    "data_available_rate": 1.0
                    if details.get("away_injury_data_available")
                    else 0.0,
                }
            )
    if live_rows:
        summary = (
            pd.DataFrame(live_rows)
            .groupby("team_idx", as_index=False)
            .mean(numeric_only=True)
        )
        summary["team"] = summary["team_idx"].map(team_abbr)
        return summary.sort_values("avg_estimated_value_missing", ascending=False).reset_index(
            drop=True
        )

    injuries = _join_feature_dates(load_injury_features(path))
    if injuries.empty:
        return pd.DataFrame()
    recent = injuries.sort_values("date").groupby("team_idx", as_index=False).tail(window)
    summary = recent.groupby("team_idx", as_index=False).agg(
        avg_players_out=("players_out_count", "mean"),
        avg_questionable=("players_questionable_count", "mean"),
        avg_estimated_value_missing=("estimated_value_missing", "mean"),
        avg_minutes_missing=("minutes_missing", "mean"),
        data_available_rate=("injury_data_available", "mean"),
    )
    summary["team"] = summary["team_idx"].map(team_abbr)
    return summary.sort_values("avg_estimated_value_missing", ascending=False).reset_index(
        drop=True
    )


def build_news_summary(window: int = 15, path: Path | None = None) -> pd.DataFrame:
    """Build a team-level news summary from the most recent games."""
    live_payload = load_latest_daily_predictions() if path is None else None
    live_rows = []
    if live_payload:
        for prediction in live_payload.get("predictions", []):
            details = prediction.get("context_details", {})
            if not details:
                continue
            live_rows.append(
                {
                    "team_idx": int(prediction["home_team_idx"]),
                    "avg_sentiment_24h": float(details.get("home_weighted_sentiment_72h", 0.0)),
                    "avg_sentiment_72h": float(details.get("home_weighted_sentiment_72h", 0.0)),
                    "avg_article_volume": float(details.get("home_article_volume_24h", 0)),
                    "avg_negative_ratio": 0.0,
                    "coverage_rate": 1.0 if details.get("home_news_available") else 0.0,
                }
            )
            live_rows.append(
                {
                    "team_idx": int(prediction["away_team_idx"]),
                    "avg_sentiment_24h": float(details.get("away_weighted_sentiment_72h", 0.0)),
                    "avg_sentiment_72h": float(details.get("away_weighted_sentiment_72h", 0.0)),
                    "avg_article_volume": float(details.get("away_article_volume_24h", 0)),
                    "avg_negative_ratio": 0.0,
                    "coverage_rate": 1.0 if details.get("away_news_available") else 0.0,
                }
            )
    if live_rows:
        summary = (
            pd.DataFrame(live_rows)
            .groupby("team_idx", as_index=False)
            .mean(numeric_only=True)
        )
        summary["team"] = summary["team_idx"].map(team_abbr)
        return summary.sort_values("avg_sentiment_72h", ascending=False).reset_index(drop=True)

    news = _join_feature_dates(load_news_features(path))
    if news.empty:
        return pd.DataFrame()
    recent = news.sort_values("date").groupby("team_idx", as_index=False).tail(window)
    summary = recent.groupby("team_idx", as_index=False).agg(
        avg_sentiment_24h=("weighted_sentiment_24h", "mean"),
        avg_sentiment_72h=("weighted_sentiment_72h", "mean"),
        avg_article_volume=("article_volume_24h", "mean"),
        avg_negative_ratio=("negative_ratio_72h", "mean"),
        coverage_rate=("news_available", "mean"),
    )
    summary["team"] = summary["team_idx"].map(team_abbr)
    return summary.sort_values("avg_sentiment_72h", ascending=False).reset_index(drop=True)
