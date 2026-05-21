"""Append-only writers for the prospective forecast context store."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.context_store.schema import TABLE_SCHEMAS, initialize_context_store, table_columns
from src.utils.paths import (
    CONTEXT_STORE_DB,
    CONTEXT_STORE_PARQUET_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)

SCORER_VERSION = "deterministic_keyword_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text else None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "live"}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _json_dumps(value: object) -> str:
    return json.dumps(value or [], sort_keys=True, default=str)


def _summary_hash(value: object) -> str | None:
    text = _safe_text(value)
    if not text:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    github_sha = os.getenv("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _default_run_id(payload: dict, manifest: dict, as_of_utc: str) -> str:
    target_date = str(manifest.get("target_date") or payload.get("date") or "unknown")
    seed = "|".join(
        [
            target_date,
            as_of_utc,
            str(payload.get("model_version")),
            str(len(payload.get("predictions", []))),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    compact_time = (
        as_of_utc.replace("-", "")
        .replace(":", "")
        .replace("T", "_")
        .replace("Z", "")
    )
    return f"forecast_{target_date}_{compact_time}_{digest}"


def _as_of_utc(payload: dict, manifest: dict, override: str | None = None) -> str:
    return (
        override
        or _safe_text(payload.get("generated_at"))
        or _safe_text(manifest.get("attempted_at"))
        or _utc_now_iso()
    )


def _feature_group(feature_name: str, source: str) -> str:
    name = feature_name.lower()
    if source == "component_outputs":
        return "component_output"
    if "injury" in name or "player" in name or "questionable" in name:
        return "injury"
    if "news" in name or "article" in name or "sentiment" in name:
        return "news"
    if "probability" in name or name.endswith("_prob"):
        return "probability"
    return "context_detail"


def _entity_scope(feature_name: str) -> str:
    if feature_name.startswith("home_"):
        return "home_team"
    if feature_name.startswith("away_"):
        return "away_team"
    return "game"


def _numeric_feature_rows(
    *,
    run_id: str,
    game_id: str,
    mapping: dict[str, object],
    source: str,
    as_of_utc: str,
) -> list[dict[str, object]]:
    rows = []
    for feature_name, value in mapping.items():
        numeric_value = _safe_float(value)
        if numeric_value is None:
            bool_value = _safe_bool(value)
            if bool_value is None or not isinstance(value, bool):
                continue
            numeric_value = 1.0 if bool_value else 0.0
            feature_dtype = "bool"
        else:
            feature_dtype = "numeric"
        rows.append(
            {
                "run_id": run_id,
                "game_id": game_id,
                "entity_scope": _entity_scope(feature_name),
                "feature_group": _feature_group(feature_name, source),
                "feature_name": feature_name,
                "feature_value": numeric_value,
                "feature_dtype": feature_dtype,
                "source": source,
                "as_of_utc": as_of_utc,
            }
        )
    return rows


def _forecast_run_frame(payload: dict, manifest: dict, run_id: str, as_of_utc: str) -> pd.DataFrame:
    observability = manifest.get("publish_observability", {}) or {}
    api_status = observability.get("api_status", {}) or {}
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "as_of_utc": as_of_utc,
                "target_date": manifest.get("target_date") or payload.get("date"),
                "window_start": manifest.get("window_start") or payload.get("window_start"),
                "window_end": manifest.get("window_end") or payload.get("window_end"),
                "timezone": observability.get("timezone"),
                "git_commit": _git_commit(),
                "model_version": manifest.get("model_version") or payload.get("model_version"),
                "production_model": os.getenv("NBA_PREDICT_PRODUCTION_MODEL", "nextgen"),
                "shadow_enabled": bool(
                    manifest.get("shadow_model_version") or payload.get("shadow_model_version")
                ),
                "status": manifest.get("status"),
                "games_count": int(manifest.get("games_count", 0) or 0),
                "schedule_status": api_status.get("schedule"),
                "injury_status": api_status.get("injury"),
                "news_status": api_status.get("news"),
                "created_by": os.getenv("GITHUB_ACTOR", "local"),
            }
        ]
    )


def _game_snapshot_rows(payload: dict, run_id: str, as_of_utc: str) -> list[dict[str, object]]:
    rows = []
    for prediction in payload.get("predictions", []):
        rows.append(
            {
                "run_id": run_id,
                "game_id": _safe_text(prediction.get("game_id")),
                "game_date": _safe_text(prediction.get("game_date") or payload.get("date")),
                "game_time_utc": _safe_text(prediction.get("game_time_utc")),
                "season": _safe_text(prediction.get("season")),
                "slate_type": _safe_text(payload.get("slate_type")),
                "home_team_idx": _safe_int(prediction.get("home_team_idx")),
                "away_team_idx": _safe_int(prediction.get("away_team_idx")),
                "home_team_abbr": _safe_text(prediction.get("home_team_abbr")),
                "away_team_abbr": _safe_text(prediction.get("away_team_abbr")),
                "game_label": _safe_text(prediction.get("game_label")),
                "game_sub_label": _safe_text(prediction.get("game_sub_label")),
                "series_text": _safe_text(prediction.get("series_text")),
                "game_status_text": _safe_text(prediction.get("game_status_text")),
                "schedule_source": _safe_text(prediction.get("schedule_source")),
                "if_necessary": _safe_bool(prediction.get("if_necessary")),
                "as_of_utc": as_of_utc,
            }
        )
    return rows


def _prediction_output_rows(payload: dict, run_id: str, as_of_utc: str) -> list[dict[str, object]]:
    rows = []
    for prediction in payload.get("predictions", []):
        component_outputs = prediction.get("component_outputs", {}) or {}
        nextgen_probability = (
            component_outputs.get("nextgen_shadow_probability")
            or component_outputs.get("nextgen_probability")
        )
        ensemble_v1_probability = (
            component_outputs.get("ensemble_v1_probability")
            or component_outputs.get("ensemble_probability")
        )
        rows.append(
            {
                "run_id": run_id,
                "game_id": _safe_text(prediction.get("game_id")),
                "home_win_probability": _safe_float(prediction.get("home_win_probability")),
                "away_win_probability": _safe_float(prediction.get("away_win_probability")),
                "predicted_winner": _safe_text(prediction.get("predicted_winner")),
                "confidence_bucket": _safe_text(prediction.get("confidence_bucket")),
                "elo_probability": _safe_float(component_outputs.get("elo_probability")),
                "tabular_probability": _safe_float(component_outputs.get("tabular_probability")),
                "sequence_probability": _safe_float(component_outputs.get("sequence_probability")),
                "ensemble_v1_probability": _safe_float(ensemble_v1_probability),
                "nextgen_probability": _safe_float(nextgen_probability),
                "final_probability": _safe_float(
                    component_outputs.get("final_probability")
                    or prediction.get("home_win_probability")
                ),
                "top_model_factors_json": _json_dumps(prediction.get("top_model_factors")),
                "context_details_json": _json_dumps(prediction.get("context_details")),
                "as_of_utc": as_of_utc,
            }
        )
    return rows


def _model_feature_rows(payload: dict, run_id: str, as_of_utc: str) -> list[dict[str, object]]:
    rows = []
    for prediction in payload.get("predictions", []):
        game_id = _safe_text(prediction.get("game_id"))
        if not game_id:
            continue
        rows.extend(
            _numeric_feature_rows(
                run_id=run_id,
                game_id=game_id,
                mapping={
                    "home_win_probability": prediction.get("home_win_probability"),
                    "away_win_probability": prediction.get("away_win_probability"),
                },
                source="prediction_payload",
                as_of_utc=as_of_utc,
            )
        )
        rows.extend(
            _numeric_feature_rows(
                run_id=run_id,
                game_id=game_id,
                mapping=prediction.get("component_outputs", {}) or {},
                source="component_outputs",
                as_of_utc=as_of_utc,
            )
        )
        rows.extend(
            _numeric_feature_rows(
                run_id=run_id,
                game_id=game_id,
                mapping=prediction.get("context_details", {}) or {},
                source="context_details",
                as_of_utc=as_of_utc,
            )
        )
    return rows


def _injury_context_rows(payload: dict, run_id: str, as_of_utc: str) -> list[dict[str, object]]:
    rows = []
    for prediction in payload.get("predictions", []):
        context = prediction.get("context_details", {}) or {}
        for side in ("home", "away"):
            team_idx = _safe_int(prediction.get(f"{side}_team_idx"))
            if team_idx is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "game_id": _safe_text(prediction.get("game_id")),
                    "team_idx": team_idx,
                    "player_name_or_id": None,
                    "status": _safe_text(context.get(f"{side}_injury_report_status")),
                    "reason": None,
                    "report_generated_at": _safe_text(context.get("injury_report_generated_at")),
                    "report_source_url": _safe_text(context.get("injury_report_source_url")),
                    "injury_data_available": _safe_bool(
                        context.get(f"{side}_injury_data_available")
                    ),
                    "estimated_value_missing": _safe_float(
                        context.get(f"{side}_estimated_value_missing")
                    ),
                    "players_out_count": _safe_int(context.get(f"{side}_players_out")),
                    "players_questionable_count": _safe_int(context.get(f"{side}_questionable")),
                    "source_mode": _safe_text(context.get("injury_mode")),
                    "as_of_utc": as_of_utc,
                }
            )
    return rows


def _prediction_team_games(payload: dict) -> list[dict[str, object]]:
    rows = []
    for prediction in payload.get("predictions", []):
        game_time = pd.to_datetime(
            prediction.get("game_time_utc") or prediction.get("game_date"),
            utc=True,
            errors="coerce",
        )
        for side in ("home", "away"):
            team_idx = _safe_int(prediction.get(f"{side}_team_idx"))
            if team_idx is None:
                continue
            rows.append(
                {
                    "game_id": _safe_text(prediction.get("game_id")),
                    "team_idx": team_idx,
                    "game_time": game_time,
                    "news_collected_at": _safe_text(
                        (prediction.get("context_details", {}) or {}).get("news_collected_at")
                    ),
                }
            )
    return rows


def _load_cached_frames(directory: Path, pattern: str) -> pd.DataFrame:
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.glob(pattern)):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _join_news_rows_to_games(
    news: pd.DataFrame,
    payload: dict,
    *,
    require_collected_match: bool,
) -> pd.DataFrame:
    if news.empty:
        return news
    team_games = _prediction_team_games(payload)
    if not team_games:
        return pd.DataFrame()

    news = news.copy()
    news["published_at_ts"] = pd.to_datetime(news.get("published_at"), utc=True, errors="coerce")
    if "collected_at" in news.columns:
        news["collected_at"] = news["collected_at"].astype(str)

    rows = []
    for team_game in team_games:
        game_time = team_game["game_time"]
        team_idx = team_game["team_idx"]
        candidates = news[news["team_idx"].astype("Int64") == int(team_idx)].copy()
        if pd.notna(game_time):
            candidates = candidates[
                candidates["published_at_ts"].isna()
                | (candidates["published_at_ts"] < game_time)
            ]
        if (
            require_collected_match
            and team_game.get("news_collected_at")
            and "collected_at" in candidates
        ):
            candidates = candidates[candidates["collected_at"] == team_game["news_collected_at"]]
        if candidates.empty:
            continue
        candidates["game_id"] = team_game["game_id"]
        rows.append(candidates)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop(columns=["published_at_ts"], errors="ignore")


def _news_article_rows(
    payload: dict,
    run_id: str,
    as_of_utc: str,
    news_dir: Path,
) -> list[dict[str, object]]:
    articles = _load_cached_frames(news_dir, "team_news_articles_*.parquet")
    if articles.empty:
        articles = _load_cached_frames(news_dir, "team_news_scores_*.parquet")
        if not articles.empty:
            articles["included_in_model"] = True
            articles["excluded_reason"] = ""

    joined = _join_news_rows_to_games(articles, payload, require_collected_match=True)
    if joined.empty and not articles.empty:
        joined = _join_news_rows_to_games(articles, payload, require_collected_match=False)

    rows = []
    for _, row in joined.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "game_id": _safe_text(row.get("game_id")),
                "team_idx": _safe_int(row.get("team_idx")),
                "article_id": _safe_text(row.get("article_id")),
                "published_at": _safe_text(row.get("published_at")),
                "collected_at": _safe_text(row.get("collected_at")),
                "source": _safe_text(row.get("source")),
                "title": _safe_text(row.get("title")),
                "summary_hash": _summary_hash(row.get("summary")),
                "link": _safe_text(row.get("link")),
                "included_in_model": bool(_safe_bool(row.get("included_in_model"))),
                "excluded_reason": _safe_text(row.get("excluded_reason")),
                "article_relevance": _safe_float(row.get("article_relevance")),
                "article_relevance_reason": _safe_text(row.get("article_relevance_reason")),
                "as_of_utc": as_of_utc,
            }
        )
    return rows


def _news_score_rows(
    payload: dict,
    run_id: str,
    as_of_utc: str,
    news_dir: Path,
) -> list[dict[str, object]]:
    scores = _load_cached_frames(news_dir, "team_news_scores_*.parquet")
    joined = _join_news_rows_to_games(scores, payload, require_collected_match=True)
    if joined.empty and not scores.empty:
        joined = _join_news_rows_to_games(scores, payload, require_collected_match=False)

    rows = []
    for _, row in joined.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "game_id": _safe_text(row.get("game_id")),
                "team_idx": _safe_int(row.get("team_idx")),
                "article_id": _safe_text(row.get("article_id")),
                "overall_sentiment": _safe_float(row.get("overall_sentiment")),
                "injury_concern": _safe_float(row.get("injury_concern")),
                "pressure": _safe_float(row.get("pressure")),
                "team_cohesion": _safe_float(row.get("team_cohesion")),
                "motivation": _safe_float(row.get("motivation")),
                "llm_confidence": _safe_float(row.get("llm_confidence")),
                "scorer_version": SCORER_VERSION,
                "as_of_utc": as_of_utc,
            }
        )
    return rows


def build_context_frames(
    payload: dict,
    manifest: dict,
    *,
    run_id: str | None = None,
    as_of_utc: str | None = None,
    news_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build context-store table frames from a published forecast payload."""
    resolved_as_of = _as_of_utc(payload, manifest, as_of_utc)
    resolved_run_id = run_id or _default_run_id(payload, manifest, resolved_as_of)
    resolved_news_dir = Path(news_dir or (RAW_DIR / "news"))

    frames = {
        "forecast_runs": _forecast_run_frame(payload, manifest, resolved_run_id, resolved_as_of),
        "game_snapshots": pd.DataFrame(
            _game_snapshot_rows(payload, resolved_run_id, resolved_as_of)
        ),
        "model_features": pd.DataFrame(
            _model_feature_rows(payload, resolved_run_id, resolved_as_of)
        ),
        "injury_context": pd.DataFrame(
            _injury_context_rows(payload, resolved_run_id, resolved_as_of)
        ),
        "news_articles": pd.DataFrame(
            _news_article_rows(payload, resolved_run_id, resolved_as_of, resolved_news_dir)
        ),
        "news_scores": pd.DataFrame(
            _news_score_rows(payload, resolved_run_id, resolved_as_of, resolved_news_dir)
        ),
        "prediction_outputs": pd.DataFrame(
            _prediction_output_rows(payload, resolved_run_id, resolved_as_of)
        ),
    }

    return {
        table_name: frame.reindex(columns=table_columns(table_name))
        for table_name, frame in frames.items()
    }


def _run_exists(conn: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    result = conn.execute(
        "SELECT COUNT(*) FROM forecast_runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    return bool(result and result[0] > 0)


def _append_frame(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    ordered = frame.reindex(columns=table_columns(table_name))
    conn.register("_context_frame", ordered)
    try:
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM _context_frame")
    finally:
        conn.unregister("_context_frame")


def _write_parquet_snapshots(
    frames: dict[str, pd.DataFrame],
    parquet_root: Path,
    run_id: str,
) -> None:
    safe_run_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run_id)
    for table_name, frame in frames.items():
        if frame.empty:
            continue
        table_dir = parquet_root / table_name
        table_dir.mkdir(parents=True, exist_ok=True)
        out_path = table_dir / f"{safe_run_id}.parquet"
        if out_path.exists():
            msg = f"Context-store Parquet snapshot already exists: {out_path}"
            raise ValueError(msg)
        frame.to_parquet(out_path, index=False)


def save_forecast_context(
    payload: dict,
    manifest: dict,
    *,
    db_path: Path | None = None,
    parquet_root: Path | None = None,
    run_id: str | None = None,
    as_of_utc: str | None = None,
    news_dir: Path | None = None,
) -> str:
    """Append a successful published forecast to the context store."""
    resolved_db, resolved_parquet = initialize_context_store(
        db_path=db_path or CONTEXT_STORE_DB,
        parquet_root=parquet_root or CONTEXT_STORE_PARQUET_DIR,
    )
    resolved_as_of = _as_of_utc(payload, manifest, as_of_utc)
    resolved_run_id = run_id or _default_run_id(payload, manifest, resolved_as_of)
    frames = build_context_frames(
        payload,
        manifest,
        run_id=resolved_run_id,
        as_of_utc=resolved_as_of,
        news_dir=news_dir,
    )

    with duckdb.connect(str(resolved_db)) as conn:
        if _run_exists(conn, resolved_run_id):
            msg = f"Context-store run already exists: {resolved_run_id}"
            raise ValueError(msg)
        conn.execute("BEGIN TRANSACTION")
        try:
            for table_name in TABLE_SCHEMAS:
                if table_name == "outcomes":
                    continue
                _append_frame(conn, table_name, frames.get(table_name, pd.DataFrame()))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    _write_parquet_snapshots(frames, resolved_parquet, resolved_run_id)
    return resolved_run_id
