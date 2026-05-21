"""Schema management for the prospective forecast context store."""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.utils.paths import CONTEXT_STORE_DB, CONTEXT_STORE_PARQUET_DIR

ColumnSpec = tuple[str, str]

TABLE_SCHEMAS: dict[str, list[ColumnSpec]] = {
    "forecast_runs": [
        ("run_id", "VARCHAR"),
        ("as_of_utc", "VARCHAR"),
        ("target_date", "VARCHAR"),
        ("window_start", "VARCHAR"),
        ("window_end", "VARCHAR"),
        ("timezone", "VARCHAR"),
        ("git_commit", "VARCHAR"),
        ("model_version", "VARCHAR"),
        ("production_model", "VARCHAR"),
        ("shadow_enabled", "BOOLEAN"),
        ("status", "VARCHAR"),
        ("games_count", "INTEGER"),
        ("schedule_status", "VARCHAR"),
        ("injury_status", "VARCHAR"),
        ("news_status", "VARCHAR"),
        ("created_by", "VARCHAR"),
    ],
    "game_snapshots": [
        ("run_id", "VARCHAR"),
        ("game_id", "VARCHAR"),
        ("game_date", "VARCHAR"),
        ("game_time_utc", "VARCHAR"),
        ("season", "VARCHAR"),
        ("slate_type", "VARCHAR"),
        ("home_team_idx", "INTEGER"),
        ("away_team_idx", "INTEGER"),
        ("home_team_abbr", "VARCHAR"),
        ("away_team_abbr", "VARCHAR"),
        ("game_label", "VARCHAR"),
        ("game_sub_label", "VARCHAR"),
        ("series_text", "VARCHAR"),
        ("game_status_text", "VARCHAR"),
        ("schedule_source", "VARCHAR"),
        ("if_necessary", "BOOLEAN"),
        ("as_of_utc", "VARCHAR"),
    ],
    "model_features": [
        ("run_id", "VARCHAR"),
        ("game_id", "VARCHAR"),
        ("entity_scope", "VARCHAR"),
        ("feature_group", "VARCHAR"),
        ("feature_name", "VARCHAR"),
        ("feature_value", "DOUBLE"),
        ("feature_dtype", "VARCHAR"),
        ("source", "VARCHAR"),
        ("as_of_utc", "VARCHAR"),
    ],
    "injury_context": [
        ("run_id", "VARCHAR"),
        ("game_id", "VARCHAR"),
        ("team_idx", "INTEGER"),
        ("player_name_or_id", "VARCHAR"),
        ("status", "VARCHAR"),
        ("reason", "VARCHAR"),
        ("report_generated_at", "VARCHAR"),
        ("report_source_url", "VARCHAR"),
        ("injury_data_available", "BOOLEAN"),
        ("estimated_value_missing", "DOUBLE"),
        ("players_out_count", "INTEGER"),
        ("players_questionable_count", "INTEGER"),
        ("source_mode", "VARCHAR"),
        ("as_of_utc", "VARCHAR"),
    ],
    "news_articles": [
        ("run_id", "VARCHAR"),
        ("game_id", "VARCHAR"),
        ("team_idx", "INTEGER"),
        ("article_id", "VARCHAR"),
        ("published_at", "VARCHAR"),
        ("collected_at", "VARCHAR"),
        ("source", "VARCHAR"),
        ("title", "VARCHAR"),
        ("summary_hash", "VARCHAR"),
        ("link", "VARCHAR"),
        ("included_in_model", "BOOLEAN"),
        ("excluded_reason", "VARCHAR"),
        ("article_relevance", "DOUBLE"),
        ("article_relevance_reason", "VARCHAR"),
        ("as_of_utc", "VARCHAR"),
    ],
    "news_scores": [
        ("run_id", "VARCHAR"),
        ("game_id", "VARCHAR"),
        ("team_idx", "INTEGER"),
        ("article_id", "VARCHAR"),
        ("overall_sentiment", "DOUBLE"),
        ("injury_concern", "DOUBLE"),
        ("pressure", "DOUBLE"),
        ("team_cohesion", "DOUBLE"),
        ("motivation", "DOUBLE"),
        ("llm_confidence", "DOUBLE"),
        ("scorer_version", "VARCHAR"),
        ("as_of_utc", "VARCHAR"),
    ],
    "prediction_outputs": [
        ("run_id", "VARCHAR"),
        ("game_id", "VARCHAR"),
        ("home_win_probability", "DOUBLE"),
        ("away_win_probability", "DOUBLE"),
        ("predicted_winner", "VARCHAR"),
        ("confidence_bucket", "VARCHAR"),
        ("elo_probability", "DOUBLE"),
        ("tabular_probability", "DOUBLE"),
        ("sequence_probability", "DOUBLE"),
        ("ensemble_v1_probability", "DOUBLE"),
        ("nextgen_probability", "DOUBLE"),
        ("final_probability", "DOUBLE"),
        ("top_model_factors_json", "VARCHAR"),
        ("context_details_json", "VARCHAR"),
        ("as_of_utc", "VARCHAR"),
    ],
    "outcomes": [
        ("game_id", "VARCHAR"),
        ("final_home_score", "INTEGER"),
        ("final_away_score", "INTEGER"),
        ("home_win", "BOOLEAN"),
        ("completed_at", "VARCHAR"),
        ("outcome_source", "VARCHAR"),
        ("hydrated_at", "VARCHAR"),
    ],
}

PREGAME_TABLES = tuple(table for table in TABLE_SCHEMAS if table != "outcomes")
OUTCOME_COLUMNS = {"final_home_score", "final_away_score", "home_win", "completed_at"}


def table_columns(table_name: str) -> list[str]:
    """Return ordered column names for a context-store table."""
    return [name for name, _ in TABLE_SCHEMAS[table_name]]


def validate_pregame_schema() -> None:
    """Guard against accidentally leaking outcome fields into pregame tables."""
    for table_name in PREGAME_TABLES:
        overlap = OUTCOME_COLUMNS.intersection(table_columns(table_name))
        if overlap:
            msg = f"Pregame table {table_name} contains outcome columns: {sorted(overlap)}"
            raise ValueError(msg)


def _create_table_sql(table_name: str, columns: list[ColumnSpec]) -> str:
    column_sql = ",\n    ".join(f"{name} {dtype}" for name, dtype in columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {column_sql}\n)"


def initialize_context_store(
    db_path: Path | None = None,
    parquet_root: Path | None = None,
) -> tuple[Path, Path]:
    """Create the DuckDB schema and matching Parquet table directories."""
    validate_pregame_schema()

    resolved_db = Path(db_path or CONTEXT_STORE_DB)
    resolved_parquet = Path(parquet_root or CONTEXT_STORE_PARQUET_DIR)
    resolved_db.parent.mkdir(parents=True, exist_ok=True)
    resolved_parquet.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(resolved_db)) as conn:
        for table_name, columns in TABLE_SCHEMAS.items():
            conn.execute(_create_table_sql(table_name, columns))
            (resolved_parquet / table_name).mkdir(parents=True, exist_ok=True)

    return resolved_db, resolved_parquet
