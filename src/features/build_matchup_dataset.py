"""Build the final matchup dataset: one row per game with home/away features + diff.

Merges rolling features + schedule features for both home and away teams,
computes difference features, and attaches the target label.

Usage:
    python -m src.features.build_matchup_dataset
"""

import logging

import pandas as pd

from src.features.lineup_features import LINEUP_FEATURE_COLS, build_lineup_features
from src.features.player_value_features import build_player_value_features
from src.features.projected_availability import build_projected_availability
from src.features.rolling_features import compute_rolling_features
from src.features.schedule_features import compute_schedule_features
from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Key features to compute home-minus-away differences for
DIFF_FEATURES = [
    "season_win_pct", "last_10_win_pct",
    "season_point_diff", "last_10_point_diff",
    "season_net_rating", "last_10_net_rating",
    "season_off_rating", "season_def_rating",
    "last_10_off_rating", "last_10_def_rating",
    "season_pace", "last_10_pace",
    "season_ts_pct", "season_efg_pct",
    "season_turnover_pct", "season_assist_pct",
]

PROJECTED_VALUE_SUMMARY_COLS = [
    "projected_player_value_available",
    "projected_player_value_missing",
    "projected_top5_value_available",
    "projected_top5_value_missing",
    "projected_top8_value_available",
    "projected_top8_value_missing",
    "projected_top8_availability_mean",
    "projected_top8_confidence_mean",
]

ENRICHED_DIFF_FEATURES = [
    *LINEUP_FEATURE_COLS,
    *PROJECTED_VALUE_SUMMARY_COLS,
]


def build_matchup_dataset(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build the matchup dataset by merging home/away rolling + schedule features.

    Args:
        games: Games table (one row per game).
        team_logs: Team game logs (one row per team per game).

    Returns:
        Matchup dataset with one row per game, home/away/diff features, and target.
    """
    # Compute rolling and schedule features
    logger.info("Computing rolling features...")
    rolling = compute_rolling_features(team_logs)

    logger.info("Computing schedule features...")
    schedule = compute_schedule_features(team_logs)

    # Merge rolling + schedule on (game_id, team_idx)
    team_features = rolling.merge(schedule, on=["game_id", "team_idx"], how="left")

    # Identify feature columns (exclude metadata)
    meta_cols = {"game_id", "team_idx", "date", "season"}
    feature_cols = [c for c in team_features.columns if c not in meta_cols]

    # Split into home and away features
    home_features = team_features.merge(
        games[["game_id", "home_team_idx"]],
        left_on=["game_id", "team_idx"],
        right_on=["game_id", "home_team_idx"],
    ).drop(columns=["home_team_idx"])

    away_features = team_features.merge(
        games[["game_id", "away_team_idx"]],
        left_on=["game_id", "team_idx"],
        right_on=["game_id", "away_team_idx"],
    ).drop(columns=["away_team_idx"])

    # Rename columns with home_/away_ prefix
    home_renamed = home_features[["game_id"] + feature_cols].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    away_renamed = away_features[["game_id"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )

    # Start from games table
    matchup = games[["game_id", "date", "season", "home_team_idx", "away_team_idx"]].copy()

    # Merge home and away features
    matchup = matchup.merge(home_renamed, on="game_id", how="left")
    matchup = matchup.merge(away_renamed, on="game_id", how="left")

    # Compute diff features (home - away)
    for feat in DIFF_FEATURES:
        home_col = f"home_{feat}"
        away_col = f"away_{feat}"
        if home_col in matchup.columns and away_col in matchup.columns:
            matchup[f"diff_{feat}"] = matchup[home_col] - matchup[away_col]

    # Add target
    matchup["target_home_win"] = games.set_index("game_id").loc[
        matchup["game_id"], "home_win"
    ].values

    # Sort by date
    matchup = matchup.sort_values("date").reset_index(drop=True)

    logger.info(
        "Matchup dataset: %d rows, %d columns, home win rate: %.1f%%",
        len(matchup),
        len(matchup.columns),
        matchup["target_home_win"].mean() * 100,
    )

    return matchup


def summarize_projected_player_values(
    projected_availability: pd.DataFrame,
    *,
    starter_size: int = 5,
    rotation_size: int = 8,
) -> pd.DataFrame:
    """Aggregate player-level projected value rows into team-game summaries."""
    if projected_availability.empty:
        return pd.DataFrame(columns=["game_id", "team_idx", *PROJECTED_VALUE_SUMMARY_COLS])

    value_column = (
        "player_value_score"
        if "player_value_score" in projected_availability.columns
        else "role_score"
    )
    rows: list[dict[str, object]] = []

    for (game_id, team_idx), group in projected_availability.groupby(
        ["game_id", "team_idx"],
        sort=False,
    ):
        ordered = group.sort_values(
            [value_column, "role_score", "player_id"],
            ascending=[False, False, True],
        ).copy()
        ordered["effective_value"] = ordered[value_column] * ordered["availability_score"]
        ordered["missing_value"] = ordered[value_column] * (1.0 - ordered["availability_score"])

        starters = ordered.head(starter_size)
        rotation = ordered.head(rotation_size)
        rows.append(
            {
                "game_id": game_id,
                "team_idx": int(team_idx),
                "projected_player_value_available": float(ordered["effective_value"].sum()),
                "projected_player_value_missing": float(ordered["missing_value"].sum()),
                "projected_top5_value_available": float(starters["effective_value"].sum()),
                "projected_top5_value_missing": float(starters["missing_value"].sum()),
                "projected_top8_value_available": float(rotation["effective_value"].sum()),
                "projected_top8_value_missing": float(rotation["missing_value"].sum()),
                "projected_top8_availability_mean": float(rotation["availability_score"].mean()),
                "projected_top8_confidence_mean": float(rotation["projection_confidence"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["game_id", "team_idx"]).reset_index(drop=True)


def build_enriched_matchup_dataset(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
    player_logs: pd.DataFrame,
    *,
    player_value_features: pd.DataFrame | None = None,
    projected_availability: pd.DataFrame | None = None,
    lineup_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a parallel matchup dataset with player-/lineup-aware features added."""
    matchup = build_matchup_dataset(games, team_logs)

    value_features = player_value_features
    if value_features is None:
        value_features = build_player_value_features(games, player_logs)

    projected = projected_availability
    if projected is None:
        projected, _ = build_projected_availability(
            games,
            player_logs,
            player_value_features=value_features,
        )

    lineup_rows = lineup_features_df
    if lineup_rows is None:
        lineup_rows = build_lineup_features(games, player_logs, projected)

    projected_summary = summarize_projected_player_values(projected)
    team_enrichment = lineup_rows.merge(
        projected_summary,
        on=["game_id", "team_idx"],
        how="outer",
    )
    if team_enrichment.empty:
        return matchup

    enrichment_cols = [col for col in team_enrichment.columns if col not in {"game_id", "team_idx"}]
    home_enrichment = (
        team_enrichment.merge(
            games[["game_id", "home_team_idx"]],
            left_on=["game_id", "team_idx"],
            right_on=["game_id", "home_team_idx"],
            how="inner",
        )
        .drop(columns=["home_team_idx", "team_idx"])
        .rename(columns={col: f"home_{col}" for col in enrichment_cols})
    )
    away_enrichment = (
        team_enrichment.merge(
            games[["game_id", "away_team_idx"]],
            left_on=["game_id", "team_idx"],
            right_on=["game_id", "away_team_idx"],
            how="inner",
        )
        .drop(columns=["away_team_idx", "team_idx"])
        .rename(columns={col: f"away_{col}" for col in enrichment_cols})
    )

    enriched = matchup.merge(home_enrichment, on="game_id", how="left")
    enriched = enriched.merge(away_enrichment, on="game_id", how="left")

    for feature_name in ENRICHED_DIFF_FEATURES:
        home_col = f"home_{feature_name}"
        away_col = f"away_{feature_name}"
        if home_col in enriched.columns and away_col in enriched.columns:
            enriched[f"diff_{feature_name}"] = enriched[home_col] - enriched[away_col]

    return enriched.sort_values("date").reset_index(drop=True)


def main():
    """Build and save the matchup dataset."""
    setup_logging()

    games_path = PROCESSED_DIR / "games.parquet"
    logs_path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
    player_logs_path = PROCESSED_DIR / "player_game_logs" / "player_game_logs.parquet"

    logger.info("Loading data...")
    games = pd.read_parquet(games_path)
    team_logs = pd.read_parquet(logs_path)

    logger.info("Building matchup dataset...")
    matchup = build_matchup_dataset(games, team_logs)

    out_path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matchup.to_parquet(out_path, index=False)
    logger.info("Saved to %s", out_path)

    if not player_logs_path.exists():
        logger.info("Skipping enriched matchup build because %s is missing", player_logs_path)
        return

    player_logs = pd.read_parquet(player_logs_path)
    player_value_features = build_player_value_features(games, player_logs)
    value_out_dir = PROCESSED_DIR / "player_value_features"
    value_out_dir.mkdir(parents=True, exist_ok=True)
    player_value_features.to_parquet(
        value_out_dir / "player_value_features.parquet",
        index=False,
    )

    projected_availability, unresolved = build_projected_availability(
        games,
        player_logs,
        player_value_features=player_value_features,
    )
    projected_out_dir = PROCESSED_DIR / "projected_availability"
    projected_out_dir.mkdir(parents=True, exist_ok=True)
    projected_availability.to_parquet(
        projected_out_dir / "projected_availability.parquet",
        index=False,
    )
    unresolved.to_parquet(
        projected_out_dir / "unresolved_injury_entities.parquet",
        index=False,
    )

    lineup_features_df = build_lineup_features(games, player_logs, projected_availability)
    lineup_out_dir = PROCESSED_DIR / "lineup_features"
    lineup_out_dir.mkdir(parents=True, exist_ok=True)
    lineup_features_df.to_parquet(lineup_out_dir / "lineup_features.parquet", index=False)

    logger.info("Building enriched matchup dataset...")
    enriched = build_enriched_matchup_dataset(
        games,
        team_logs,
        player_logs,
        player_value_features=player_value_features,
        projected_availability=projected_availability,
        lineup_features_df=lineup_features_df,
    )
    enriched_out_path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset_enriched.parquet"
    enriched.to_parquet(enriched_out_path, index=False)
    logger.info("Saved enriched matchup dataset to %s", enriched_out_path)


if __name__ == "__main__":
    main()
