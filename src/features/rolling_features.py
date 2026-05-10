"""Compute leakage-safe rolling features for each team.

CRITICAL: All features for game N use only data from games 1..N-1.
This is enforced via .shift(1) on all rolling computations.

Usage:
    python -m src.features.rolling_features
"""

import logging

import pandas as pd
import yaml

from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


def _load_feature_config() -> dict:
    """Load feature engineering configuration."""
    with open(CONFIGS_DIR / "feature_config.yaml") as f:
        return yaml.safe_load(f)


def compute_rolling_features(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features per team, shifted by 1 game to prevent leakage.

    For each team-game row, rolling features are computed from all PRIOR games
    of that team in the current season.

    Args:
        team_logs: Team game logs DataFrame (from Phase 1).

    Returns:
        DataFrame with game_id, team_idx, and all rolling feature columns.
    """
    team_logs = team_logs.sort_values(["team_idx", "season", "date"]).copy()

    # Stats to compute rolling averages for
    stat_cols = [
        "won", "point_diff", "net_rating", "off_rating", "def_rating",
        "pace", "ts_pct", "efg_pct", "turnover_pct",
        "free_throw_rate", "assist_pct",
        "off_rebound_pct", "def_rebound_pct",
    ]

    config = _load_feature_config()
    windows = config["rolling_windows"]  # [5, 10]

    all_features = []

    for (team_idx, season), group in team_logs.groupby(["team_idx", "season"]):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        features = pd.DataFrame({
            "game_id": group["game_id"].values,
            "team_idx": group["team_idx"].values,
            "date": group["date"].values,
            "season": group["season"].values,
        })

        for col in stat_cols:
            vals = group[col].astype(float)

            # Season expanding mean (all prior games) — shift(1) to exclude current
            features[f"season_{col}"] = vals.expanding().mean().shift(1)

            # Rolling windows
            for w in windows:
                features[f"last_{w}_{col}"] = (
                    vals.rolling(window=w, min_periods=1).mean().shift(1)
                )

        # Special: win_pct is just rolling mean of 'won'
        # Already computed as season_won, last_5_won, last_10_won
        # Rename for clarity
        features = features.rename(columns={
            "season_won": "season_win_pct",
            "last_5_won": "last_5_win_pct",
            "last_10_won": "last_10_win_pct",
        })

        # Games played in season (before current game)
        features["season_games_played"] = range(n)  # 0, 1, 2, ... (0 means first game)

        all_features.append(features)

    result = pd.concat(all_features, ignore_index=True)
    logger.info(
        "Computed rolling features: %d rows, %d columns",
        len(result), len(result.columns),
    )
    return result


def main():
    """Build rolling features and save."""
    setup_logging()

    logs_path = PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
    logger.info("Loading team game logs from %s", logs_path)
    team_logs = pd.read_parquet(logs_path)

    logger.info("Computing rolling features...")
    features = compute_rolling_features(team_logs)

    out_path = PROCESSED_DIR / "team_rolling_features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Saved rolling features to %s (%d rows, %d cols)",
                out_path, len(features), len(features.columns))


if __name__ == "__main__":
    main()
