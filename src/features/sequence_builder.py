"""Build padded game sequences for each team per game.

For each game, constructs two sequences (home + away):
  - Each sequence contains the team's last N games (default N=20)
  - Early-season games with < N history are left-padded with zeros
  - A binary mask indicates real vs padded positions
  - CRITICAL: Only games BEFORE the current game are included (no leakage)

Usage:
    python -m src.features.sequence_builder
"""

import logging

import numpy as np
import pandas as pd
import yaml

from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Features to include per game step in the sequence
SEQUENCE_FEATURES = [
    "won", "point_diff", "net_rating", "off_rating", "def_rating",
    "pace", "ts_pct", "efg_pct", "turnover_pct",
    "off_rebound_pct", "def_rebound_pct", "free_throw_rate",
    "assist_pct", "steal_pct", "block_pct", "is_home",
]


def _load_sequence_config() -> dict:
    """Load sequence model configuration."""
    with open(CONFIGS_DIR / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    return config["sequence"]


def build_team_sequences(
    team_logs: pd.DataFrame,
    games: pd.DataFrame,
    seq_len: int = 20,
    features: list[str] | None = None,
) -> dict:
    """Build padded game sequences for every game in the dataset.

    For each game, produces:
      - home_seq: [seq_len, n_features] — home team's last N games
      - away_seq: [seq_len, n_features] — away team's last N games
      - home_mask: [seq_len] — 1 for real games, 0 for padding
      - away_mask: [seq_len] — 1 for real games, 0 for padding

    Args:
        team_logs: Per-team game logs (from Phase 1).
        games: Games table (one row per game).
        seq_len: Number of previous games to include.
        features: Feature columns per game step.

    Returns:
        Dict with arrays: home_sequences, away_sequences,
        home_masks, away_masks, game_ids, targets.
    """
    if features is None:
        features = SEQUENCE_FEATURES

    n_features = len(features)
    team_logs = team_logs.sort_values(["team_idx", "date"]).copy()
    games = games.sort_values("date").copy()

    # Pre-compute team history as a dict of {team_idx: sorted DataFrame}
    team_history = {}
    for team_idx, group in team_logs.groupby("team_idx"):
        team_history[team_idx] = group.sort_values("date")[
            ["game_id", "date"] + features
        ].reset_index(drop=True)

    n_games = len(games)
    home_sequences = np.zeros((n_games, seq_len, n_features), dtype=np.float32)
    away_sequences = np.zeros((n_games, seq_len, n_features), dtype=np.float32)
    home_masks = np.zeros((n_games, seq_len), dtype=np.float32)
    away_masks = np.zeros((n_games, seq_len), dtype=np.float32)
    game_ids = []
    targets = np.zeros(n_games, dtype=np.float32)

    for i, (_, game) in enumerate(games.iterrows()):
        game_date = pd.Timestamp(game["date"])
        game_id = game["game_id"]
        home_idx = int(game["home_team_idx"])
        away_idx = int(game["away_team_idx"])

        game_ids.append(game_id)
        targets[i] = float(game["home_win"])

        # Build home sequence
        home_hist = team_history[home_idx]
        prior_home = home_hist[home_hist["date"] < game_date]
        if len(prior_home) > 0:
            # Take last seq_len games
            recent = prior_home.iloc[-seq_len:]
            n_real = len(recent)
            values = recent[features].values.astype(np.float32)
            # Left-pad: real games go at the end
            home_sequences[i, seq_len - n_real:, :] = values
            home_masks[i, seq_len - n_real:] = 1.0

        # Build away sequence
        away_hist = team_history[away_idx]
        prior_away = away_hist[away_hist["date"] < game_date]
        if len(prior_away) > 0:
            recent = prior_away.iloc[-seq_len:]
            n_real = len(recent)
            values = recent[features].values.astype(np.float32)
            away_sequences[i, seq_len - n_real:, :] = values
            away_masks[i, seq_len - n_real:] = 1.0

    # Replace NaN with 0 (shouldn't happen, but safety)
    home_sequences = np.nan_to_num(home_sequences, nan=0.0)
    away_sequences = np.nan_to_num(away_sequences, nan=0.0)

    logger.info(
        "Built sequences: %d games, seq_len=%d, n_features=%d",
        n_games, seq_len, n_features,
    )

    return {
        "home_sequences": home_sequences,
        "away_sequences": away_sequences,
        "home_masks": home_masks,
        "away_masks": away_masks,
        "game_ids": np.array(game_ids),
        "targets": targets,
    }


def build_team_sequences_for_games(
    team_logs: pd.DataFrame,
    target_games: pd.DataFrame,
    seq_len: int = 20,
    features: list[str] | None = None,
) -> dict:
    """Build padded sequences only for the requested target games.

    This is the live-inference sibling of :func:`build_team_sequences`. It uses
    the same leakage-safe rule (`team_log.date < game.date`) but avoids
    constructing arrays for every historical game when publishing a tiny slate.
    """
    if features is None:
        features = SEQUENCE_FEATURES

    n_features = len(features)
    team_logs = team_logs.sort_values(["team_idx", "date"]).copy()
    target_games = target_games.sort_values("date").copy()
    team_logs["date"] = pd.to_datetime(team_logs["date"])
    target_games["date"] = pd.to_datetime(target_games["date"])

    team_history = {}
    for team_idx, group in team_logs.groupby("team_idx"):
        team_history[int(team_idx)] = group.sort_values("date")[
            ["game_id", "date"] + features
        ].reset_index(drop=True)

    n_games = len(target_games)
    home_sequences = np.zeros((n_games, seq_len, n_features), dtype=np.float32)
    away_sequences = np.zeros((n_games, seq_len, n_features), dtype=np.float32)
    home_masks = np.zeros((n_games, seq_len), dtype=np.float32)
    away_masks = np.zeros((n_games, seq_len), dtype=np.float32)
    game_ids = []
    targets = np.zeros(n_games, dtype=np.float32)

    def _fill_sequence(
        history: pd.DataFrame | None,
        game_date: pd.Timestamp,
        sequence: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if history is None or history.empty:
            return

        prior = history[history["date"] < game_date]
        if prior.empty:
            return

        recent = prior.iloc[-seq_len:]
        n_real = len(recent)
        sequence[seq_len - n_real:, :] = recent[features].values.astype(np.float32)
        mask[seq_len - n_real:] = 1.0

    for i, (_, game) in enumerate(target_games.iterrows()):
        game_date = pd.Timestamp(game["date"])
        game_id = game["game_id"]
        home_idx = int(game["home_team_idx"])
        away_idx = int(game["away_team_idx"])

        game_ids.append(game_id)
        if "home_win" in game.index and not pd.isna(game["home_win"]):
            targets[i] = float(game["home_win"])

        _fill_sequence(
            team_history.get(home_idx),
            game_date,
            home_sequences[i],
            home_masks[i],
        )
        _fill_sequence(
            team_history.get(away_idx),
            game_date,
            away_sequences[i],
            away_masks[i],
        )

    home_sequences = np.nan_to_num(home_sequences, nan=0.0)
    away_sequences = np.nan_to_num(away_sequences, nan=0.0)

    logger.info(
        "Built target sequences: %d games, seq_len=%d, n_features=%d",
        n_games,
        seq_len,
        n_features,
    )

    return {
        "home_sequences": home_sequences,
        "away_sequences": away_sequences,
        "home_masks": home_masks,
        "away_masks": away_masks,
        "game_ids": np.array(game_ids),
        "targets": targets,
    }


def build_context_features(
    matchup: pd.DataFrame,
    games: pd.DataFrame,
) -> np.ndarray:
    """Extract context features (schedule, rest days) for each game.

    These are the non-sequence features that feed into the ContextMLP.

    Returns:
        np.ndarray of shape [n_games, n_context_features]
    """
    context_cols = [
        "home_rest_days", "away_rest_days",
        "home_back_to_back", "away_back_to_back",
        "home_games_last_7", "away_games_last_7",
        "home_games_last_14", "away_games_last_14",
    ]

    # Align matchup with games ordering
    matchup_sorted = matchup.sort_values("date").reset_index(drop=True)
    available_cols = [c for c in context_cols if c in matchup_sorted.columns]

    context = matchup_sorted[available_cols].fillna(0).values.astype(np.float32)
    logger.info("Context features: %d games × %d features", context.shape[0], context.shape[1])
    return context


def main():
    """Build and save sequences."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    team_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")

    config = _load_sequence_config()
    seq_len = config["length"]

    logger.info("Building sequences (seq_len=%d)...", seq_len)
    sequences = build_team_sequences(team_logs, games, seq_len=seq_len)

    out_dir = PROCESSED_DIR / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "game_sequences.npz",
        **sequences,
    )
    logger.info("Saved sequences to %s", out_dir / "game_sequences.npz")

    # Also save context features
    matchup = pd.read_parquet(PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet")
    context = build_context_features(matchup, games)
    np.save(out_dir / "context_features.npy", context)
    logger.info("Saved context features to %s", out_dir / "context_features.npy")


if __name__ == "__main__":
    main()
