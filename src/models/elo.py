"""Elo rating system for NBA game prediction.

Computes Elo ratings for each team based on game results.
Produces P(home_win) = 1 / (1 + 10^((away_elo - home_elo - home_adv) / 400)).

Usage:
    python -m src.models.elo
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

BASELINES_DIR = MODELS_DIR / "baselines"


def _load_elo_config() -> dict:
    """Load Elo configuration from model_config.yaml."""
    with open(CONFIGS_DIR / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    return config["elo"]


class EloModel:
    """Standard Elo rating system with home-court advantage."""

    def __init__(
        self,
        k_factor: float = 20,
        home_advantage: float = 100,
        initial_rating: float = 1500,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: dict[int, float] = {}

    def _get_rating(self, team_idx: int) -> float:
        return self.ratings.get(team_idx, self.initial_rating)

    def predict_proba(self, home_idx: int, away_idx: int) -> float:
        """Predict P(home_win) using current Elo ratings."""
        home_elo = self._get_rating(home_idx) + self.home_advantage
        away_elo = self._get_rating(away_idx)
        expected = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        return expected

    def update(self, home_idx: int, away_idx: int, home_won: int) -> None:
        """Update ratings after a game result."""
        expected = self.predict_proba(home_idx, away_idx)
        actual = float(home_won)

        home_elo = self._get_rating(home_idx)
        away_elo = self._get_rating(away_idx)

        self.ratings[home_idx] = home_elo + self.k_factor * (actual - expected)
        self.ratings[away_idx] = away_elo + self.k_factor * (expected - actual)

    def apply_season_regression(self, factor: float = 0.75) -> None:
        """Regress ratings toward the mean between seasons.

        This prevents ratings from drifting too far over multiple seasons.
        """
        mean_rating = np.mean(list(self.ratings.values())) if self.ratings else self.initial_rating
        for team_idx in self.ratings:
            self.ratings[team_idx] = (
                factor * self.ratings[team_idx] + (1 - factor) * mean_rating
            )


def train_elo(games: pd.DataFrame, config: dict | None = None) -> tuple[EloModel, pd.DataFrame]:
    """Train Elo model on games, returning model and per-game predictions.

    Games must be sorted by date. For each game, we first predict, then update.

    Returns:
        (model, predictions_df) where predictions_df has:
        game_id, season, home_team_idx, away_team_idx, pred_home_win, actual_home_win
    """
    if config is None:
        config = _load_elo_config()

    model = EloModel(
        k_factor=config["k_factor"],
        home_advantage=config["home_advantage"],
        initial_rating=config["initial_rating"],
    )

    games = games.sort_values("date").copy()
    predictions = []
    prev_season = None

    for _, row in games.iterrows():
        season = row["season"]
        # Apply season regression at season boundaries
        if prev_season is not None and season != prev_season:
            model.apply_season_regression()
        prev_season = season

        home_idx = int(row["home_team_idx"])
        away_idx = int(row["away_team_idx"])
        home_won = int(row["home_win"])

        # Predict BEFORE updating
        pred = model.predict_proba(home_idx, away_idx)
        predictions.append({
            "game_id": row["game_id"],
            "season": season,
            "home_team_idx": home_idx,
            "away_team_idx": away_idx,
            "pred_home_win": pred,
            "actual_home_win": home_won,
        })

        # Update AFTER predicting
        model.update(home_idx, away_idx, home_won)

    preds_df = pd.DataFrame(predictions)
    logger.info(
        "Elo training complete: %d games, %d teams rated",
        len(preds_df), len(model.ratings),
    )
    return model, preds_df


def main():
    """Train Elo model and save results."""
    setup_logging()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    config = _load_elo_config()

    logger.info("Training Elo model (K=%d, home_adv=%d)...",
                config["k_factor"], config["home_advantage"])

    model, preds_df = train_elo(games, config)

    # Save predictions
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    preds_df.to_parquet(BASELINES_DIR / "elo_predictions.parquet", index=False)

    # Save final ratings
    ratings = {str(k): v for k, v in model.ratings.items()}
    with open(BASELINES_DIR / "elo_ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)

    logger.info("Saved Elo predictions and ratings to %s", BASELINES_DIR)


if __name__ == "__main__":
    main()
