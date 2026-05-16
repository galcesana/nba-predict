"""Prediction inference pipeline.

Generates predictions for a set of target games using all trained models.
Handles feature extraction for both tabular and neural models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

from src.features.build_matchup_dataset import build_matchup_dataset
from src.features.injury_features import INJURY_FEATURE_COLS
from src.features.news_features import NEWS_FEATURE_COLS
from src.features.sequence_builder import build_context_features, build_team_sequences
from src.models.elo import EloModel
from src.models.matchup_fusion_model import MatchupFusionModel
from src.utils.paths import CONFIGS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

BASELINES_DIR = MODELS_DIR / "baselines"
NEURAL_DIR = MODELS_DIR / "neural"
ENSEMBLE_DIR = MODELS_DIR / "ensembles"


class PredictionPipeline:
    def __init__(self, use_cuda: bool = False):
        """Initialize pipeline and load all models/artifacts."""
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self._load_configs()
        self._load_models()

    def _load_configs(self):
        with open(CONFIGS_DIR / "model_config.yaml") as f:
            self.model_config = yaml.safe_load(f)

        with open(BASELINES_DIR / "feature_columns.json") as f:
            self.xgb_feature_cols = json.load(f)

    def _load_models(self):
        logger.info("Loading Elo model...")
        self.elo = EloModel(
            k_factor=self.model_config["elo"]["k_factor"],
            home_advantage=self.model_config["elo"]["home_advantage"],
            initial_rating=self.model_config["elo"]["initial_rating"],
        )
        with open(BASELINES_DIR / "elo_ratings.json") as f:
            ratings = json.load(f)
            self.elo.ratings = {int(k): float(v) for k, v in ratings.items()}

        logger.info("Loading XGBoost model and scaler...")
        self.scaler = joblib.load(BASELINES_DIR / "scaler.joblib")
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(BASELINES_DIR / "xgboost.json"))

        logger.info("Loading Neural model...")
        neural_cfg = self.model_config["neural"]
        
        # Load test results to get exact feature streams
        with open(NEURAL_DIR / "test_results.json") as f:
            test_results = json.load(f)
            
        feature_streams = test_results.get("feature_streams", ["performance", "context", "injury", "news"])
        
        # Determine dims
        # sequence features = 16
        # context features = 8
        # injury = 6
        # news = 6
        self.neural = MatchupFusionModel(
            game_feature_dim=16,
            context_feature_dim=8,
            injury_feature_dim=6,
            news_feature_dim=6,
            hidden_dim=neural_cfg["hidden_dim"],
            num_gru_layers=neural_cfg["num_layers"],
            feature_streams=feature_streams,
        ).to(self.device)
        self.neural.load_state_dict(torch.load(NEURAL_DIR / "best_model.pt", weights_only=True, map_location=self.device))
        self.neural.eval()

        logger.info("Loading Ensemble model...")
        self.meta_model = joblib.load(ENSEMBLE_DIR / "meta_model.joblib")
        self.calibrator = joblib.load(ENSEMBLE_DIR / "calibrator.joblib")

    def _get_top_factors(self, matchup_row: pd.Series) -> list[str]:
        """Generate human-readable factors based on z-scores or thresholds."""
        factors = []
        # Simple heuristic factors for UI
        if matchup_row.get("diff_last_10_net_rating", 0) > 5:
            factors.append("Home team has significantly better recent net rating")
        elif matchup_row.get("diff_last_10_net_rating", 0) < -5:
            factors.append("Away team has significantly better recent net rating")
            
        if matchup_row.get("home_rest_days", 1) == 0:
            factors.append("Home team is on a back-to-back")
        if matchup_row.get("away_rest_days", 1) == 0:
            factors.append("Away team is on a back-to-back")
            
        if not factors:
            factors.append("Matchup appears statistically balanced")
            
        return factors

    def predict_games(
        self,
        target_games: pd.DataFrame,
        historical_games: pd.DataFrame,
        historical_team_logs: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Predict outcomes for target_games.
        
        Args:
            target_games: Games to predict (needs game_id, date, season, home_team_idx, away_team_idx).
            historical_games: All past games up to target_games.
            historical_team_logs: All past team logs up to target_games.
            
        Returns:
            List of dictionaries matching the daily prediction JSON schema.
        """
        logger.info("Starting inference for %d games", len(target_games))
        
        # 1. Combine historical and target for feature building
        # target_games may not have home_win / target. Set dummy.
        combined_games = pd.concat([historical_games, target_games], ignore_index=True)
        if "home_win" not in combined_games.columns:
            combined_games["home_win"] = 0
            
        # Create dummy team logs for target games
        dummy_logs = []
        for _, row in target_games.iterrows():
            dummy_logs.append({
                "game_id": row["game_id"], "team_idx": row["home_team_idx"], 
                "date": row["date"], "season": row["season"], "is_home": 1,
                "points": 0, "opp_points": 0, "win": 0
            })
            dummy_logs.append({
                "game_id": row["game_id"], "team_idx": row["away_team_idx"], 
                "date": row["date"], "season": row["season"], "is_home": 0,
                "points": 0, "opp_points": 0, "win": 0
            })
        dummy_logs_df = pd.DataFrame(dummy_logs)
        # Combine logs
        combined_logs = pd.concat([historical_team_logs, dummy_logs_df], ignore_index=True)
        
        # 2. Build Matchup Dataset
        logger.info("Building matchup dataset...")
        matchup = build_matchup_dataset(combined_games, combined_logs)
        
        # Filter matchup to just target games
        target_matchups = matchup[matchup["game_id"].isin(target_games["game_id"])].copy()
        
        if len(target_matchups) != len(target_games):
            logger.warning("Matchup generation yielded %d rows, expected %d", len(target_matchups), len(target_games))
            
        results = []
        
        # 3. XGBoost probabilities
        X_xgb = target_matchups[self.xgb_feature_cols].fillna(0).values
        xgb_probs = self.xgb.predict_proba(X_xgb)[:, 1]
        
        # 4. Neural Model features
        seq_len = self.model_config["sequence"]["length"]
        sequences = build_team_sequences(combined_logs, combined_games, seq_len=seq_len)
        context = build_context_features(target_matchups, combined_games)
        
        # Find indices in sequences for target games
        game_id_to_idx = {gid: i for i, gid in enumerate(sequences["game_ids"])}
        seq_indices = np.array([game_id_to_idx[gid] for gid in target_matchups["game_id"]])
        
        h_seq = torch.from_numpy(sequences["home_sequences"][seq_indices]).to(self.device)
        a_seq = torch.from_numpy(sequences["away_sequences"][seq_indices]).to(self.device)
        h_mask = torch.from_numpy(sequences["home_masks"][seq_indices]).to(self.device)
        a_mask = torch.from_numpy(sequences["away_masks"][seq_indices]).to(self.device)
        ctx_tensor = torch.from_numpy(context).to(self.device)
        
        # Injury and News (proxy defaults for now)
        n = len(target_matchups)
        injury_dim = len(INJURY_FEATURE_COLS) - 1
        news_dim = len(NEWS_FEATURE_COLS) - 1
        
        h_inj = torch.zeros((n, injury_dim)).to(self.device)
        a_inj = torch.zeros((n, injury_dim)).to(self.device)
        h_news = torch.zeros((n, news_dim)).to(self.device)
        a_news = torch.zeros((n, news_dim)).to(self.device)
        news_avail = torch.zeros((n, 1)).to(self.device)
        
        # 5. Neural predictions
        with torch.no_grad():
            neural_probs = self.neural(
                h_seq, a_seq, ctx_tensor, h_mask, a_mask,
                home_injury=h_inj, away_injury=a_inj,
                home_news=h_news, away_news=a_news, news_available=news_avail
            ).cpu().numpy().squeeze(1)
            
        # Ensure array if single prediction
        if neural_probs.ndim == 0:
            neural_probs = np.array([neural_probs])
            
        # 6. Ensemble and Combine
        for i, (_, row) in enumerate(target_matchups.iterrows()):
            home_idx = int(row["home_team_idx"])
            away_idx = int(row["away_team_idx"])
            
            # Elo probability
            elo_prob = self.elo.predict_proba(home_idx, away_idx)
            
            # Prepare ensemble input
            # Order must match prob_cols from training: ['neural_prob', 'xgboost_prob', 'elo_prob']
            # We can check self.meta_model.feature_names_in_ if sklearn > 1.0, but we'll assume the order.
            # In ensemble.py we load them in that order via glob/merge, let's pass a dataframe to predict.
            ensemble_in = pd.DataFrame([{
                "neural_prob": neural_probs[i],
                "xgboost_prob": xgb_probs[i],
                "elo_prob": elo_prob
            }])
            
            raw_ensemble_prob = self.meta_model.predict_proba(ensemble_in.values)[:, 1][0]
            final_prob = self.calibrator.predict_proba(np.array([raw_ensemble_prob]))[0]
            
            conf = "high" if (final_prob > 0.7 or final_prob < 0.3) else ("medium" if (final_prob > 0.6 or final_prob < 0.4) else "low")
            
            results.append({
                "game_id": row["game_id"],
                "home_team_idx": home_idx,
                "away_team_idx": away_idx,
                "home_win_probability": float(round(final_prob, 4)),
                "away_win_probability": float(round(1 - final_prob, 4)),
                "predicted_winner": "home" if final_prob >= 0.5 else "away",
                "confidence_bucket": conf,
                "component_outputs": {
                    "elo_probability": float(round(elo_prob, 4)),
                    "tabular_probability": float(round(xgb_probs[i], 4)),
                    "sequence_probability": float(round(neural_probs[i], 4)),
                    "final_probability": float(round(final_prob, 4)),
                },
                "top_model_factors": self._get_top_factors(row)
            })
            
        return results
