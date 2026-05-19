"""Prediction inference pipeline.

Generates predictions for a set of target games using all trained models.
Builds the same context/injury/news streams used during training, falling back
to proxy defaults when raw daily inputs are unavailable.
"""

import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

from src.features.build_matchup_dataset import append_enriched_features, build_matchup_dataset
from src.features.injury_features import INJURY_FEATURE_COLS, build_injury_features
from src.features.lineup_features import build_lineup_features
from src.features.news_features import NEWS_FEATURE_COLS, build_news_features
from src.features.player_value_features import build_player_value_features
from src.features.projected_availability import build_projected_availability
from src.features.sequence_builder import build_context_features, build_team_sequences_for_games
from src.models.elo import EloModel
from src.models.matchup_fusion_model import MatchupFusionModel
from src.utils.paths import CONFIGS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

BASELINES_DIR = MODELS_DIR / "baselines"
NEURAL_DIR = MODELS_DIR / "neural"
ENSEMBLE_DIR = MODELS_DIR / "ensembles"
NEXTGEN_DIR = MODELS_DIR / "ensembles_nextgen"
ENSEMBLE_MODEL_VERSION = "ensemble_v1"
NEXTGEN_INPUT_COLS = [
    "neural_prob",
    "xgboost_prob",
    "elo_prob",
    "enriched_catboost_prob",
    "enriched_lightgbm_prob",
]
NEXTGEN_SHADOW_MODEL_VERSION = "nextgen_full_raw_v1"
NEXTGEN_VALUE_TUNED_SHADOW_MODEL_VERSION = "nextgen_full_value_tuned_v2"
DEFAULT_PRODUCTION_MODEL = "nextgen"


def _resolve_production_model(value: str | None = None) -> str:
    """Resolve the active production model family."""
    selected = (value or os.getenv("NBA_PREDICT_PRODUCTION_MODEL") or DEFAULT_PRODUCTION_MODEL)
    selected = selected.strip().lower()
    aliases = {
        "ensemble": "ensemble",
        "ensemble_v1": "ensemble",
        "legacy": "ensemble",
        "nextgen": "nextgen",
        "nextgen_full": "nextgen",
        "nextgen_full_value_tuned_v2": "nextgen",
    }
    if selected not in aliases:
        logger.warning(
            "Unknown NBA_PREDICT_PRODUCTION_MODEL=%r; falling back to %s.",
            selected,
            DEFAULT_PRODUCTION_MODEL,
        )
        return DEFAULT_PRODUCTION_MODEL
    return aliases[selected]


def _env_flag_enabled(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


class PredictionPipeline:
    def __init__(
        self,
        use_cuda: bool = False,
        enable_nextgen_shadow: bool | None = None,
        production_model: str | None = None,
    ):
        """Initialize pipeline and load all models/artifacts."""
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.production_model = _resolve_production_model(production_model)
        self.active_model_version = ENSEMBLE_MODEL_VERSION
        self.nextgen_shadow_requested = (
            _env_flag_enabled("NBA_PREDICT_NEXTGEN_SHADOW")
            if enable_nextgen_shadow is None
            else enable_nextgen_shadow
        )
        self.nextgen_shadow_enabled = False
        self._nextgen_player_logs_cache: pd.DataFrame | None = None
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

        feature_streams = test_results.get(
            "feature_streams",
            ["performance", "context", "injury", "news"],
        )

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
        self.neural.load_state_dict(
            torch.load(
                NEURAL_DIR / "best_model.pt",
                weights_only=True,
                map_location=self.device,
            )
        )
        self.neural.eval()

        logger.info("Loading Ensemble model...")
        self.meta_model = joblib.load(ENSEMBLE_DIR / "meta_model.joblib")
        self.calibrator = joblib.load(ENSEMBLE_DIR / "calibrator.joblib")
        self._load_nextgen_shadow_models()

    def _load_nextgen_shadow_models(self) -> None:
        """Load next-gen candidate artifacts when shadow mode is explicitly enabled."""
        if not (self.nextgen_shadow_requested or self.production_model == "nextgen"):
            return

        required_paths = {
            "catboost": NEXTGEN_DIR / "enriched_catboost.joblib",
            "lightgbm": NEXTGEN_DIR / "enriched_lightgbm.joblib",
            "feature_columns": NEXTGEN_DIR / "enriched_feature_columns.json",
            "meta_model": NEXTGEN_DIR / "meta_model.joblib",
            "calibrator": NEXTGEN_DIR / "calibrator.joblib",
            "player_logs": PROCESSED_DIR / "player_game_logs" / "player_game_logs.parquet",
        }
        missing = [str(path) for path in required_paths.values() if not path.exists()]
        if missing:
            logger.warning("Next-gen shadow mode disabled; missing artifacts: %s", missing)
            return

        logger.info("Loading next-gen shadow artifacts...")
        self.nextgen_catboost = joblib.load(required_paths["catboost"])
        self.nextgen_lightgbm = joblib.load(required_paths["lightgbm"])
        self.nextgen_shadow_model_version = NEXTGEN_SHADOW_MODEL_VERSION
        try:
            with open(required_paths["feature_columns"]) as f:
                feature_payload = json.load(f)
            if isinstance(feature_payload, list):
                # Backward compatibility for the original shared-column shadow bundle.
                self.nextgen_catboost_feature_cols = feature_payload
                self.nextgen_lightgbm_feature_cols = feature_payload
            else:
                model_features = feature_payload["models"]
                self.nextgen_catboost_feature_cols = model_features["catboost"]["columns"]
                self.nextgen_lightgbm_feature_cols = model_features["lightgbm"]["columns"]
                if feature_payload.get("schema_version") == "value_tuned_inputs_v1":
                    self.nextgen_shadow_model_version = (
                        NEXTGEN_VALUE_TUNED_SHADOW_MODEL_VERSION
                    )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Next-gen shadow mode disabled; invalid feature-column metadata: %s",
                exc,
            )
            return
        self.nextgen_meta_model = joblib.load(required_paths["meta_model"])
        self.nextgen_calibrator = joblib.load(required_paths["calibrator"])
        self.nextgen_player_logs_path = required_paths["player_logs"]
        self.nextgen_shadow_enabled = True
        if self.production_model == "nextgen":
            self.active_model_version = self.nextgen_shadow_model_version

    def _build_auxiliary_feature_arrays(
        self,
        combined_logs: pd.DataFrame,
        target_games: pd.DataFrame,
        target_matchups: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """Build injury/news inputs plus provenance metadata for target games."""
        injury_cols = [c for c in INJURY_FEATURE_COLS if c != "injury_data_available"]
        news_cols = [c for c in NEWS_FEATURE_COLS if c != "news_available"]
        target_game_ids = set(target_matchups["game_id"])

        injury_frames = []
        injury_path = PROCESSED_DIR / "injury_features" / "injury_features.parquet"
        if injury_path.exists():
            precomputed_injury = pd.read_parquet(injury_path)
            injury_frames.append(
                precomputed_injury[precomputed_injury["game_id"].isin(target_game_ids)]
            )

        injury_game_ids = set()
        for frame in injury_frames:
            injury_game_ids.update(frame["game_id"].tolist())
        missing_injury_games = target_games[
            target_games["game_id"].isin(target_game_ids - injury_game_ids)
        ]
        if not missing_injury_games.empty:
            injury_frames.append(
                build_injury_features(
                    missing_injury_games,
                    combined_logs,
                    allow_live_fetch=True,
                )
            )

        injury_df = (
            pd.concat(injury_frames, ignore_index=True).drop_duplicates(
                ["game_id", "team_idx"],
                keep="last",
            )
            if injury_frames
            else pd.DataFrame()
        )

        news_frames = []
        news_path = PROCESSED_DIR / "news_features" / "news_features.parquet"
        if news_path.exists():
            precomputed_news = pd.read_parquet(news_path)
            news_frames.append(
                precomputed_news[precomputed_news["game_id"].isin(target_game_ids)]
            )

        news_game_ids = set()
        for frame in news_frames:
            news_game_ids.update(frame["game_id"].tolist())
        missing_news_games = target_games[
            target_games["game_id"].isin(target_game_ids - news_game_ids)
        ]
        if not missing_news_games.empty:
            news_frames.append(build_news_features(missing_news_games, allow_live_fetch=True))

        news_df = (
            pd.concat(news_frames, ignore_index=True).drop_duplicates(
                ["game_id", "team_idx"],
                keep="last",
            )
            if news_frames
            else pd.DataFrame()
        )

        injury_lookup = (
            injury_df.set_index(["game_id", "team_idx"]) if not injury_df.empty else None
        )
        news_lookup = news_df.set_index(["game_id", "team_idx"]) if not news_df.empty else None

        home_injury = []
        away_injury = []
        home_news = []
        away_news = []
        news_available = []
        context_details = []

        def _metadata_text(source: pd.Series | None, key: str) -> str:
            if source is None:
                return ""
            value = source.get(key)
            if value is None or pd.isna(value):
                return ""
            return str(value)

        for _, row in target_matchups.iterrows():
            game_id = row["game_id"]
            home_idx = int(row["home_team_idx"])
            away_idx = int(row["away_team_idx"])

            home_key = (game_id, home_idx)
            away_key = (game_id, away_idx)

            home_injury_row = None
            if injury_lookup is not None and home_key in injury_lookup.index:
                home_injury_row = injury_lookup.loc[home_key]
                home_injury.append(home_injury_row[injury_cols].to_numpy(dtype=np.float32))
            else:
                home_injury.append(np.zeros(len(injury_cols), dtype=np.float32))

            away_injury_row = None
            if injury_lookup is not None and away_key in injury_lookup.index:
                away_injury_row = injury_lookup.loc[away_key]
                away_injury.append(away_injury_row[injury_cols].to_numpy(dtype=np.float32))
            else:
                away_injury.append(np.zeros(len(injury_cols), dtype=np.float32))

            home_news_row = None
            if news_lookup is not None and home_key in news_lookup.index:
                home_news_row = news_lookup.loc[home_key]
                home_news.append(home_news_row[news_cols].to_numpy(dtype=np.float32))
                availability = float(home_news_row["news_available"])
            else:
                home_news.append(np.zeros(len(news_cols), dtype=np.float32))
                availability = 0.0

            away_news_row = None
            if news_lookup is not None and away_key in news_lookup.index:
                away_news_row = news_lookup.loc[away_key]
                away_news.append(away_news_row[news_cols].to_numpy(dtype=np.float32))
                availability = max(availability, float(away_news_row["news_available"]))
            else:
                away_news.append(np.zeros(len(news_cols), dtype=np.float32))

            news_available.append([availability])
            home_injury_available = bool(
                home_injury_row is not None
                and float(home_injury_row.get("injury_data_available", 0.0)) > 0
            )
            away_injury_available = bool(
                away_injury_row is not None
                and float(away_injury_row.get("injury_data_available", 0.0)) > 0
            )
            home_news_available = bool(
                home_news_row is not None and float(home_news_row.get("news_available", 0.0)) > 0
            )
            away_news_available = bool(
                away_news_row is not None and float(away_news_row.get("news_available", 0.0)) > 0
            )
            context_details.append(
                {
                    "injury_mode": (
                        "live"
                        if home_injury_available and away_injury_available
                        else "partial"
                        if home_injury_available or away_injury_available
                        else "fallback"
                    ),
                    "news_mode": (
                        "live"
                        if home_news_available and away_news_available
                        else "partial"
                        if home_news_available or away_news_available
                        else "fallback"
                    ),
                    "home_injury_data_available": home_injury_available,
                    "away_injury_data_available": away_injury_available,
                    "home_players_out": int(home_injury_row.get("players_out_count", 0))
                    if home_injury_row is not None
                    else 0,
                    "away_players_out": int(away_injury_row.get("players_out_count", 0))
                    if away_injury_row is not None
                    else 0,
                    "home_estimated_value_missing": float(
                        home_injury_row.get("estimated_value_missing", 0.0)
                    )
                    if home_injury_row is not None
                    else 0.0,
                    "away_estimated_value_missing": float(
                        away_injury_row.get("estimated_value_missing", 0.0)
                    )
                    if away_injury_row is not None
                    else 0.0,
                    "home_questionable": int(home_injury_row.get("players_questionable_count", 0))
                    if home_injury_row is not None
                    else 0,
                    "away_questionable": int(away_injury_row.get("players_questionable_count", 0))
                    if away_injury_row is not None
                    else 0,
                    "injury_report_generated_at": max(
                        _metadata_text(home_injury_row, "report_generated_at"),
                        _metadata_text(away_injury_row, "report_generated_at"),
                    )
                    or None,
                    "home_news_available": home_news_available,
                    "away_news_available": away_news_available,
                    "home_article_volume_24h": int(home_news_row.get("article_volume_24h", 0))
                    if home_news_row is not None
                    else 0,
                    "away_article_volume_24h": int(away_news_row.get("article_volume_24h", 0))
                    if away_news_row is not None
                    else 0,
                    "home_weighted_sentiment_72h": float(
                        home_news_row.get("weighted_sentiment_72h", 0.0)
                    )
                    if home_news_row is not None
                    else 0.0,
                    "away_weighted_sentiment_72h": float(
                        away_news_row.get("weighted_sentiment_72h", 0.0)
                    )
                    if away_news_row is not None
                    else 0.0,
                    "home_article_count_72h": int(home_news_row.get("article_count_72h", 0))
                    if home_news_row is not None
                    else 0,
                    "away_article_count_72h": int(away_news_row.get("article_count_72h", 0))
                    if away_news_row is not None
                    else 0,
                    "latest_article_at": max(
                        _metadata_text(home_news_row, "latest_article_at"),
                        _metadata_text(away_news_row, "latest_article_at"),
                    )
                    or None,
                    "news_collected_at": max(
                        _metadata_text(home_news_row, "news_collected_at"),
                        _metadata_text(away_news_row, "news_collected_at"),
                    )
                    or None,
                }
            )

        return (
            np.array(home_injury, dtype=np.float32),
            np.array(away_injury, dtype=np.float32),
            np.array(home_news, dtype=np.float32),
            np.array(away_news, dtype=np.float32),
            np.array(news_available, dtype=np.float32),
            context_details,
        )

    def _build_nextgen_shadow_probabilities(
        self,
        *,
        target_games: pd.DataFrame,
        target_matchups: pd.DataFrame,
        neural_probs: np.ndarray,
        xgb_probs: np.ndarray,
    ) -> dict[str, dict[str, float | str]]:
        """Build opt-in next-gen candidate probabilities for shadow comparison."""
        if not self.nextgen_shadow_enabled:
            return {}

        if self._nextgen_player_logs_cache is None:
            player_logs = pd.read_parquet(self.nextgen_player_logs_path)
            player_logs["date"] = pd.to_datetime(player_logs["date"])
            self._nextgen_player_logs_cache = player_logs
        else:
            player_logs = self._nextgen_player_logs_cache
        target_dates = pd.to_datetime(target_games["date"])
        cutoff = pd.Timestamp(target_dates.min())
        historical_player_logs = player_logs[player_logs["date"] < cutoff].copy()
        if historical_player_logs.empty:
            logger.warning(
                "Next-gen shadow unavailable; no historical player logs before %s.",
                cutoff,
            )
            return {}

        player_value_features = build_player_value_features(
            target_games,
            historical_player_logs,
        )
        projected_availability, _ = build_projected_availability(
            target_games,
            historical_player_logs,
            player_value_features=player_value_features,
        )
        lineup_features = build_lineup_features(
            target_games,
            historical_player_logs,
            projected_availability,
        )
        enriched = append_enriched_features(
            target_matchups,
            target_games,
            projected_availability=projected_availability,
            lineup_features_df=lineup_features,
        )

        target_game_ids = target_matchups["game_id"].astype(str).tolist()
        enriched_targets = (
            enriched.assign(game_id=enriched["game_id"].astype(str))
            .drop_duplicates("game_id", keep="last")
            .set_index("game_id")
            .reindex(target_game_ids)
        )
        X_catboost = enriched_targets.reindex(
            columns=self.nextgen_catboost_feature_cols
        ).fillna(0.0)
        X_lightgbm = enriched_targets.reindex(
            columns=self.nextgen_lightgbm_feature_cols
        ).fillna(0.0)
        if X_catboost.empty or X_lightgbm.empty:
            logger.warning("Next-gen shadow unavailable; enriched target rows were not generated.")
            return {}
        enriched_catboost_prob = self.nextgen_catboost.predict_proba(
            X_catboost.to_numpy()
        )[:, 1]
        enriched_lightgbm_prob = self.nextgen_lightgbm.predict_proba(X_lightgbm)[:, 1]
        elo_probs = np.array(
            [
                self.elo.predict_proba(int(row["home_team_idx"]), int(row["away_team_idx"]))
                for _, row in target_matchups.iterrows()
            ]
        )
        meta_frame = pd.DataFrame(
            {
                "neural_prob": neural_probs,
                "xgboost_prob": xgb_probs,
                "elo_prob": elo_probs,
                "enriched_catboost_prob": enriched_catboost_prob,
                "enriched_lightgbm_prob": enriched_lightgbm_prob,
            }
        )[NEXTGEN_INPUT_COLS]
        raw_probs = self.nextgen_meta_model.predict_proba(meta_frame.to_numpy())[:, 1]
        calibrated_probs = self.nextgen_calibrator.predict_proba(raw_probs)

        return {
            game_id: {
                "model_version": self.nextgen_shadow_model_version,
                "nextgen_shadow_probability": float(raw_prob),
                "nextgen_shadow_calibrated_probability": float(calibrated_prob),
                "enriched_catboost_probability": float(cat_prob),
                "enriched_lightgbm_probability": float(lgbm_prob),
            }
            for game_id, raw_prob, calibrated_prob, cat_prob, lgbm_prob in zip(
                target_game_ids,
                raw_probs,
                calibrated_probs,
                enriched_catboost_prob,
                enriched_lightgbm_prob,
                strict=False,
            )
        }

    def _get_top_factors(
        self,
        matchup_row: pd.Series,
        home_injury: np.ndarray,
        away_injury: np.ndarray,
        home_news: np.ndarray,
        away_news: np.ndarray,
        news_available: float,
    ) -> list[str]:
        """Generate human-readable factors based on z-scores or thresholds."""
        factors = []
        if matchup_row.get("diff_last_10_net_rating", 0) > 5:
            factors.append("Home team has significantly better recent net rating")
        elif matchup_row.get("diff_last_10_net_rating", 0) < -5:
            factors.append("Away team has significantly better recent net rating")

        if matchup_row.get("home_rest_days", 1) == 0:
            factors.append("Home team is on a back-to-back")
        if matchup_row.get("away_rest_days", 1) == 0:
            factors.append("Away team is on a back-to-back")

        injury_gap = float(away_injury[-1] - home_injury[-1])
        if injury_gap > 1.5:
            factors.append("Away team projects as more shorthanded")
        elif injury_gap < -1.5:
            factors.append("Home team projects as more shorthanded")

        if news_available > 0:
            sentiment_gap = float(home_news[1] - away_news[1])
            if sentiment_gap > 0.15:
                factors.append("Home team has more positive recent news sentiment")
            elif sentiment_gap < -0.15:
                factors.append("Away team has more positive recent news sentiment")

        if not factors:
            factors.append("Matchup appears statistically balanced")

        return factors[:3]

    def predict_games(
        self,
        target_games: pd.DataFrame,
        historical_games: pd.DataFrame,
        historical_team_logs: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Predict outcomes for target_games.

        Args:
            target_games: Games to predict with game_id, date, season,
                home_team_idx, and away_team_idx.
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
            dummy_logs.append(
                {
                    "game_id": row["game_id"],
                    "team_idx": row["home_team_idx"],
                    "date": row["date"],
                    "season": row["season"],
                    "is_home": 1,
                    "points": 0,
                    "opp_points": 0,
                    "win": 0,
                }
            )
            dummy_logs.append(
                {
                    "game_id": row["game_id"],
                    "team_idx": row["away_team_idx"],
                    "date": row["date"],
                    "season": row["season"],
                    "is_home": 0,
                    "points": 0,
                    "opp_points": 0,
                    "win": 0,
                }
            )
        dummy_logs_df = pd.DataFrame(dummy_logs)
        # Combine logs
        combined_logs = pd.concat([historical_team_logs, dummy_logs_df], ignore_index=True)

        # 2. Build Matchup Dataset
        logger.info("Building matchup dataset...")
        matchup = build_matchup_dataset(combined_games, combined_logs)

        # Filter matchup to just target games
        target_matchups = matchup[matchup["game_id"].isin(target_games["game_id"])].copy()

        if len(target_matchups) != len(target_games):
            logger.warning(
                "Matchup generation yielded %d rows, expected %d",
                len(target_matchups),
                len(target_games),
            )

        results = []

        # 3. XGBoost probabilities
        X_xgb = target_matchups[self.xgb_feature_cols].fillna(0).values
        xgb_probs = self.xgb.predict_proba(X_xgb)[:, 1]

        # 4. Neural Model features
        seq_len = self.model_config["sequence"]["length"]
        sequences = build_team_sequences_for_games(
            historical_team_logs,
            target_matchups,
            seq_len=seq_len,
        )
        context = build_context_features(target_matchups, combined_games)

        # Find indices in sequences for target games
        game_id_to_idx = {gid: i for i, gid in enumerate(sequences["game_ids"])}
        seq_indices = np.array([game_id_to_idx[gid] for gid in target_matchups["game_id"]])

        h_seq = torch.from_numpy(sequences["home_sequences"][seq_indices]).to(self.device)
        a_seq = torch.from_numpy(sequences["away_sequences"][seq_indices]).to(self.device)
        h_mask = torch.from_numpy(sequences["home_masks"][seq_indices]).to(self.device)
        a_mask = torch.from_numpy(sequences["away_masks"][seq_indices]).to(self.device)
        ctx_tensor = torch.from_numpy(context).to(self.device)

        (
            home_injury_arr,
            away_injury_arr,
            home_news_arr,
            away_news_arr,
            news_avail_arr,
            context_details,
        ) = self._build_auxiliary_feature_arrays(combined_logs, target_games, target_matchups)

        h_inj = torch.from_numpy(home_injury_arr).to(self.device)
        a_inj = torch.from_numpy(away_injury_arr).to(self.device)
        h_news = torch.from_numpy(home_news_arr).to(self.device)
        a_news = torch.from_numpy(away_news_arr).to(self.device)
        news_avail = torch.from_numpy(news_avail_arr).to(self.device)

        # 5. Neural predictions
        with torch.no_grad():
            neural_probs = self.neural(
                h_seq,
                a_seq,
                ctx_tensor,
                h_mask,
                a_mask,
                home_injury=h_inj,
                away_injury=a_inj,
                home_news=h_news,
                away_news=a_news,
                news_available=news_avail,
            ).cpu().numpy().squeeze(1)

        # Ensure array if single prediction
        if neural_probs.ndim == 0:
            neural_probs = np.array([neural_probs])

        nextgen_shadow = self._build_nextgen_shadow_probabilities(
            target_games=target_games,
            target_matchups=target_matchups,
            neural_probs=neural_probs,
            xgb_probs=xgb_probs,
        )

        # 6. Ensemble and Combine
        for i, (_, row) in enumerate(target_matchups.iterrows()):
            home_idx = int(row["home_team_idx"])
            away_idx = int(row["away_team_idx"])
            game_id = str(row["game_id"])

            # Elo probability
            elo_prob = self.elo.predict_proba(home_idx, away_idx)

            # Prepare ensemble input
            ensemble_in = pd.DataFrame(
                [
                    {
                        "neural_prob": neural_probs[i],
                        "xgboost_prob": xgb_probs[i],
                        "elo_prob": elo_prob,
                    }
                ]
            )

            raw_ensemble_prob = self.meta_model.predict_proba(ensemble_in.values)[:, 1][0]
            ensemble_final_prob = self.calibrator.predict_proba(
                np.array([raw_ensemble_prob])
            )[0]
            shadow = nextgen_shadow.get(game_id)
            final_prob = float(ensemble_final_prob)
            if self.production_model == "nextgen" and shadow:
                final_prob = float(shadow["nextgen_shadow_probability"])

            if final_prob > 0.7 or final_prob < 0.3:
                conf = "high"
            elif final_prob > 0.6 or final_prob < 0.4:
                conf = "medium"
            else:
                conf = "low"

            component_outputs = {
                "elo_probability": float(round(elo_prob, 4)),
                "tabular_probability": float(round(xgb_probs[i], 4)),
                "sequence_probability": float(round(neural_probs[i], 4)),
                "ensemble_v1_probability": float(round(ensemble_final_prob, 4)),
                "final_probability": float(round(final_prob, 4)),
            }
            context = dict(context_details[i])
            if shadow:
                shadow_prob = float(shadow["nextgen_shadow_probability"])
                component_outputs.update(
                    {
                        "nextgen_shadow_probability": float(round(shadow_prob, 4)),
                        "nextgen_shadow_calibrated_probability": float(
                            round(float(shadow["nextgen_shadow_calibrated_probability"]), 4)
                        ),
                        "enriched_catboost_probability": float(
                            round(float(shadow["enriched_catboost_probability"]), 4)
                        ),
                        "enriched_lightgbm_probability": float(
                            round(float(shadow["enriched_lightgbm_probability"]), 4)
                        ),
                    }
                )
                context["nextgen_shadow_mode"] = "available"
                context["nextgen_shadow_model_version"] = shadow["model_version"]
                context["nextgen_shadow_delta"] = float(
                    round(shadow_prob - ensemble_final_prob, 4)
                )
                if self.production_model == "nextgen":
                    context["nextgen_shadow_mode"] = "promoted"
                    context["production_baseline_model_version"] = ENSEMBLE_MODEL_VERSION
                    context["production_baseline_delta"] = context["nextgen_shadow_delta"]
            elif self.production_model == "nextgen":
                context["production_model_fallback"] = ENSEMBLE_MODEL_VERSION
            elif self.nextgen_shadow_requested:
                context["nextgen_shadow_mode"] = "unavailable"

            results.append(
                {
                    "game_id": row["game_id"],
                    "home_team_idx": home_idx,
                    "away_team_idx": away_idx,
                    "home_win_probability": float(round(final_prob, 4)),
                    "away_win_probability": float(round(1 - final_prob, 4)),
                    "predicted_winner": "home" if final_prob >= 0.5 else "away",
                    "confidence_bucket": conf,
                    "component_outputs": component_outputs,
                    "context_details": context,
                    "shadow_outputs": shadow or {},
                    "top_model_factors": self._get_top_factors(
                        row,
                        home_injury_arr[i],
                        away_injury_arr[i],
                        home_news_arr[i],
                        away_news_arr[i],
                        float(news_avail_arr[i, 0]),
                    ),
                }
            )

        return results
