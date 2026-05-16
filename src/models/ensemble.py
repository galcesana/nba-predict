"""Ensemble meta-model — combines multiple model predictions.

Uses logistic regression to combine:
  - Elo probability
  - XGBoost probability
  - Neural model probability

Fit on validation set, evaluated on test set.

Usage:
    python -m src.models.ensemble
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib

from src.models.calibrate import PlattCalibrator
from src.utils.logging import setup_logging
from src.utils.paths import MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

ENSEMBLE_DIR = MODELS_DIR / "ensembles"


def load_model_predictions(split: str = "test") -> pd.DataFrame:
    """Load predictions from all trained models for a given split.

    Args:
        split: "val" or "test"

    Returns:
        DataFrame with game_id, actual, and per-model prediction columns.
    """
    baselines_dir = MODELS_DIR / "baselines"
    neural_dir = MODELS_DIR / "neural"

    # Load neural predictions
    neural_path = neural_dir / f"neural_predictions_{split}.parquet"
    if not neural_path.exists():
        raise FileNotFoundError(f"Neural predictions not found: {neural_path}")
    neural_df = pd.read_parquet(neural_path)

    # Load baseline predictions
    baseline_preds_path = baselines_dir / f"tabular_predictions_{split}.parquet"
    if baseline_preds_path.exists():
        baseline_df = pd.read_parquet(baseline_preds_path)
    else:
        # Try loading from evaluation results
        baseline_df = None

    # Load Elo predictions
    # Elo doesn't have split files, just one big file.
    elo_path = baselines_dir / "elo_predictions.parquet"
    if elo_path.exists():
        elo_df = pd.read_parquet(elo_path)
    else:
        elo_df = None

    # Build merged dataframe
    merged = neural_df[["game_id", "pred_home_win", "actual_home_win"]].copy()
    merged = merged.rename(columns={"pred_home_win": "neural_prob"})

    if baseline_df is not None and "pred_xgboost" in baseline_df.columns:
        xgb_df = baseline_df[["game_id", "pred_xgboost"]].copy()
        xgb_df = xgb_df.rename(columns={"pred_xgboost": "xgboost_prob"})
        merged = merged.merge(xgb_df, on="game_id", how="left")
    elif baseline_df is not None and "xgboost_prob" in baseline_df.columns:
        merged = merged.merge(
            baseline_df[["game_id", "xgboost_prob"]],
            on="game_id", how="left",
        )
    elif baseline_df is not None and "pred_home_win" in baseline_df.columns:
        xgb_df = baseline_df[["game_id", "pred_home_win"]].copy()
        xgb_df = xgb_df.rename(columns={"pred_home_win": "xgboost_prob"})
        merged = merged.merge(xgb_df, on="game_id", how="left")

    if elo_df is not None:
        elo_cols = [c for c in elo_df.columns if "prob" in c.lower() or "pred" in c.lower()]
        if elo_cols:
            elo_rename = {elo_cols[0]: "elo_prob"}
            merged = merged.merge(
                elo_df[["game_id", elo_cols[0]]].rename(columns=elo_rename),
                on="game_id", how="left",
            )

    return merged


def train_ensemble(
    val_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
) -> dict:
    """Train logistic regression ensemble on validation predictions.

    Args:
        val_preds: Validation set predictions from all models.
        test_preds: Test set predictions from all models.

    Returns:
        Dict with trained model, predictions, and metrics.
    """
    # Determine available models
    prob_cols = [c for c in val_preds.columns if c.endswith("_prob")]
    logger.info("Ensemble input models: %s", prob_cols)

    if len(prob_cols) == 0:
        raise ValueError("No model predictions available for ensemble")

    # Prepare features
    X_val = val_preds[prob_cols].fillna(0.5).values
    y_val = val_preds["actual_home_win"].values

    X_test = test_preds[prob_cols].fillna(0.5).values
    y_test = test_preds["actual_home_win"].values

    # Train logistic regression meta-model
    meta_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    meta_model.fit(X_val, y_val)

    # Predict on val and test
    val_ensemble_probs = meta_model.predict_proba(X_val)[:, 1]
    test_ensemble_probs = meta_model.predict_proba(X_test)[:, 1]

    # Calibrate ensemble
    calibrator = PlattCalibrator()
    calibrator.fit(val_ensemble_probs, y_val)
    test_calibrated_probs = calibrator.predict_proba(test_ensemble_probs)

    # Metrics
    metrics = {
        "ensemble_raw": {
            "accuracy": float(((test_ensemble_probs >= 0.5) == y_test).mean()),
            "log_loss": float(log_loss(y_test, test_ensemble_probs)),
            "brier_score": float(brier_score_loss(y_test, test_ensemble_probs)),
            "roc_auc": float(roc_auc_score(y_test, test_ensemble_probs)),
        },
        "ensemble_calibrated": {
            "accuracy": float(((test_calibrated_probs >= 0.5) == y_test).mean()),
            "log_loss": float(log_loss(y_test, test_calibrated_probs)),
            "brier_score": float(brier_score_loss(y_test, test_calibrated_probs)),
            "roc_auc": float(roc_auc_score(y_test, test_calibrated_probs)),
        },
        "model_weights": {
            col: float(w)
            for col, w in zip(prob_cols, meta_model.coef_[0])
        },
        "model_intercept": float(meta_model.intercept_[0]),
        "n_models": len(prob_cols),
    }

    logger.info("Ensemble test accuracy: %.3f", metrics["ensemble_raw"]["accuracy"])
    logger.info("Ensemble test log_loss: %.4f", metrics["ensemble_raw"]["log_loss"])
    logger.info("Model weights: %s", metrics["model_weights"])

    return {
        "meta_model": meta_model,
        "calibrator": calibrator,
        "test_predictions": test_calibrated_probs,
        "metrics": metrics,
        "prob_cols": prob_cols,
    }


def main():
    """Train and save ensemble model."""
    setup_logging()

    # Load predictions
    logger.info("Loading model predictions...")
    try:
        val_preds = load_model_predictions("val")
        test_preds = load_model_predictions("test")
    except FileNotFoundError as e:
        logger.warning("Cannot build ensemble: %s", e)
        logger.info("Run baseline and neural training first.")
        return

    # Train ensemble
    logger.info("Training ensemble meta-model...")
    results = train_ensemble(val_preds, test_preds)

    # Save
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["meta_model"], ENSEMBLE_DIR / "meta_model.joblib")
    joblib.dump(results["calibrator"], ENSEMBLE_DIR / "calibrator.joblib")

    # Save predictions
    test_preds_out = test_preds[["game_id", "actual_home_win"]].copy()
    test_preds_out["ensemble_prob"] = results["test_predictions"]
    test_preds_out.to_parquet(ENSEMBLE_DIR / "ensemble_predictions_test.parquet", index=False)

    # Save metrics
    with open(ENSEMBLE_DIR / "ensemble_results.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)

    logger.info("Ensemble saved to %s", ENSEMBLE_DIR)


if __name__ == "__main__":
    main()
