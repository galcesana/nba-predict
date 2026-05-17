"""Tabular baseline models: Logistic Regression and XGBoost.

Trains on the matchup feature table from Phase 2, using time-based splits.

Usage:
    python -m src.models.tabular_model
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

BASELINES_DIR = MODELS_DIR / "baselines"


def _load_split_config() -> dict:
    """Load train/val/test split config."""
    with open(CONFIGS_DIR / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    return config["splits"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get numeric model feature columns from a matchup dataset."""
    exclude = {
        "game_id",
        "date",
        "season",
        "season_type",
        "home_team_idx",
        "away_team_idx",
        "target_home_win",
    }
    return [
        column
        for column in df.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(df[column])
    ]


def split_by_season(
    df: pd.DataFrame,
    split_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split matchup dataset into train/val/test by season.

    Returns:
        (train_df, val_df, test_df)
    """
    if split_config is None:
        split_config = _load_split_config()

    train_end = split_config["train_end"]
    val_season = split_config["validation"]
    test_seasons = split_config["test"]

    # Parse season years for ordering (e.g., "2021-22" -> 2021)
    def season_start_year(s: str) -> int:
        return int(s.split("-")[0])

    train = df[df["season"].apply(season_start_year) <= season_start_year(train_end)]
    val = df[df["season"] == val_season]
    test = df[df["season"].isin(test_seasons)]

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
) -> LogisticRegression:
    """Train logistic regression on scaled features."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )
    X_scaled = scaler.transform(X_train)
    model.fit(X_scaled, y_train)
    logger.info("Logistic regression trained (C=%.1f)", model.C)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping on validation set."""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "XGBoost trained: %d trees (best iteration: %d)",
        model.n_estimators, model.best_iteration,
    )
    return model


def main():
    """Train all tabular baselines and save."""
    setup_logging()

    # Load matchup dataset
    matchup_path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet"
    logger.info("Loading matchup dataset from %s", matchup_path)
    df = pd.read_parquet(matchup_path)

    # Split
    split_config = _load_split_config()
    train_df, val_df, test_df = split_by_season(df, split_config)

    feature_cols = get_feature_columns(df)
    logger.info("Using %d features", len(feature_cols))

    # Prepare arrays — fill NaN with 0 for early-season games
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_home_win"].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["target_home_win"].values

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # --- Home baseline ---
    home_rate = y_train.mean()
    logger.info("Home baseline — always predict P(home_win)=%.3f", home_rate)

    # --- Logistic Regression ---
    logger.info("Training logistic regression...")
    lr_model = train_logistic_regression(X_train, y_train, scaler)

    # --- XGBoost ---
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # Save models
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, BASELINES_DIR / "scaler.joblib")
    joblib.dump(lr_model, BASELINES_DIR / "logistic_regression.joblib")
    xgb_model.save_model(str(BASELINES_DIR / "xgboost.json"))

    # Save feature column list
    with open(BASELINES_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save predictions for evaluation
    models_info = {
        "home_baseline": {"home_win_rate": float(home_rate)},
        "logistic_regression": {"C": lr_model.C, "n_features": len(feature_cols)},
        "xgboost": {
            "best_iteration": int(xgb_model.best_iteration),
            "n_estimators": int(xgb_model.n_estimators),
        },
    }
    with open(BASELINES_DIR / "models_info.json", "w") as f:
        json.dump(models_info, f, indent=2)

    # Generate predictions for all splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = split_df[feature_cols].fillna(0).values
        X_scaled = scaler.transform(X)

        preds = pd.DataFrame({
            "game_id": split_df["game_id"].values,
            "season": split_df["season"].values,
            "actual_home_win": split_df["target_home_win"].values,
            "pred_home_baseline": home_rate,
            "pred_logistic": lr_model.predict_proba(X_scaled)[:, 1],
            "pred_xgboost": xgb_model.predict_proba(X)[:, 1],
        })
        preds.to_parquet(BASELINES_DIR / f"tabular_predictions_{split_name}.parquet", index=False)
        logger.info("Saved %s predictions: %d rows", split_name, len(preds))

    logger.info("All tabular baselines trained and saved to %s", BASELINES_DIR)


if __name__ == "__main__":
    main()
