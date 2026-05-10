"""Evaluate all baseline models and generate metrics + calibration plots.

Produces:
  - models/baselines/evaluation_results.json  — metrics for all models
  - models/baselines/calibration_plot.png      — calibration curves

Usage:
    python -m src.models.evaluate
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.models.calibrate import calibrate_predictions
from src.models.elo import train_elo
from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

BASELINES_DIR = MODELS_DIR / "baselines"


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute all evaluation metrics for binary probability predictions."""
    y_pred_class = (y_pred_proba >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred_class)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }


def compute_calibration_error(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE).

    Returns:
        (ece, bins) where bins is a list of dicts with bin details.
    """
    bins_data = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_weight = 0
    ece = 0.0

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])

        count = mask.sum()
        if count == 0:
            bins_data.append({
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "count": 0,
                "avg_predicted": None,
                "avg_actual": None,
            })
            continue

        avg_predicted = float(y_pred_proba[mask].mean())
        avg_actual = float(y_true[mask].mean())
        gap = abs(avg_predicted - avg_actual)

        ece += gap * count
        total_weight += count

        bins_data.append({
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "count": int(count),
            "avg_predicted": avg_predicted,
            "avg_actual": avg_actual,
            "gap": float(gap),
        })

    if total_weight > 0:
        ece /= total_weight

    return float(ece), bins_data


def plot_calibration_curves(
    model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot calibration curves for multiple models.

    Args:
        model_predictions: {model_name: (y_true, y_pred_proba)}
        output_path: Path to save the plot.
        n_bins: Number of calibration bins.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration plot
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    for idx, (name, (y_true, y_pred)) in enumerate(model_predictions.items()):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_actuals = []

        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            if mask.sum() > 10:
                bin_centers.append(y_pred[mask].mean())
                bin_actuals.append(y_true[mask].mean())

        color = colors[idx % len(colors)]
        ax1.plot(bin_centers, bin_actuals, "o-", label=name, color=color, linewidth=2)

    ax1.set_xlabel("Predicted probability", fontsize=12)
    ax1.set_ylabel("Actual win rate", fontsize=12)
    ax1.set_title("Calibration Curves", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Prediction histogram
    ax2 = axes[1]
    for idx, (name, (_, y_pred)) in enumerate(model_predictions.items()):
        color = colors[idx % len(colors)]
        ax2.hist(y_pred, bins=50, alpha=0.5, label=name, color=color)

    ax2.set_xlabel("Predicted probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Calibration plot saved to %s", output_path)


def rolling_validation(
    games: pd.DataFrame,
    matchup: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    """Run rolling time-based validation.

    Train on seasons up to Y, validate on season Y+1.
    """
    seasons = sorted(matchup["season"].unique())

    # We need at least 5 seasons for training
    fold_results = []
    for i in range(5, len(seasons) - 1):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_mask = matchup["season"].isin(train_seasons)
        val_mask = matchup["season"] == val_season

        X_train = matchup.loc[train_mask, feature_cols].fillna(0).values
        y_train = matchup.loc[train_mask, "target_home_win"].values
        X_val = matchup.loc[val_mask, feature_cols].fillna(0).values
        y_val = matchup.loc[val_mask, "target_home_win"].values

        if len(y_val) == 0:
            continue

        # Quick XGBoost per fold
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, eval_metric="logloss",
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, preds)
        metrics["fold"] = f"train→{val_season}"
        metrics["train_size"] = int(len(y_train))
        metrics["val_size"] = int(len(y_val))
        fold_results.append(metrics)

        logger.info(
            "Rolling fold %s: acc=%.3f, log_loss=%.3f",
            val_season, metrics["accuracy"], metrics["log_loss"],
        )

    return fold_results


def main():
    """Evaluate all baseline models."""
    setup_logging()

    # Load data
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    matchup = pd.read_parquet(PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet")

    # Load feature columns
    with open(BASELINES_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    # Load split config and split data
    from src.models.tabular_model import split_by_season
    train_df, val_df, test_df = split_by_season(matchup)

    # --- Elo predictions ---
    logger.info("Evaluating Elo model...")
    _, elo_preds = train_elo(games)

    # Split Elo predictions into val/test by season
    import yaml
    with open(CONFIGS_DIR / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    val_season = config["splits"]["validation"]
    test_seasons = config["splits"]["test"]

    elo_val = elo_preds[elo_preds["season"] == val_season]
    elo_test = elo_preds[elo_preds["season"].isin(test_seasons)]

    # --- Load tabular predictions ---
    tab_val = pd.read_parquet(BASELINES_DIR / "tabular_predictions_val.parquet")
    tab_test = pd.read_parquet(BASELINES_DIR / "tabular_predictions_test.parquet")

    # --- Evaluate all models on VAL set ---
    all_results = {}
    model_predictions_val = {}

    # Home baseline
    home_rate = train_df["target_home_win"].mean()
    y_val = tab_val["actual_home_win"].values
    y_test = tab_test["actual_home_win"].values

    home_preds_val = np.full(len(y_val), home_rate)
    metrics = compute_metrics(y_val, home_preds_val)
    ece, _ = compute_calibration_error(y_val, home_preds_val)
    metrics["calibration_error"] = ece
    all_results["home_baseline_val"] = metrics
    model_predictions_val["Home Baseline"] = (y_val, home_preds_val)

    # Elo
    elo_val_preds = elo_val["pred_home_win"].values
    elo_val_actuals = elo_val["actual_home_win"].values
    metrics = compute_metrics(elo_val_actuals, elo_val_preds)
    ece, _ = compute_calibration_error(elo_val_actuals, elo_val_preds)
    metrics["calibration_error"] = ece
    all_results["elo_val"] = metrics
    model_predictions_val["Elo"] = (elo_val_actuals, elo_val_preds)

    # Logistic Regression
    lr_preds_val = tab_val["pred_logistic"].values
    metrics = compute_metrics(y_val, lr_preds_val)
    ece, _ = compute_calibration_error(y_val, lr_preds_val)
    metrics["calibration_error"] = ece
    all_results["logistic_regression_val"] = metrics
    model_predictions_val["Logistic Regression"] = (y_val, lr_preds_val)

    # XGBoost
    xgb_preds_val = tab_val["pred_xgboost"].values
    metrics = compute_metrics(y_val, xgb_preds_val)
    ece, _ = compute_calibration_error(y_val, xgb_preds_val)
    metrics["calibration_error"] = ece
    all_results["xgboost_val"] = metrics
    model_predictions_val["XGBoost"] = (y_val, xgb_preds_val)

    # --- Evaluate on TEST set ---
    model_predictions_test = {}

    home_preds_test = np.full(len(y_test), home_rate)
    metrics = compute_metrics(y_test, home_preds_test)
    ece, _ = compute_calibration_error(y_test, home_preds_test)
    metrics["calibration_error"] = ece
    all_results["home_baseline_test"] = metrics
    model_predictions_test["Home Baseline"] = (y_test, home_preds_test)

    elo_test_preds = elo_test["pred_home_win"].values
    elo_test_actuals = elo_test["actual_home_win"].values
    metrics = compute_metrics(elo_test_actuals, elo_test_preds)
    ece, _ = compute_calibration_error(elo_test_actuals, elo_test_preds)
    metrics["calibration_error"] = ece
    all_results["elo_test"] = metrics
    model_predictions_test["Elo"] = (elo_test_actuals, elo_test_preds)

    lr_preds_test = tab_test["pred_logistic"].values
    metrics = compute_metrics(y_test, lr_preds_test)
    ece, _ = compute_calibration_error(y_test, lr_preds_test)
    metrics["calibration_error"] = ece
    all_results["logistic_regression_test"] = metrics
    model_predictions_test["Logistic Regression"] = (y_test, lr_preds_test)

    xgb_preds_test = tab_test["pred_xgboost"].values
    metrics = compute_metrics(y_test, xgb_preds_test)
    ece, _ = compute_calibration_error(y_test, xgb_preds_test)
    metrics["calibration_error"] = ece
    all_results["xgboost_test"] = metrics
    model_predictions_test["XGBoost"] = (y_test, xgb_preds_test)

    # --- Calibrate XGBoost on val, evaluate on test ---
    logger.info("Calibrating XGBoost predictions...")
    calibrator = calibrate_predictions(xgb_preds_val, y_val, method="platt")
    xgb_cal_test = calibrator.predict_proba(xgb_preds_test)
    metrics = compute_metrics(y_test, xgb_cal_test)
    ece, _ = compute_calibration_error(y_test, xgb_cal_test)
    metrics["calibration_error"] = ece
    all_results["xgboost_calibrated_test"] = metrics
    model_predictions_test["XGBoost (calibrated)"] = (y_test, xgb_cal_test)

    # Save calibrator
    joblib.dump(calibrator, BASELINES_DIR / "xgb_platt_calibrator.joblib")

    # --- Rolling validation ---
    logger.info("Running rolling validation...")
    rolling_results = rolling_validation(games, matchup, feature_cols)
    all_results["rolling_validation"] = rolling_results

    # --- Save results ---
    with open(BASELINES_DIR / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Generate calibration plots ---
    plot_calibration_curves(
        model_predictions_test,
        BASELINES_DIR / "calibration_plot.png",
    )

    # --- Print summary ---
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE EVALUATION SUMMARY (Test Set)")
    logger.info("=" * 70)
    for model_name in ["home_baseline_test", "elo_test", "logistic_regression_test",
                       "xgboost_test", "xgboost_calibrated_test"]:
        m = all_results[model_name]
        name = model_name.replace("_test", "").replace("_", " ").title()
        logger.info(
            "%-25s  acc=%.3f  log_loss=%.4f  brier=%.4f  roc_auc=%.3f  ECE=%.4f",
            name, m["accuracy"], m["log_loss"], m["brier_score"],
            m["roc_auc"], m["calibration_error"],
        )
    logger.info("=" * 70)

    logger.info("Evaluation complete. Results saved to %s", BASELINES_DIR)


if __name__ == "__main__":
    main()
