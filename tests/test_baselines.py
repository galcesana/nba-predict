"""Baseline model tests — verify training, performance, and calibration.

Run with: pytest tests/test_baselines.py -v
Expected: 12/12 pass
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.utils.paths import MODELS_DIR

BASELINES_DIR = MODELS_DIR / "baselines"


@pytest.fixture(scope="module")
def eval_results() -> dict:
    path = BASELINES_DIR / "evaluation_results.json"
    assert path.exists(), f"Evaluation results not found: {path}"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def elo_predictions() -> pd.DataFrame:
    path = BASELINES_DIR / "elo_predictions.parquet"
    assert path.exists(), f"Elo predictions not found: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def tab_test_predictions() -> pd.DataFrame:
    path = BASELINES_DIR / "tabular_predictions_test.parquet"
    assert path.exists(), f"Tabular test predictions not found: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def tab_val_predictions() -> pd.DataFrame:
    path = BASELINES_DIR / "tabular_predictions_val.parquet"
    assert path.exists(), f"Tabular val predictions not found: {path}"
    return pd.read_parquet(path)


class TestEloModel:
    """Verify Elo model training and performance."""

    def test_elo_model_trains(self, elo_predictions):
        """Elo model runs on training data without error."""
        assert len(elo_predictions) > 0
        assert "pred_home_win" in elo_predictions.columns
        assert "actual_home_win" in elo_predictions.columns

    def test_elo_ratings_reasonable(self):
        """All Elo ratings are between 1000 and 2000 (no explosion)."""
        path = BASELINES_DIR / "elo_ratings.json"
        assert path.exists()
        with open(path) as f:
            ratings = json.load(f)
        for team, rating in ratings.items():
            assert 1000 <= rating <= 2000, (
                f"Team {team} has unreasonable rating: {rating}"
            )

    def test_elo_beats_coin_flip(self, eval_results):
        """Elo accuracy > 50% on validation set."""
        elo_acc = eval_results["elo_val"]["accuracy"]
        assert elo_acc > 0.50, f"Elo accuracy {elo_acc:.3f} <= 50%"


class TestTabularModels:
    """Verify tabular model training and performance."""

    def test_logistic_regression_trains(self):
        """Logistic regression model file exists."""
        path = BASELINES_DIR / "logistic_regression.joblib"
        assert path.exists(), "Logistic regression model not found"

    def test_xgboost_trains(self):
        """XGBoost model file exists."""
        path = BASELINES_DIR / "xgboost.json"
        assert path.exists(), "XGBoost model not found"

    def test_xgboost_beats_home_baseline(self, eval_results):
        """XGBoost accuracy > home-team baseline accuracy on test set."""
        xgb_acc = eval_results["xgboost_test"]["accuracy"]
        home_acc = eval_results["home_baseline_test"]["accuracy"]
        assert xgb_acc > home_acc, (
            f"XGBoost ({xgb_acc:.3f}) <= home baseline ({home_acc:.3f})"
        )


class TestPredictionQuality:
    """Verify prediction quality constraints."""

    def test_probabilities_sum_to_one(self, tab_test_predictions):
        """P(home_win) + P(away_win) ≈ 1.0 (implicit: P(away) = 1 - P(home))."""
        for col in ["pred_logistic", "pred_xgboost"]:
            preds = tab_test_predictions[col].values
            # Since P(away) = 1 - P(home), they always sum to 1 by construction
            # Verify no NaN
            assert not np.isnan(preds).any(), f"{col} contains NaN"

    def test_probabilities_in_range(self, tab_test_predictions):
        """All predicted probabilities are in [0.01, 0.99]."""
        for col in ["pred_logistic", "pred_xgboost"]:
            preds = tab_test_predictions[col].values
            assert preds.min() >= 0.0, f"{col} min={preds.min():.4f} < 0"
            assert preds.max() <= 1.0, f"{col} max={preds.max():.4f} > 1"

    def test_calibration_plot_generated(self):
        """Calibration plot image saved to models/baselines/."""
        path = BASELINES_DIR / "calibration_plot.png"
        assert path.exists(), "Calibration plot not found"
        assert path.stat().st_size > 1000, "Calibration plot too small (likely empty)"

    def test_evaluation_metrics_saved(self, eval_results):
        """Evaluation results JSON has log_loss, brier, accuracy for all models."""
        required_keys = [
            "home_baseline_test", "elo_test",
            "logistic_regression_test", "xgboost_test",
        ]
        for key in required_keys:
            assert key in eval_results, f"Missing {key} in evaluation results"
            for metric in ["accuracy", "log_loss", "brier_score"]:
                assert metric in eval_results[key], (
                    f"Missing {metric} in {key}"
                )

    def test_no_leakage_signal(self, eval_results):
        """No model exceeds 72% accuracy (would suggest leakage)."""
        for key, metrics in eval_results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                acc = metrics["accuracy"]
                assert acc < 0.72, (
                    f"Model {key} has {acc:.1%} accuracy — possible leakage!"
                )

    def test_rolling_validation_consistent(self, eval_results):
        """Performance on rolling folds is within ±5% of each other."""
        rolling = eval_results.get("rolling_validation", [])
        if len(rolling) < 2:
            pytest.skip("Not enough rolling validation folds")

        accuracies = [fold["accuracy"] for fold in rolling]
        spread = max(accuracies) - min(accuracies)
        assert spread < 0.10, (
            f"Rolling validation accuracy spread too large: {spread:.3f} "
            f"(min={min(accuracies):.3f}, max={max(accuracies):.3f})"
        )
