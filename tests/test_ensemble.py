"""Tests for the Ensemble meta-model.

Run with: pytest tests/test_ensemble.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import train_ensemble


@pytest.fixture
def mock_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create mock val and test predictions."""
    np.random.seed(42)
    
    def _create_df(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "game_id": np.arange(n),
            "actual_home_win": np.random.randint(0, 2, n),
            "neural_prob": np.random.uniform(0.3, 0.7, n),
            "xgboost_prob": np.random.uniform(0.3, 0.7, n),
            "elo_prob": np.random.uniform(0.3, 0.7, n),
        })

    return _create_df(200), _create_df(100)


class TestEnsembleModel:
    def test_train_ensemble(self, mock_predictions):
        """Ensemble trains and produces valid probabilities."""
        val_preds, test_preds = mock_predictions
        
        results = train_ensemble(val_preds, test_preds)
        
        assert "meta_model" in results
        assert "test_predictions" in results
        assert "metrics" in results
        
        preds = results["test_predictions"]
        assert len(preds) == len(test_preds)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_missing_models(self, mock_predictions):
        """Ensemble handles missing base models gracefully (if at least 1 exists)."""
        val_preds, test_preds = mock_predictions
        
        # Drop xgboost
        val_partial = val_preds.drop(columns=["xgboost_prob"])
        test_partial = test_preds.drop(columns=["xgboost_prob"])
        
        results = train_ensemble(val_partial, test_partial)
        
        assert "xgboost_prob" not in results["metrics"]["model_weights"]
        assert len(results["prob_cols"]) == 2
        
    def test_no_models_raises_error(self, mock_predictions):
        """Ensemble raises error if no probability columns exist."""
        val_preds, test_preds = mock_predictions
        
        val_empty = val_preds[["game_id", "actual_home_win"]]
        test_empty = test_preds[["game_id", "actual_home_win"]]
        
        with pytest.raises(ValueError, match="No model predictions available"):
            train_ensemble(val_empty, test_empty)
