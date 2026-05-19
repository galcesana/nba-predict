"""Calibration utilities: Platt scaling and isotonic regression.

Applies post-hoc calibration to model predictions so that
predicted 60% games actually win ~60% of the time.

Usage:
    python -m src.models.calibrate
"""

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt scaling (sigmoid) calibration.

    Fits a logistic regression on (raw_score, label) pairs to produce
    calibrated probabilities.
    """

    def __init__(self):
        self._model = LogisticRegression(max_iter=1000)

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        """Fit the calibrator on validation predictions and labels."""
        raw_probs = np.asarray(raw_probs).reshape(-1, 1)
        labels = np.asarray(labels)
        self._model.fit(raw_probs, labels)
        logger.info("Platt calibrator fitted")
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        raw_probs = np.asarray(raw_probs).reshape(-1, 1)
        return self._model.predict_proba(raw_probs)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibration.

    Non-parametric calibration that preserves monotonicity.
    """

    def __init__(self):
        self._model = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """Fit the calibrator on validation predictions and labels."""
        self._model.fit(raw_probs, labels)
        logger.info("Isotonic calibrator fitted")
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        return self._model.predict(np.asarray(raw_probs))


def calibrate_predictions(
    raw_probs: np.ndarray,
    labels: np.ndarray,
    method: str = "platt",
) -> PlattCalibrator | IsotonicCalibrator:
    """Fit a calibrator on (predictions, labels) pairs.

    Args:
        raw_probs: Uncalibrated predicted probabilities.
        labels: Actual binary labels.
        method: "platt" or "isotonic".

    Returns:
        Fitted calibrator.
    """
    if method == "platt":
        cal = PlattCalibrator()
    elif method == "isotonic":
        cal = IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    cal.fit(raw_probs, labels)
    return cal
