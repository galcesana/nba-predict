# Phase 3 — Baseline Models

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 2 |
| **Unlocks** | Phase 4, 5, 6 |

---

## Goal

Train and evaluate baseline models to establish minimum performance benchmarks. All future models must beat these. Baselines also serve as data-leakage detectors — if a model massively outperforms baselines, suspect leakage first.

---

## Deliverables Checklist

- [x] `src/models/elo.py` — Elo rating system
- [x] `src/models/tabular_model.py` — Logistic regression + XGBoost
- [x] `src/models/evaluate.py` — evaluation script with all metrics
- [x] `src/models/calibrate.py` — Platt scaling / isotonic regression
- [x] Time-based train/val/test splits implemented
- [x] Evaluation results saved to `models/baselines/`
- [x] Calibration plots generated
- [x] All verification tests pass (12/12)

---

## Key Implementation Details

### Models to train

| Model | Description |
|-------|-------------|
| **Home-team baseline** | Always predict home team wins (~58-60% accuracy) |
| **Elo model** | Standard Elo with K=20, home advantage=100 |
| **Logistic regression** | On MVP feature set (~35 features) |
| **XGBoost** | On MVP feature set, tuned with early stopping |

### Time-based splits

```text
Train:      2014-15 through 2021-22
Validation: 2022-23
Test:       2023-24 and 2024-25
```

Also implement rolling validation:
```text
Train 2014-2019 → validate 2020
Train 2014-2020 → validate 2021
Train 2014-2021 → validate 2022
Train 2014-2022 → validate 2023
```

### Metrics to compute

```text
Primary:    log_loss, brier_score, calibration_error
Secondary:  accuracy, roc_auc
Subset:     performance_by_confidence_bucket, performance_on_close_games
```

### Calibration

```text
After training each model:
1. Apply Platt scaling (logistic calibration) on validation set
2. Plot calibration curves (predicted prob vs actual win rate)
3. Store calibrated model alongside raw model
```

### Expected baseline performance (rough targets)

```text
Home-team baseline:  ~58% accuracy, ~0.69 log loss
Elo:                 ~62% accuracy, ~0.65 log loss
Logistic regression: ~63% accuracy, ~0.64 log loss
XGBoost:             ~64-66% accuracy, ~0.62 log loss
```

---

## Verification Tests

Run: `pytest tests/test_baselines.py -v`

```python
# tests/test_baselines.py

def test_elo_model_trains():
    """Elo model runs on training data without error."""

def test_elo_ratings_reasonable():
    """All Elo ratings are between 1000 and 2000 (no explosion)."""

def test_elo_beats_coin_flip():
    """Elo accuracy > 50% on validation set."""

def test_logistic_regression_trains():
    """Logistic regression fits on training features without error."""

def test_xgboost_trains():
    """XGBoost fits with early stopping on validation set."""

def test_xgboost_beats_home_baseline():
    """XGBoost accuracy > home-team baseline accuracy."""

def test_probabilities_sum_to_one():
    """For all models: P(home_win) + P(away_win) ≈ 1.0."""

def test_probabilities_in_range():
    """All predicted probabilities are in [0.01, 0.99]."""

def test_calibration_plot_generated():
    """Calibration plot image saved to models/baselines/."""

def test_evaluation_metrics_saved():
    """Evaluation results JSON exists with log_loss, brier, accuracy."""

def test_no_leakage_signal():
    """No model exceeds 72% accuracy (would suggest leakage)."""

def test_rolling_validation_consistent():
    """Performance on rolling folds is within ±5% of each other."""
```

**Expected: 12/12 pass.**

---

## Definition of Done

- [x] All 12 verification tests pass
- [x] All 4 models trained and saved
- [x] Metrics JSON and calibration plots in `models/baselines/`
- [x] `make train-baseline` and `make evaluate` run end-to-end
- [x] Baseline metrics documented in Notes below for future comparison

---

## Notes & Learnings

```
Record baseline metrics here for comparison in later phases:

TEST SET RESULTS (2023-24 + 2024-25, 2455 games):
Home baseline:       accuracy=0.544  log_loss=0.6907  brier=0.2488  roc_auc=0.500  ECE=0.0266
Elo:                 accuracy=0.637  log_loss=0.6286  brier=0.2198  roc_auc=0.714  ECE=0.0782
Logistic regression: accuracy=0.642  log_loss=0.6260  brier=0.2172  roc_auc=0.706  ECE=0.0253
XGBoost:             accuracy=0.645  log_loss=0.6217  brier=0.2162  roc_auc=0.710  ECE=0.0313
XGBoost (calibrated): accuracy=0.646  log_loss=0.6316  brier=0.2206  roc_auc=0.710  ECE=0.0599

ROLLING VALIDATION (XGBoost):
2019-20: acc=0.622  2020-21: acc=0.608  2021-22: acc=0.606
2022-23: acc=0.612  2023-24: acc=0.627  2024-25: acc=0.638

Notes:
- Skipped exploratory notebook (02_elo_baseline.ipynb) for now
- Used XGBoost only (LightGBM not installed), performance meets targets
- XGBoost early stopped at iteration 54/500
- Platt calibration slightly increased log loss on test set (overfitting on small val)
- Rolling validation shows stable ~0.61-0.64 accuracy across folds (spread < 5%)
- 59/59 total tests passing
```
