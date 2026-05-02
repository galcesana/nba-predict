# Phase 3 — Baseline Models

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 2 |
| **Unlocks** | Phase 4, 5, 6 |

---

## Goal

Train and evaluate baseline models to establish minimum performance benchmarks. All future models must beat these. Baselines also serve as data-leakage detectors — if a model massively outperforms baselines, suspect leakage first.

---

## Deliverables Checklist

- [ ] `src/models/elo.py` — Elo rating system
- [ ] `src/models/tabular_model.py` — Logistic regression + XGBoost/LightGBM
- [ ] `src/models/evaluate.py` — evaluation script with all metrics
- [ ] `src/models/calibrate.py` — Platt scaling / isotonic regression
- [ ] Time-based train/val/test splits implemented
- [ ] Evaluation results saved to `models/baselines/`
- [ ] Calibration plots generated
- [ ] `notebooks/02_elo_baseline.ipynb` — exploratory analysis
- [ ] All verification tests pass

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

- [ ] All 12 verification tests pass
- [ ] All 4 models trained and saved
- [ ] Metrics JSON and calibration plots in `models/baselines/`
- [ ] `make train-baseline` and `make evaluate` run end-to-end
- [ ] Baseline metrics documented in Notes below for future comparison

---

## Notes & Learnings

```
Record baseline metrics here for comparison in later phases:

Home baseline:      accuracy=___  log_loss=___  brier=___
Elo:                accuracy=___  log_loss=___  brier=___
Logistic regression: accuracy=___  log_loss=___  brier=___
XGBoost:            accuracy=___  log_loss=___  brier=___
```
