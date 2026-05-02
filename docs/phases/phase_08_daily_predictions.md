# Phase 8 — Daily Prediction System

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 7 |
| **Unlocks** | Phase 9 |

---

## Goal

Build a production-ready daily prediction pipeline: fetch today's games, compute features, run the model, and output calibrated predictions as JSON.

---

## Deliverables Checklist

- [ ] `src/models/predict.py` — prediction pipeline (single game + batch)
- [ ] `predict_today.py` (or `src/app/predict_today.py`) — daily runner
- [ ] Daily feature generation (rolling stats, injuries, news for today)
- [ ] Predictions saved to `predictions/daily/YYYY-MM-DD.json`
- [ ] Historical backtest runner
- [ ] Rule-based `top_model_factors` generation
- [ ] `make predict-today` works end-to-end
- [ ] All verification tests pass

---

## Key Implementation Details

### Daily pipeline flow

```text
1. Fetch today's scheduled games
2. For each game:
   a. Compute home team's rolling features (from latest data)
   b. Compute away team's rolling features
   c. Fetch current injury reports
   d. Collect recent news articles (if available)
   e. Build feature vectors
   f. Run through trained model
   g. Apply calibration
   h. Generate top_model_factors
3. Save predictions JSON
4. Log summary
```

### Prediction output schema

```json
{
  "date": "2026-01-15",
  "generated_at": "2026-01-15T15:00:00Z",
  "model_version": "ensemble_v1",
  "predictions": [
    {
      "game_id": "0022500451",
      "home_team_idx": 4,
      "away_team_idx": 17,
      "home_win_probability": 0.64,
      "away_win_probability": 0.36,
      "predicted_winner": "home",
      "confidence_bucket": "medium",
      "component_outputs": {
        "elo_probability": 0.58,
        "tabular_probability": 0.61,
        "sequence_probability": 0.66,
        "final_probability": 0.64
      },
      "top_model_factors": [
        "home team has stronger recent net rating",
        "away team is on short rest"
      ]
    }
  ]
}
```

### top_model_factors (rule-based MVP)

```text
Compare each feature to season average.
Flag features > 1σ from mean.
Map flagged features to human-readable templates:
  rest_diff > 1σ → "away team is on short rest"
  net_rating_diff > 1σ → "home team has stronger recent net rating"
  injury_value_missing high → "key player(s) out for away team"
  sentiment_diff notable → "home team news sentiment is positive"
```

### Backtest runner

```text
Run the full prediction pipeline on historical dates.
Compare predictions to actual outcomes.
Save results to predictions/historical_backtests/
```

---

## Verification Tests

Run: `pytest tests/test_predictions.py -v`

```python
# tests/test_predictions.py

def test_predict_single_game():
    """Prediction pipeline returns valid output for one game."""

def test_prediction_schema():
    """Output JSON matches expected schema (Pydantic validation)."""

def test_probabilities_sum_to_one():
    """home_win_prob + away_win_prob ≈ 1.0 for every prediction."""

def test_probabilities_in_range():
    """All probabilities in [0.01, 0.99]."""

def test_confidence_bucket_valid():
    """confidence_bucket is one of: low, medium, high."""

def test_component_outputs_present():
    """component_outputs has elo, tabular, sequence, final."""

def test_top_factors_generated():
    """top_model_factors is a non-empty list of strings."""

def test_daily_predictions_saved():
    """predictions/daily/YYYY-MM-DD.json is created and valid."""

def test_backtest_on_known_date():
    """Run backtest on a past date. Predictions match expected format."""

def test_backtest_accuracy_reasonable():
    """Backtest accuracy on 100 games is > 55%."""

def test_predict_today_runs():
    """make predict-today (or equivalent) completes without error."""

def test_prediction_deterministic():
    """Same game predicted twice produces identical output."""
```

**Expected: 12/12 pass.**

---

## Definition of Done

- [ ] All 12 verification tests pass
- [ ] `make predict-today` produces valid JSON
- [ ] Backtest on at least one past week produces reasonable results
- [ ] Prediction output is human-readable and debuggable

---

## Notes & Learnings

```
Daily pipeline runtime: ___ seconds per game / ___ seconds for full slate
Backtest results (sample week):
  Accuracy: ___
  Log loss: ___
  Average confidence: ___
```
