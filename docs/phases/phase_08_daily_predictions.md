# Phase 8 — Daily Prediction System

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 7 |
| **Unlocks** | Phase 9 |

---

## Goal

Build a production-ready daily prediction pipeline: fetch today's games, compute features, run the model, and output calibrated predictions as JSON.

---

## Deliverables Checklist

- [x] `src/models/predict.py` — prediction pipeline (single game + batch)
- [x] `predict_today.py` (or `src/app/predict_today.py`) — daily runner
- [x] Daily feature generation (rolling stats, injuries, news for today)
- [x] Predictions saved to `predictions/daily/YYYY-MM-DD.json`
- [x] Historical backtest runner
- [x] Rule-based `top_model_factors` generation
- [x] `make predict-today` works end-to-end
- [x] All verification tests pass

---

## Key Implementation Details

### Daily pipeline flow

```text
1. Fetch today's scheduled games
2. For each game:
   a. Compute home team's rolling/context features from pre-tip-off history
   b. Compute away team's rolling/context features
   c. Reuse processed injury/news features when the game already exists historically
   d. Otherwise build proxy or live injury/news vectors for the target slate
   e. Run base models and the calibrated ensemble
   f. Generate top_model_factors
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
    """Backtest runner saves a valid report."""

def test_backtest_accuracy_reasonable():
    """Backtest report metrics are deterministic in script coverage."""

def test_predict_today_runs():
    """Daily runner completes without hitting the live API in CI."""

def test_prediction_deterministic():
    """Same game predicted twice produces identical output."""
```

**Expected: 12/12 pass.**

---

## Definition of Done

- [x] All 12 verification tests pass
- [x] `make predict-today` produces valid JSON
- [x] Backtest on a verified historical date range produces reasonable results
- [x] Prediction output is human-readable and debuggable

---

## Notes & Learnings

```
Verification runs completed on 2026-05-16:
  Historical daily run:
    python -m src.app.predict_today --date 2024-01-15
    Saved 11-game slate to predictions/daily/2024-01-15.json

  Historical backtest:
    python -m src.app.run_backtest --start-date 2024-01-15 --end-date 2024-01-16
    Accuracy: 0.714
    Log loss: 0.6318
    Total games: 14

Implementation notes:
  - make predict-today now targets src.app.predict_today
  - ScoreboardV3 is used first for schedule fetching, with ScoreboardV2 fallback
  - Inference reuses processed injury/news features for historical games and
    builds missing target-slate rows on the fly
  - News still degrades gracefully to the Phase 6 zero-vector path when no raw
    pregame news data is available
```
