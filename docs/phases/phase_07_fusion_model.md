# Phase 7 — Full Fusion Model

| Field | Value |
|-------|-------|
| **Size** | L (1–2 weeks) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 4 + 5 + 6 |
| **Unlocks** | Phase 8 |

---

## Goal

Combine all encoder streams (performance, injury, news, context) into the full multi-encoder fusion model. Train, calibrate, and prove it beats baselines without becoming miscalibrated.

---

## Deliverables Checklist

- [ ] `src/models/injury_encoder.py` — shared injury MLP encoder
- [ ] `src/models/news_encoder.py` — shared news MLP encoder
- [ ] Updated `src/models/matchup_fusion_model.py` — full 4-stream fusion
- [ ] `src/models/ensemble.py` — logistic regression meta-model
- [ ] `src/models/calibrate.py` — temperature/Platt/isotonic calibration
- [ ] Full fusion model trained and saved
- [ ] Ensemble trained and saved
- [ ] Comprehensive ablation study completed
- [ ] Calibration plots for all model variants
- [ ] All verification tests pass

---

## Key Implementation Details

### Full fusion architecture

```text
Home sequence   → SharedTeamEncoder → home_perf_state
Away sequence   → SharedTeamEncoder → away_perf_state

Home injury vec → SharedInjuryEncoder → home_injury_state
Away injury vec → SharedInjuryEncoder → away_injury_state

Home news vec   → SharedNewsEncoder → home_news_state
Away news vec   → SharedNewsEncoder → away_news_state

Context vec     → ContextMLP → context_state

Concatenate:
  [home_perf, away_perf, perf_diff, perf_product,
   home_injury, away_injury, injury_diff,
   home_news, away_news, news_diff,
   context, news_available_flag]

→ FusionMLP → Sigmoid → P(home_win)
```

### Ablation study (train all 6 variants)

```text
Model A: stats only (sequence encoder + context)
Model B: stats + schedule
Model C: stats + schedule + injuries
Model D: stats + schedule + news
Model E: stats + schedule + injuries + news (full fusion)
Model F: ensemble (Elo + XGBoost + Model E)
```

### Ensemble meta-model

```text
Input features:  elo_prob, xgboost_prob, sequence_prob,
                 injury_summary, news_summary, model_confidence
Model:           Logistic regression (fit on validation set)
Output:          Calibrated final probability
```

### Calibration

```text
1. Temperature scaling on full fusion model
2. Platt scaling on ensemble output
3. Verify: predicted 60% → actual ~60% win rate
```

---

## Verification Tests

Run: `pytest tests/test_fusion_model.py -v`

```python
# tests/test_fusion_model.py

def test_injury_encoder_shape():
    """InjuryEncoder output: [batch, injury_hidden_dim]."""

def test_news_encoder_shape():
    """NewsEncoder output: [batch, news_hidden_dim]."""

def test_news_encoder_handles_zeros():
    """Zero news vector (no-news case) produces valid output."""

def test_fusion_model_forward():
    """Full fusion model runs forward pass without error."""

def test_fusion_output_shape():
    """Output is [batch_size, 1] probabilities in (0, 1)."""

def test_ablation_all_variants_train():
    """All 6 ablation variants train without error."""

def test_full_model_beats_baselines():
    """Model E log_loss < XGBoost baseline log_loss."""

def test_ensemble_beats_best_single():
    """Ensemble log_loss <= best single model log_loss."""

def test_calibration_applied():
    """Post-calibration model exists and produces valid probabilities."""

def test_calibration_quality():
    """Expected calibration error < 0.05 on validation set."""

def test_no_nan_in_predictions():
    """No NaN values in any model's predictions."""

def test_ablation_results_saved():
    """Ablation results JSON exists with all 6 variants' metrics."""
```

**Expected: 12/12 pass.**

---

## Definition of Done

- [ ] All 12 verification tests pass
- [ ] All 6 ablation variants trained and compared
- [ ] Ensemble model trained and calibrated
- [ ] Best model saved to `models/ensembles/`
- [ ] Ablation results table documented in Notes

---

## Notes & Learnings

```
Ablation results:
  Model A (stats only):          log_loss=___  brier=___  accuracy=___
  Model B (+ schedule):          log_loss=___  brier=___  accuracy=___
  Model C (+ injuries):          log_loss=___  brier=___  accuracy=___
  Model D (+ news):              log_loss=___  brier=___  accuracy=___
  Model E (full fusion):         log_loss=___  brier=___  accuracy=___
  Model F (ensemble):            log_loss=___  brier=___  accuracy=___

Does news help?      yes/no  (Δ log_loss=___)
Does injuries help?  yes/no  (Δ log_loss=___)
Best single model:   ___
Best overall:        ___
Calibration error:   ___
```
