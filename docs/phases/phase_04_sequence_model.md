# Phase 4 — Team Sequence Model

| Field | Value |
|-------|-------|
| **Size** | L (1–2 weeks) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 3 |
| **Unlocks** | Phase 6, 7 |

---

## Goal

Build a neural sequence model that encodes each team's recent game history (last N=20 games) and predicts game outcomes through a matchup fusion head. This is the core ML architecture.

---

## Deliverables Checklist

- [x] `src/features/sequence_builder.py` — build padded game sequences
- [x] `src/models/team_encoder.py` — GRU encoder (shared weights)
- [x] `src/models/matchup_fusion_model.py` — fusion head + sigmoid
- [x] `src/models/train.py` — training loop with early stopping
- [x] Sequence dataset with padding and masking
- [x] Model trained and saved to `models/neural/`
- [x] Comparison against Phase 3 baselines documented
- [x] All verification tests pass (14/14)

---

## Key Implementation Details

### Sequence dataset

For each game, build two sequences (home team + away team):

```text
Input shape: [N, F_game]  where N=20, F_game=~20 features per game
Features per game step: net_rating, off_rating, def_rating, pace,
  ts_pct, efg_pct, turnover_pct, rebound_pct, point_diff, won, ...
```

### Padding and masking

```text
Team has K < 20 previous games:
  sequence = [zero_pad] * (20 - K) + [game_K, ..., game_1]
  mask     = [0] * (20 - K) + [1] * K

Team has K >= 20 previous games:
  sequence = [game_20, ..., game_1]
  mask     = [1] * 20

Optional: cross-season warm-start with previous season's last games
```

### Model architecture

```text
Home sequence → SharedTeamEncoder(GRU) → home_state [hidden_dim]
Away sequence → SharedTeamEncoder(GRU) → away_state [hidden_dim]
Context features → ContextMLP → context_state [16]

Concatenate:
  [home_state, away_state,
   home_state - away_state,
   home_state * away_state,
   context_state]

→ FusionMLP → Sigmoid → P(home_win)
```

### Training config

```text
Loss:          Binary cross-entropy
Optimizer:     AdamW (lr=1e-3, weight_decay=1e-4)
Batch size:    64
Max epochs:    100
Early stop:    patience=10 on validation log_loss
Dropout:       0.3
Hidden dim:    64
GRU layers:    2
Seed:          42
```

### Encoder options to try

```text
Start: GRU (simpler, less overfit risk)
Later: TCN or small Transformer (if GRU plateaus)
```

---

## Verification Tests

Run: `pytest tests/test_sequence_model.py -v`

```python
# tests/test_sequence_model.py

def test_sequence_builder_shapes():
    """Output sequences have shape [N_games, seq_len, F_game]."""

def test_sequence_padding_correct():
    """Early-season games are left-padded with zeros."""

def test_mask_matches_padding():
    """Mask is 0 where sequence is zero-padded, 1 elsewhere."""

def test_no_leakage_in_sequences():
    """For game on date D, all games in sequence are before D."""

def test_team_encoder_output_shape():
    """TeamEncoder(batch) returns [batch_size, hidden_dim]."""

def test_shared_weights():
    """Home and away encoders share the same parameters (same id)."""

def test_fusion_model_output_shape():
    """FusionModel returns [batch_size, 1] probabilities."""

def test_probabilities_valid():
    """All outputs are in (0, 1)."""

def test_model_trains_one_epoch():
    """Training loop completes one epoch without error."""

def test_loss_decreases():
    """Training loss after 5 epochs < training loss at epoch 1."""

def test_model_saves_and_loads():
    """Model can be saved to disk and reloaded with same predictions."""

def test_model_beats_coin_flip():
    """Sequence model accuracy > 55% on validation set."""

def test_gradients_flow():
    """No NaN gradients after a forward-backward pass."""

def test_predictions_deterministic():
    """Same input produces same output (seeded)."""
```

**Expected: 14/14 pass.**

---

## Definition of Done

- [x] All 14 verification tests pass
- [x] Model trained with best checkpoint saved
- [x] Comparison table: sequence model vs all baselines (log_loss, brier, accuracy)
- [x] `make train-model` runs end-to-end
- [x] Sequence model beats XGBoost: 65.1% vs 64.5% accuracy, 0.619 vs 0.622 log loss

---

## Notes & Learnings

```
Sequence model:     accuracy=0.651  log_loss=0.6188
vs XGBoost:         Δ accuracy=+0.6%  Δ log_loss=-0.003 (better)
Architecture used:  GRU (2 layers, hidden=64, dropout=0.3)
Parameters:         99,377
Training:           22 epochs, early stop at 12 (patience=10)
Best val_loss:      0.6503

Notes:
- GRU beats all baselines on first try
- CPU training took ~8 minutes for 22 epochs
- Skipped exploratory notebook (03_sequence_dataset_debug.ipynb) for now
- 73/73 total tests passing
```
