# Phase 4 — Team Sequence Model

| Field | Value |
|-------|-------|
| **Size** | L (1–2 weeks) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 3 |
| **Unlocks** | Phase 6, 7 |

---

## Goal

Build a neural sequence model that encodes each team's recent game history (last N=20 games) and predicts game outcomes through a matchup fusion head. This is the core ML architecture.

---

## Deliverables Checklist

- [ ] `src/features/sequence_builder.py` — build padded game sequences
- [ ] `src/models/team_encoder.py` — GRU/TCN encoder (shared weights)
- [ ] `src/models/matchup_fusion_model.py` — fusion head + sigmoid
- [ ] `src/models/train.py` — training loop with early stopping
- [ ] Sequence dataset with padding and masking
- [ ] Model trained and saved to `models/neural/`
- [ ] Comparison against Phase 3 baselines documented
- [ ] `notebooks/03_sequence_dataset_debug.ipynb`
- [ ] All verification tests pass

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

- [ ] All 14 verification tests pass
- [ ] Model trained with best checkpoint saved
- [ ] Comparison table: sequence model vs all baselines (log_loss, brier, accuracy)
- [ ] `make train-model` runs end-to-end
- [ ] If sequence model does not beat XGBoost yet, that is OK — architecture correctness is the goal

---

## Notes & Learnings

```
Sequence model:     accuracy=___  log_loss=___  brier=___
vs XGBoost:         Δ accuracy=___  Δ log_loss=___
Architecture used:  GRU / TCN / Transformer
Notes:
```
