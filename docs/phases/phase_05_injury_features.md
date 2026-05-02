# Phase 5 — Injury Features

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 3 |
| **Unlocks** | Phase 7 |

---

## Goal

Ingest structured injury data, compute the basketball impact of missing/limited players, and build an injury feature vector for each team per game. Prove via ablation whether injury data improves prediction.

---

## Deliverables Checklist

- [ ] `src/data/fetch_injuries.py` — injury report collection
- [ ] Injury data cached in `data/raw/injuries/`
- [ ] Player value estimation (minutes, usage, plus-minus)
- [ ] `src/features/injury_features.py` — team injury vector builder
- [ ] Injury features saved to `data/processed/injury_features/`
- [ ] Ablation: model with injuries vs model without injuries
- [ ] All verification tests pass

---

## Key Implementation Details

### Injury data sources

```text
Primary: Official NBA Injury Reports (via nba_api or scraping)
Fallback: ESPN injuries page, CBS Sports, RotoWire
```

### Injury feature vector (per team per game)

```text
players_out_count
players_questionable_count
starter_out_count
top_3_usage_players_out
minutes_missing            (sum of avg MPG of out players)
usage_missing              (sum of usage rates of out players)
estimated_value_missing    (minutes × usage × box_plus_minus)
```

### Player value estimation

```text
player_value = recent_minutes_per_game × usage_rate × estimated_plus_minus

Use rolling 10-game averages for player stats.
Only consider players assigned to the team on game date (trade-aware).
```

### Handling missing injury data

```text
If no injury report available for a game:
  Set all injury features to 0 (assume full roster)
  Set injury_data_available = 0 flag
  The model learns to handle missing injury info gracefully
```

---

## Verification Tests

Run: `pytest tests/test_injury_features.py -v`

```python
# tests/test_injury_features.py

def test_injury_features_exist():
    """Injury features parquet file exists and is loadable."""

def test_injury_features_schema():
    """Required columns: players_out_count, starter_out_count,
    minutes_missing, usage_missing, estimated_value_missing."""

def test_injury_features_non_negative():
    """All injury feature values are >= 0."""

def test_players_out_count_reasonable():
    """players_out_count is between 0 and 15 (no team has >15 out)."""

def test_minutes_missing_reasonable():
    """minutes_missing is between 0 and 240 (max 5 starters × 48 min)."""

def test_trade_awareness():
    """A player traded mid-season: their injury only counts for
    the team they belong to on the game date."""

def test_injury_available_flag():
    """Games with no injury data have injury_data_available=0."""

def test_ablation_with_injuries():
    """Train XGBoost with and without injury features.
    Log both log_loss values for comparison."""

def test_injury_features_before_game():
    """Injury report timestamp is before game start time."""

def test_one_row_per_team_per_game():
    """Each (game_id, team_idx) pair appears exactly once."""
```

**Expected: 10/10 pass.**

---

## Definition of Done

- [ ] All 10 verification tests pass
- [ ] Injury features saved to `data/processed/injury_features/`
- [ ] Ablation results documented in Notes below
- [ ] Decision made: do injury features help? By how much?

---

## Notes & Learnings

```
Ablation results:
  XGBoost without injuries: log_loss=___  brier=___
  XGBoost with injuries:    log_loss=___  brier=___
  Improvement:              Δ log_loss=___

Injury data coverage: ___% of games have injury reports
Best injury feature (by SHAP importance): ___
```
