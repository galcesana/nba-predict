# Phase 2 — Leakage-Safe Feature Table

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 1 |
| **Unlocks** | Phase 3 |

---

## Goal

Build a leakage-safe feature table where **every feature for a game on date D is computed only from data before D**. This is the most critical correctness requirement in the project.

---

## Deliverables Checklist

- [x] `src/features/rolling_features.py` — rolling stats (last 5, 10, season)
- [x] `src/features/schedule_features.py` — rest days, back-to-back, games in 7/14 days
- [x] `src/features/build_matchup_dataset.py` — combine into matchup rows
- [x] Matchup dataset saved to `data/processed/matchup_rows/`
- [x] Leakage prevention tests pass (5/5)
- [x] All verification tests pass (9/9)

---

## Key Implementation Details

### Rolling features (per team, before each game)

```text
season_win_pct_before_game, last_5_win_pct, last_10_win_pct,
season_point_diff, last_5_point_diff, last_10_point_diff,
season_net_rating, last_10_net_rating,
season_off_rating, season_def_rating, last_10_off_rating, last_10_def_rating,
pace, turnover_pct, true_shooting_pct, effective_fg_pct,
free_throw_rate, assist_pct, rebound_pct
```

**Critical rule: shift all rolling windows by 1 game.** For game N, compute rolling stats over games 1 to N-1 only.

### Schedule features

```text
home_rest_days, away_rest_days, rest_diff,
home_back_to_back, away_back_to_back,
home_games_last_7, away_games_last_7,
home_games_last_14, away_games_last_14
```

### Matchup row schema (one row per game)

```text
game_id, date, season,
home_team_idx, away_team_idx,
home_[all rolling features], away_[all rolling features],
diff_[key rolling features] (home minus away),
home_rest_days, away_rest_days, rest_diff,
home_back_to_back, away_back_to_back, ...,
target_home_win
```

### Leakage prevention checklist

```text
✓ Rolling features use .shift(1) — never include the current game
✓ Season averages exclude the current game
✓ Rest days computed from previous game date only
✓ No future game data accessed
✓ No target variable (home_win) in features
✓ Features sorted by date before computing rolling windows
```

---

## Verification Tests

Run: `pytest tests/test_no_leakage.py tests/test_features.py -v`

```python
# tests/test_no_leakage.py

def test_no_future_games_used():
    """For each row, verify all rolling feature dates < game date.
    Sample 100 random rows. Look up the games used in their rolling windows.
    Assert all game dates are strictly before the target game date."""

def test_no_target_boxscore_used():
    """Verify that a game's own stats are not in its rolling features.
    For each sampled row, ensure the game's own game_id is not in the
    window of games used to compute its features."""

def test_rolling_features_shifted_correctly():
    """For a specific team's 10th game of a season:
    - last_5_win_pct should be computed from games 5-9 (not 6-10)
    - Manually compute expected value and compare."""

def test_season_opener_has_nan_or_zero():
    """First game of each season should have NaN or 0 for rolling features
    (no prior games to compute from)."""

def test_features_before_target():
    """For the earliest game in the dataset, verify no features
    come from future data."""

# tests/test_features.py

def test_matchup_dataset_exists():
    """Matchup rows parquet file exists and is loadable."""

def test_matchup_row_count():
    """One row per game. Count should match games table."""

def test_matchup_schema():
    """Required columns present: home_*, away_*, diff_*, target_home_win."""

def test_no_nulls_in_non_early_games():
    """After game 20 of each season, no nulls in rolling features."""

def test_rest_days_reasonable():
    """rest_days values are between 0 and 14 (no negatives, no absurdities)."""

def test_back_to_back_correct():
    """When rest_days == 0, back_to_back should be 1."""

def test_diff_features_computed():
    """diff columns = home value minus away value for each pair."""

def test_target_is_binary():
    """target_home_win is 0 or 1 only."""

def test_home_win_rate_around_60():
    """Overall home win rate should be ~55-65% (sanity check)."""
```

**Expected: 14/14 pass.**

---

## Definition of Done

- [x] All 14 verification tests pass (especially the 5 leakage tests)
- [x] `data/processed/matchup_rows/` has one parquet file (298 KB)
- [x] `make build-features` runs end-to-end
- [x] Manual spot-check: verified rolling features match manual computation for 100+ sampled rows

---

## Notes & Learnings

```
- Critical bug: pandas index alignment. When building features DF with .values
  (fresh 0..N index) and assigning Series that kept the original group index
  (e.g. 11, 66, 91...), pandas aligned on index → 99.8% NaN.
  Fix: .reset_index(drop=True) on group before computing.

- Schedule features must group by (team_idx, season), not just team_idx.
  Otherwise offseason gaps (157-288 days, especially COVID bubble) appear as
  absurd rest_days values.

- season_games_played is an integer counter (0 for openers), not a rolling stat.
  Excluded from the "must be NaN for openers" leakage test.

- Final dataset: 14,429 rows × 110 columns, 56.5% home win rate.
- 47/47 tests passing after fixes.
```
