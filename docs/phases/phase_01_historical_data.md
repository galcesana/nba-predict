# Phase 1 — Historical Data Foundation

| Field | Value |
|-------|-------|
| **Size** | M (3–5 days) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 0 |
| **Unlocks** | Phase 2 |

---

## Goal

Fetch historical NBA game data (2014-15 through current season), normalize team IDs, build the core game table and team game logs, and cache all raw API responses as Parquet snapshots.

---

## Deliverables Checklist

- [ ] `NbaApiProvider.fetch_games()` implemented
- [ ] `NbaApiProvider.fetch_team_game_logs()` implemented
- [ ] `NbaApiProvider.fetch_box_scores()` implemented
- [ ] `NbaApiProvider.fetch_player_info()` implemented
- [ ] Raw API responses cached as Parquet in `data/raw/nba_api/`
- [ ] `src/data/fetch_games.py` — CLI script to fetch all seasons
- [ ] `src/data/fetch_boxscores.py` — CLI script to fetch box scores
- [ ] `src/data/clean_data.py` — normalize columns, map team names to idx
- [ ] `src/anonymization/team_mapping.py` — team name ↔ idx resolution
- [ ] `src/anonymization/player_mapping.py` — player name ↔ idx resolution
- [ ] `player_to_idx.json` generated and saved
- [ ] `player_team_assignments` table built (handles trades)
- [ ] Clean game table saved to `data/processed/`
- [ ] Team game logs saved to `data/processed/team_game_logs/`
- [ ] All verification tests pass

---

## Key Implementation Details

### Game table schema

```text
game_id, date, season, home_team_idx, away_team_idx,
home_score, away_score, home_win, arena, start_time_utc
```

### Team game log schema (one row per team per game)

```text
game_id, date, season, team_idx, opponent_team_idx, is_home, won,
points_for, points_against, point_diff,
off_rating, def_rating, net_rating, pace,
efg_pct, ts_pct, turnover_pct, off_rebound_pct, def_rebound_pct,
free_throw_rate, assist_pct, steal_pct, block_pct
```

### Player-team assignment table

```text
player_id, team_idx, start_date, end_date, source
```

Query pattern: `start_date <= game_date AND (end_date IS NULL OR end_date >= game_date)`

### Data fetching strategy

```text
1. Fetch season schedules one season at a time
2. Sleep 1 second between API calls
3. Cache each season's raw response as Parquet before processing
4. Validate response schemas — abort if structure changes unexpectedly
5. Log progress (season X/Y, games fetched: N)
6. If a cached Parquet exists for a season, skip re-fetching
```

### Seasons to fetch

```text
2014-15, 2015-16, 2016-17, 2017-18, 2018-19,
2019-20, 2020-21, 2021-22, 2022-23, 2023-24,
2024-25, 2025-26 (current/partial)
```

---

## New Files

| File | Purpose |
|------|---------|
| `src/data/fetch_games.py` | Download game schedules and results per season |
| `src/data/fetch_boxscores.py` | Download team/player box scores per game |
| `src/data/clean_data.py` | Normalize, map to anonymous IDs, validate |
| `src/anonymization/team_mapping.py` | Team name → idx lookup and reverse |
| `src/anonymization/player_mapping.py` | Player name → idx lookup and reverse |
| `src/features/build_team_game_logs.py` | Build one-row-per-team-per-game table |

---

## Verification Tests

Run: `pytest tests/test_data_foundation.py -v`

```python
# tests/test_data_foundation.py

def test_games_table_exists():
    """Processed games table saved and loadable."""

def test_games_table_schema():
    """Games table has required columns: game_id, date, season,
    home_team_idx, away_team_idx, home_score, away_score, home_win."""

def test_games_table_no_nulls():
    """No null values in critical columns."""

def test_team_indices_valid():
    """All team_idx values in games table are in range 0-29."""

def test_home_win_binary():
    """home_win column contains only 0 and 1."""

def test_team_game_logs_exist():
    """Team game logs table saved and loadable."""

def test_team_game_logs_two_rows_per_game():
    """Each game_id appears exactly twice (home + away)."""

def test_team_game_logs_schema():
    """Team game logs have all required stat columns."""

def test_season_coverage():
    """At least 10 seasons of data present."""

def test_games_per_season():
    """Each full season has ~1,230 games (±50 for lockout/bubble)."""

def test_player_team_assignments():
    """Player-team table exists. Traded players have multiple rows."""

def test_raw_cache_exists():
    """Parquet cache files exist in data/raw/nba_api/."""

def test_no_duplicate_games():
    """No duplicate game_id values in games table."""
```

**Expected: 13/13 pass.**

---

## Definition of Done

- [ ] All 13 verification tests pass
- [ ] `data/processed/games.parquet` exists with ~12,000+ rows
- [ ] `data/processed/team_game_logs/` exists with ~24,000+ rows
- [ ] Raw cache in `data/raw/nba_api/` has one file per season
- [ ] `make fetch-data` runs end-to-end without errors

---

## Notes & Learnings

```
(fill in during implementation)
```
