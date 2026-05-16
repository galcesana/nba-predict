# Phase 10 - Live Publishing Layer

| Field | Value |
|-------|-------|
| **Size** | M (2-4 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 9 |
| **Unlocks** | Production-friendly daily forecast delivery |

---

## Goal

Add a tracked published forecast path that Streamlit Cloud can read directly from `main`, automate daily publication with GitHub Actions, and expose publish status/staleness to the dashboard.

---

## Deliverables Checklist

- [x] `src/app/publish_today.py` - publish CLI with manifest output
- [x] `published/daily/YYYY-MM-DD.json` + `published/daily/latest.json`
- [x] `published/manifest.json`
- [x] Dashboard source precedence: local -> published -> bundled
- [x] Manifest-driven status banner for published, no-games, stale, and bundled states
- [x] `.github/workflows/publish_daily.yml` scheduled automation
- [x] Minimal tracked inference bundle committed for clean-checkout publishing
- [x] Tracking docs updated for Phase 10
- [x] Verification tests pass

---

## Key Implementation Details

### Publish flow

```text
python -m src.app.publish_today [--date YYYY-MM-DD] [--timezone America/New_York]

1. Resolve target date in America/New_York when --date is omitted
2. Run the existing prediction pipeline into a temporary directory
3. On success, atomically publish:
   - published/daily/YYYY-MM-DD.json
   - published/daily/latest.json
   - published/manifest.json
4. On no-games days, update only published/manifest.json
5. On failure, leave existing published files untouched and exit non-zero
```

### Manifest contract

```json
{
  "status": "published | no_games",
  "target_date": "YYYY-MM-DD",
  "attempted_at": "UTC ISO timestamp",
  "latest_available_date": "YYYY-MM-DD | null",
  "published_file": "published/daily/YYYY-MM-DD.json | null",
  "games_count": 0,
  "model_version": "ensemble_v1 | null"
}
```

### Dashboard data precedence

```text
1. predictions/daily/*.json     (local development preview)
2. published/daily/*.json       (tracked deployment forecasts)
3. src/app/bundled_data/*.json  (last-resort fallback)
```

### Tracked inference bundle

```text
data/processed/games.parquet
data/processed/team_game_logs/team_game_logs.parquet
data/processed/injury_features/injury_features.parquet
data/processed/news_features/news_features.parquet
models/baselines/scaler.joblib
models/neural/best_model.pt
```

### GitHub Actions automation

```text
Workflow: .github/workflows/publish_daily.yml
Triggers:
  - schedule: 5 15 * * * UTC
  - workflow_dispatch
Behavior:
  - install dependencies
  - run python -m src.app.publish_today
  - commit changed files in published/
  - push back to main
```

---

## Verification Tests

Run:

```bash
pytest tests/test_publishing.py tests/test_dashboard.py -q
pytest tests -q
python -m src.app.publish_today --date 2024-01-15
```

Automated coverage added:

- publish success writes dated file, `latest.json`, and manifest
- no-games publish preserves `latest.json` and updates manifest only
- explicit date handling is deterministic
- default date resolves in `America/New_York`
- failures do not overwrite existing published files
- dashboard source precedence covers local, published, and bundled
- dashboard archive reads from `published/daily`
- dashboard status copy reflects published, no-games, stale, and bundled states

**Actual:**

- [x] `pytest tests/test_publishing.py tests/test_dashboard.py -q` -> 18 passed
- [x] `pytest tests -q` -> 129 passed
- [x] `python -m src.app.publish_today --date 2024-01-15` -> published 11-game slate
- [ ] Remote `workflow_dispatch` not executed from the local environment

---

## Definition of Done

- [x] Published forecast artifacts are tracked under `published/`
- [x] Clean-checkout publishing dependencies are tracked in git
- [x] Dashboard uses published forecasts before bundled fallback
- [x] Status banner reflects manifest freshness/no-games state
- [x] Scheduled GitHub Actions workflow is present
- [x] All automated verification checks pass

---

## Notes & Learnings

```text
Publishing is intentionally separate from training and feature rebuilding.
The scheduled job uses the committed inference bundle plus live schedule lookup.
The initial tracked publish was generated for 2024-01-15 to seed the archive.

Verification completed on 2026-05-16:
  - pytest tests/test_publishing.py tests/test_dashboard.py -q -> 18 passed
  - pytest tests -q -> 129 passed
  - python -m src.app.publish_today --date 2024-01-15 -> published/daily/2024-01-15.json
```
