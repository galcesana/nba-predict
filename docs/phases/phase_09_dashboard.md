# Phase 9 - Product Dashboard

| Field | Value |
|-------|-------|
| **Size** | L (1-2 weeks) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 8 |
| **Unlocks** | Phase 10 |

---

## Goal

Build a Streamlit dashboard that presents daily predictions, model performance, calibration charts, and debugging views for news sentiment. This is the user-facing product layer.

---

## Deliverables Checklist

- [x] `src/app/streamlit_app.py` - main dashboard application
- [x] Today's predictions page
- [x] Historical prediction archive page
- [x] Model performance / backtest page
- [x] Calibration charts page
- [x] Game detail page
- [x] News sentiment debug view
- [x] Team form / trend view
- [x] Injury impact view
- [x] All verification tests pass

---

## Key Implementation Details

### Dashboard pages

| Page | Content |
|------|---------|
| **Today's Games** | Prediction slate, probability bars, confidence badges |
| **Game Detail** | Single-game deep dive, component outputs, top factors, recent form |
| **Model Performance** | Accuracy, log loss, ablations, rolling validation trend |
| **Calibration** | Calibration curve, reliability summary, ECE |
| **Team Form** | Recent results, point differential, net-rating trends |
| **News Sentiment** | Team-level news feature summary and fallback state |
| **Archive** | Past predictions and backtests with filtering |

### Design principles

```text
Dark analytics shell with teal/copper accents
Probability bars with away/home color split
Confidence badges: low (gray), medium (blue), high (green)
Responsive layout verified at 1920x1080 and 1366x768
Graceful fallback behavior for missing daily/backtest/news artifacts
```

### Data flow at Phase 9 completion

```text
Dashboard reads from:
  predictions/daily/*.json
  predictions/historical_backtests/*.json
  data/processed/news_features/
  data/processed/team_game_logs/
  models/baselines/ and models/ensembles/
```

---

## Verification Tests

Run:

```bash
pytest tests/test_dashboard.py -q
streamlit run streamlit_app.py
```

Automated coverage included:

- app import/startup checks
- predictions page rendering
- game detail rendering
- archive loading
- calibration chart generation
- team form rendering
- empty-data resilience
- probability consistency
- news summary coverage

**Actual:** `pytest tests/test_dashboard.py -q` -> 10 passed

### Manual Browser Tests

```text
[x] Dashboard loads locally
[x] Today's predictions show correct games
[x] Game detail renders for a selected matchup
[x] Calibration chart renders correctly
[x] Team form shows recent games
[x] News view shows coverage / fallback state
[x] Theme looks production-ready
[x] Layout works at 1920x1080 and 1366x768
```

---

## Definition of Done

- [x] All 10 automated tests pass
- [x] Manual browser verification completed
- [x] Dashboard runs with `streamlit run streamlit_app.py`
- [x] Screenshots captured for documentation
- [x] README updated with dashboard instructions

---

## Notes & Learnings

```text
Dashboard URL: http://localhost:8501
Startup time: roughly 8-15 seconds locally
Pages implemented: 8
Screenshots saved to:
  docs/screenshots/dashboard_today.png
  docs/screenshots/dashboard_calibration.png

Verification completed on 2026-05-16:
  - pytest tests/test_dashboard.py -q -> 10 passed
  - pytest tests -q -> 121 passed at Phase 9 completion
```
