# Phase 9 — Product Dashboard

| Field | Value |
|-------|-------|
| **Size** | L (1–2 weeks) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 8 |
| **Unlocks** | — (final phase) |

---

## Goal

Build a Streamlit dashboard that presents today's predictions, model performance, calibration charts, and debugging views for news sentiment. This is the user-facing product.

---

## Deliverables Checklist

- [x] `src/app/streamlit_app.py` — main dashboard application
- [x] Today's predictions page
- [x] Historical prediction archive page
- [x] Model performance / backtest page
- [x] Calibration charts page
- [x] Game detail page (drill into a single game)
- [x] News sentiment debug view
- [x] Team form / trend view
- [x] Injury impact view
- [x] All verification tests pass

---

## Key Implementation Details

### Dashboard pages

| Page | Content |
|------|---------|
| **Today's Games** | All predictions for today, probability bars, confidence badges |
| **Game Detail** | Single game deep-dive: features, component outputs, top factors, news articles |
| **Model Performance** | Accuracy, log loss, brier over time; rolling accuracy chart |
| **Calibration** | Calibration curve, reliability diagram, ECE metric |
| **Team Form** | Team's recent results, rolling net rating, injury status |
| **News Sentiment** | Raw articles, LLM scores, aggregated sentiment per team |
| **Archive** | Past predictions with actual outcomes; filter by date/team |

### Design principles

```text
Dark analytics shell with teal/copper accents
Probability bars with away/home color split
Confidence badges: low (gray), medium (blue), high (green)
Responsive layout verified at 1920x1080 and 1366x768
Empty-state handling for missing daily/backtest/news artifacts
```

### Data flow

```text
Dashboard reads from:
  predictions/daily/*.json (today's predictions)
  predictions/historical_backtests/*.json (past performance)
  data/processed/news_features/ (sentiment debug)
  models/baselines/ & models/ensembles/ (metrics)
  data/processed/team_game_logs/ (team form)
```

---

## Verification Tests

Run: `pytest tests/test_dashboard.py -v` + browser testing

```python
# tests/test_dashboard.py

def test_streamlit_app_imports():
    """Streamlit app module imports without error."""

def test_predictions_page_renders():
    """Today's predictions page generates valid HTML/components."""

def test_game_detail_page_renders():
    """Game detail page renders for a known game_id."""

def test_calibration_chart_generates():
    """Calibration chart renders without error."""

def test_team_form_page_renders():
    """Team form page renders for a known team_idx."""

def test_archive_page_loads():
    """Archive page loads historical predictions."""

def test_no_crashes_with_empty_data():
    """Pages handle missing/empty data gracefully (no crashes)."""

def test_probability_display_correct():
    """Displayed probabilities match underlying JSON data."""

def test_news_debug_view():
    """News sentiment debug view shows articles and scores."""

def test_dashboard_startup():
    """streamlit run src/app/streamlit_app.py starts without error."""
```

**Actual: 10/10 pass (`pytest tests/test_dashboard.py -q`).**

### Manual Browser Tests

```text
[x] Dashboard loads at localhost:8501
[x] Today's predictions show correct games
[x] Clicking a game shows detail view
[x] Calibration chart renders correctly
[x] Team form shows recent games
[x] News view shows article scores / fallback state
[x] Dark theme looks professional
[x] Layout works on 1920×1080 and 1366×768
```

---

## Definition of Done

- [x] All 10 automated tests pass
- [x] All 8 manual browser tests pass
- [x] Dashboard runs with `streamlit run src/app/streamlit_app.py`
- [x] Screenshots captured for documentation
- [x] README updated with dashboard instructions

---

## Notes & Learnings

```
Dashboard URL: http://localhost:8501
Startup time: ~8-15 seconds in the local environment
Pages implemented: 8 (Today's Games, Game Detail, Archive, Performance,
                     Calibration, Team Form, Injury Impact, News Sentiment)
Screenshots saved to:
  docs/screenshots/dashboard_today.png
  docs/screenshots/dashboard_calibration.png

Verification completed on 2026-05-16:
  - pytest tests/test_dashboard.py -q -> 10 passed
  - pytest tests -q -> 121 passed
  - Browser validation covered Today, Archive, Performance, Calibration,
    Team Form, Game Detail, Injury Impact, and News Sentiment
```
