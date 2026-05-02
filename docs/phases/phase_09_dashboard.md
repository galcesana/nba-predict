# Phase 9 — Product Dashboard

| Field | Value |
|-------|-------|
| **Size** | L (1–2 weeks) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 8 |
| **Unlocks** | — (final phase) |

---

## Goal

Build a Streamlit dashboard that presents today's predictions, model performance, calibration charts, and debugging views for news sentiment. This is the user-facing product.

---

## Deliverables Checklist

- [ ] `src/app/streamlit_app.py` — main dashboard application
- [ ] Today's predictions page
- [ ] Historical prediction archive page
- [ ] Model performance / backtest page
- [ ] Calibration charts page
- [ ] Game detail page (drill into a single game)
- [ ] News sentiment debug view
- [ ] Team form / trend view
- [ ] Injury impact view
- [ ] All verification tests pass

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
Dark theme (professional sports analytics aesthetic)
Probability bars with color gradients (red → yellow → green)
Confidence badges: low (gray), medium (blue), high (green)
Responsive layout (works on desktop and tablet)
Auto-refresh option for live game days
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

**Expected: 10/10 pass.**

### Manual Browser Tests

```text
[ ] Dashboard loads at localhost:8501
[ ] Today's predictions show correct games
[ ] Clicking a game shows detail view
[ ] Calibration chart renders correctly
[ ] Team form shows recent games
[ ] News view shows article scores
[ ] Dark theme looks professional
[ ] Layout works on 1920×1080 and 1366×768
```

---

## Definition of Done

- [ ] All 10 automated tests pass
- [ ] All 8 manual browser tests pass
- [ ] Dashboard runs with `streamlit run src/app/streamlit_app.py`
- [ ] Screenshots captured for documentation
- [ ] README updated with dashboard instructions

---

## Notes & Learnings

```
Dashboard URL: http://localhost:8501
Startup time: ___ seconds
Pages implemented: ___/7
Screenshots saved to: ___
```
