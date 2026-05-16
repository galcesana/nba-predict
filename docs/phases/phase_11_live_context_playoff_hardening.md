# Phase 11 - Live Context + Playoff Hardening

| Field | Value |
|-------|-------|
| **Size** | L (4-7 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 10 |
| **Unlocks** | Trustworthy live weekly forecasts with context coverage and clearer playoff handling |

---

## Goal

Move the deployed forecast from "live schedule with fallback features" to a more truthful live product by:

- collecting real official injury report context when it exists
- collecting real recent team-news articles for live slates
- scoring those articles into structured team sentiment features
- exposing context freshness and coverage in the published payload and dashboard
- making the weekly playoff board explicitly show only the next scheduled game per series

---

## Deliverables Checklist

- [x] `src/data/team_metadata.py` - shared team alias and metadata helpers
- [x] `src/data/fetch_injuries.py` - official NBA injury-report PDF fetcher + parser
- [x] `src/data/fetch_news.py` - live team-news collector for current slate teams
- [x] `src/nlp/article_filtering.py` - team-relevance filter for live articles
- [x] `src/nlp/extract_sentiment.py` - deterministic structured article scorer
- [x] `src/nlp/sentiment_schema.py` - article-level schema
- [x] `src/features/injury_features.py` updated to overlay live official report data on fallback features
- [x] `src/features/news_features.py` updated to overlay live article aggregates on fallback features
- [x] `src/models/predict.py` updated to carry per-game context provenance into prediction payloads
- [x] `src/app/predict_today.py` updated to emit slate-level context summaries
- [x] `src/app/publish_today.py` updated to persist context summary in `published/manifest.json`
- [x] `src/app/dashboard_data.py` + `src/app/streamlit_app.py` updated to show live context coverage in the UI
- [x] Weekly playoff publishing still limits the slate to the next game per series
- [x] Tracking docs updated for Phase 11
- [x] Verification tests pass

---

## Key Implementation Details

### Live injury context

```text
1. Generate the official report PDF URL candidates for the current ET publish date
2. Download the newest available report snapshot
3. Parse team/player/status rows from the PDF
4. Map report matchups back onto the current forecast slate
5. Overlay submitted-team injury features onto the fallback injury proxy rows
```

Notes:

- The live source is the official NBA injury report PDF pattern under `ak-static.cms.nba.com`
- Teams with `NOT YET SUBMITTED` remain on fallback injury context
- The live layer is real report data, but the team-level impact weights are still heuristic

### Live news context

```text
1. Query Google News RSS for teams appearing in the live slate
2. Filter to team-relevant items
3. Score each article with a deterministic structured sentiment extractor
4. Aggregate into per-team, per-game features
5. Overlay those rows onto the fallback zero-vector news features
```

Notes:

- The collector only runs for live forecast windows
- Historical processed feature builds remain offline and deterministic
- The scorer is deterministic and schema-based, not a hidden black-box call

### Published payload improvements

Each live prediction now includes `context_details`, and the slate now carries:

```json
{
  "context_summary": {
    "injury_live_games": 0,
    "injury_partial_games": 0,
    "news_live_games": 1,
    "news_partial_games": 0,
    "injury_coverage_rate": 0.0,
    "news_coverage_rate": 0.5
  }
}
```

This same summary is copied into `published/manifest.json` for deployment consumers.

### Dashboard behavior

- `This Week's Games` now surfaces injury/news coverage metrics and latest live-context timestamps
- `Injury Impact` prefers the current live published slate summary when present
- `News Sentiment` prefers the current live published slate summary when present
- Both pages fall back gracefully and say so explicitly when coverage is missing

---

## Verification Tests

Run:

```bash
pytest tests/test_live_context.py tests/test_publishing.py tests/test_dashboard.py tests/test_predictions.py -q
pytest tests -q
python -m src.app.publish_today --date 2026-05-16
python -m ruff check src/data/team_metadata.py src/data/fetch_news.py src/data/fetch_injuries.py src/features/injury_features.py src/features/news_features.py src/nlp/article_filtering.py src/nlp/extract_sentiment.py src/nlp/sentiment_schema.py src/models/predict.py src/app/predict_today.py src/app/publish_today.py src/app/dashboard_data.py src/app/streamlit_app.py tests/test_live_context.py tests/test_publishing.py tests/test_dashboard.py tests/test_predictions.py
```

**Actual:**

- [x] `pytest tests/test_live_context.py tests/test_publishing.py tests/test_dashboard.py tests/test_predictions.py -q` -> 41 passed
- [x] `pytest tests -q` -> 140 passed
- [x] `python -m src.app.publish_today --date 2026-05-16` -> published a 2-game live weekly slate with context summary metadata
- [x] targeted `ruff check` on the touched live-context files passed
- [ ] full browser automation pass was attempted, but local Playwright runtime support was incomplete in this environment

---

## Definition of Done

- [x] Live slates attempt real injury and news context before falling back
- [x] Prediction payloads distinguish live, partial, and fallback context modes
- [x] Published manifest exposes context coverage
- [x] Dashboard copy reflects real coverage instead of proxy-only assumptions
- [x] Weekly playoff board remains limited to the next game per series
- [x] Automated verification passes

---

## Notes & Learnings

```text
The biggest product gap after Phase 10 was not distribution, it was truthfulness:
the app was "live" on schedule timing but not on context quality.

Phase 11 keeps the system honest by attaching provenance and coverage to every
published slate. That lets the UI say "live news coverage is partial" instead of
pretending every forecast has the same context quality.

The official injury report source is real, but weekly publishing still means some
later-in-the-week games will remain on fallback until their reports exist.

Verification completed on 2026-05-16:
  - pytest tests/test_live_context.py tests/test_publishing.py tests/test_dashboard.py tests/test_predictions.py -q -> 41 passed
  - pytest tests -q -> 140 passed
  - python -m src.app.publish_today --date 2026-05-16 -> published/daily/2026-05-16.json
```
