# Phase 12 - API Service Layer

| Field | Value |
|-------|-------|
| **Size** | M (2-4 days) |
| **Status** | `[x]` Complete |
| **Depends on** | Phase 11 |
| **Unlocks** | Programmatic access to published forecasts, manifest health, and model diagnostics |

---

## Goal

Expose the live published forecast and model diagnostics through a stable FastAPI service so:

- external clients can consume the current weekly slate without scraping Streamlit
- health and publish freshness can be checked automatically
- one game can be fetched directly by `game_id`
- model metrics can be reused by future integrations and monitoring

---

## Deliverables Checklist

- [x] `src/app/api.py` - FastAPI service with local CLI entrypoint
- [x] `GET /health` endpoint for service and forecast freshness
- [x] `GET /manifest` endpoint for the tracked publish manifest
- [x] `GET /forecast/week` endpoint for the current weekly slate
- [x] `GET /forecast/game/{game_id}` endpoint for one matchup
- [x] `GET /metrics` endpoint for model performance and calibration summaries
- [x] Internal API responses strip repo-local file path details
- [x] `tests/test_api.py` added with endpoint coverage
- [x] `requirements.txt` updated with FastAPI + Uvicorn runtime deps
- [x] `Makefile` updated with `serve-api`
- [x] Tracking docs updated for Phase 12
- [x] Verification tests pass

---

## Key Implementation Details

### Service design

The API deliberately reuses the existing dashboard data loaders instead of inventing a new data path:

```text
published/daily/latest.json
published/manifest.json
models/* evaluation artifacts
```

That keeps the Streamlit app and API in sync and makes the service a thin delivery layer over the same source of truth.

### Endpoint contract

The service now exposes:

```text
GET /
GET /health
GET /manifest
GET /forecast/week
GET /forecast/game/{game_id}
GET /metrics
```

Notes:

- `/health` reports `service_status` as `ok`, `degraded`, or `unavailable`
- `/forecast/week` returns both the full payload and `games_by_date` buckets
- `/forecast/game/{game_id}` adds display-friendly team labels and matchup text
- `/metrics` serializes model comparison, calibration bins, ensemble weights, and rolling validation

### Serialization and safety

The API normalizes pandas, numpy, timestamps, and `NaN` values into JSON-safe primitives and removes `source_file` fields so local filesystem paths are not leaked to clients.

### Local run command

```bash
uvicorn src.app.api:app --reload
```

or:

```bash
make serve-api
```

---

## Verification Tests

Run:

```bash
pytest tests/test_api.py -q
pytest tests -q
python -m src.app.api --help
python -m ruff check src/app/api.py tests/test_api.py
```

**Actual:**

- [x] `pytest tests/test_api.py -q` -> 8 passed
- [x] `pytest tests -q` -> 148 passed
- [x] `python -m src.app.api --help` -> CLI usage printed successfully
- [x] `python -m ruff check src/app/api.py tests/test_api.py` -> passed

---

## Definition of Done

- [x] API serves the same current forecast source the dashboard uses
- [x] Health endpoint surfaces forecast freshness cleanly
- [x] One-game lookup works by `game_id`
- [x] Metrics endpoint exposes model-quality summaries in JSON
- [x] Local run command is documented and verified
- [x] Automated verification passes

---

## Notes & Learnings

```text
Phase 12 works best as a delivery layer, not a second pipeline.

Reusing the dashboard loaders kept the API small and reduced the risk that the
Streamlit app and service would drift onto different forecast sources or metric
definitions. That also means future monitoring can hit the API and trust it is
describing the same published slate the app is showing.

Verification completed on 2026-05-17:
  - pytest tests/test_api.py -q -> 8 passed
  - pytest tests -q -> 148 passed
  - python -m src.app.api --help -> CLI usage printed
  - python -m ruff check src/app/api.py tests/test_api.py -> passed
```
