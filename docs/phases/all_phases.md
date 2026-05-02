# NBA Game Prediction — Implementation Phases

> **Master tracker for all implementation phases.**
> Update the status column as you work. Each phase has its own detailed doc linked below.

---

## Timeline Overview

| Phase | Name | Size | Est. Time | Status | Depends On |
|-------|------|------|-----------|--------|------------|
| [0](phase_00_project_setup.md) | Project Setup | S | 1–2 days | `[ ]` Not Started | — |
| [1](phase_01_historical_data.md) | Historical Data Foundation | M | 3–5 days | `[ ]` Not Started | Phase 0 |
| [2](phase_02_feature_table.md) | Leakage-Safe Feature Table | M | 3–5 days | `[ ]` Not Started | Phase 1 |
| [3](phase_03_baselines.md) | Baseline Models | M | 3–5 days | `[ ]` Not Started | Phase 2 |
| [4](phase_04_sequence_model.md) | Team Sequence Model | L | 1–2 weeks | `[ ]` Not Started | Phase 3 |
| [5](phase_05_injury_features.md) | Injury Features | M | 3–5 days | `[ ]` Not Started | Phase 3 |
| [6](phase_06_llm_sentiment.md) | LLM News/Sentiment Layer | XL | 2–3 weeks | `[ ]` Not Started | Phase 4 |
| [7](phase_07_fusion_model.md) | Full Fusion Model | L | 1–2 weeks | `[ ]` Not Started | Phase 4 + 5 + 6 |
| [8](phase_08_daily_predictions.md) | Daily Prediction System | M | 3–5 days | `[ ]` Not Started | Phase 7 |
| [9](phase_09_dashboard.md) | Product Dashboard | L | 1–2 weeks | `[ ]` Not Started | Phase 8 |

**Total estimated: ~10–14 weeks**

---

## Status Legend

```
[ ] Not Started
[/] In Progress
[x] Complete
[!] Blocked
```

---

## Dependency Graph

```text
Phase 0 → Phase 1 → Phase 2 → Phase 3 ─┬→ Phase 4 ──┐
                                         │             │
                                         ├→ Phase 5 ──┤
                                         │             │
                                         │  Phase 6 ───┤
                                         │             ▼
                                         │        Phase 7 → Phase 8 → Phase 9
                                         │
                                         └→ (Phase 6 can start after Phase 3,
                                             needs Phase 4 for integration)
```

---

## How to Use These Docs

1. **Start each phase** by reading its full doc and checking prerequisites
2. **Track progress** by checking off deliverables in the phase doc
3. **Run verification tests** at the end — do not proceed until all pass
4. **Update this master file** with the phase status as you work
5. **Leave notes** in each phase doc's "Notes & Learnings" section

---

## Test Summary Across Phases

| Phase | Tests | Focus |
|-------|-------|-------|
| 0 | 7 | Structure, configs, imports |
| 1 | 13 | Data schemas, coverage, caching |
| 2 | 14 | Leakage prevention (5), feature correctness (9) |
| 3 | 12 | Model training, performance, calibration |
| 4 | 14 | Sequences, padding, gradients, determinism |
| 5 | 10 | Injury vectors, trade awareness, ablation |
| 6 | 14 | LLM schema, cost, determinism, validation |
| 7 | 12 | Fusion architecture, ablation study, calibration |
| 8 | 12 | Prediction pipeline, output format, backtest |
| 9 | 10 + 8 manual | Dashboard pages, rendering, browser |
| **Total** | **118 + 8 manual** | |

---

## Quick Reference

### Core Principles

- **Predict probabilities**, not just winners
- **No data leakage** — only pre-tip-off information
- **Anonymous team IDs** — no name memorization
- **LLM is a feature extractor**, not the predictor
- **Reproducibility** — pin seeds, version snapshots, log everything

### Key Technical Decisions

| Decision | Choice |
|----------|--------|
| Data source | `nba_api` behind `DataProvider` interface |
| Historical news | None — zero vector + `news_available=0` pre-2023-24 |
| Sequence padding | Zero-pad + binary mask |
| LLM model | GPT-4o-mini (temp=0), Haiku/Flash fallback |
| LLM schema | 7 core fields MVP, 8 extended deferred |
| Normalization | Per-season StandardScaler, train-only fit |
| Scope | Regular season only (V1) |
| Meta-model | Logistic regression ensemble |
