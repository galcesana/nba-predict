# NBA Game Prediction - Implementation Phases

> **Master tracker for all implementation phases.**
> Update the status column as you work. Each phase has its own detailed doc linked below.
>
> This tracker is now primarily a **historical implementation record** for phases 0-12.
> The active forward roadmap lives in [../next_generation_model_roadmap.md](../next_generation_model_roadmap.md).

---

## Timeline Overview

| Phase | Name | Size | Est. Time | Status | Depends On |
|-------|------|------|-----------|--------|------------|
| [0](phase_00_project_setup.md) | Project Setup | S | 1-2 days | `[x]` Complete | - |
| [1](phase_01_historical_data.md) | Historical Data Foundation | M | 3-5 days | `[x]` Complete | Phase 0 |
| [2](phase_02_feature_table.md) | Leakage-Safe Feature Table | M | 3-5 days | `[x]` Complete | Phase 1 |
| [3](phase_03_baselines.md) | Baseline Models | M | 3-5 days | `[x]` Complete | Phase 2 |
| [4](phase_04_sequence_model.md) | Team Sequence Model | L | 1-2 weeks | `[x]` Complete | Phase 3 |
| [5](phase_05_injury_features.md) | Injury Features | M | 3-5 days | `[x]` Complete | Phase 3 |
| [6](phase_06_llm_sentiment.md) | LLM News/Sentiment Layer | XL | 2-3 weeks | `[x]` Complete | Phase 4 |
| [7](phase_07_fusion_model.md) | Full Fusion Model | L | 1-2 weeks | `[x]` Complete | Phase 4 + 5 + 6 |
| [8](phase_08_daily_predictions.md) | Daily Prediction System | M | 3-5 days | `[x]` Complete | Phase 7 |
| [9](phase_09_dashboard.md) | Product Dashboard | L | 1-2 weeks | `[x]` Complete | Phase 8 |
| [10](phase_10_live_publishing.md) | Live Publishing Layer | M | 2-4 days | `[x]` Complete | Phase 9 |
| [11](phase_11_live_context_playoff_hardening.md) | Live Context + Playoff Hardening | L | 4-7 days | `[x]` Complete | Phase 10 |
| [12](phase_12_api_service_layer.md) | API Service Layer | M | 2-4 days | `[x]` Complete | Phase 11 |

**Total estimated: ~13-17 weeks**

---

## Status Legend

```text
[ ] Not Started
[/] In Progress
[x] Complete
[!] Blocked
```

---

## Dependency Graph

```text
Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 ----+
                                 |                       |
                                 +-> Phase 5 -----------|
                                 |                       |
                                 +-> Phase 6 -----------|
                                                         v
                                                    Phase 7 -> Phase 8 -> Phase 9 -> Phase 10 -> Phase 11 -> Phase 12
```

---

## How to Use These Docs

1. **Start each phase** by reading its full doc and checking prerequisites.
2. **Track progress** by checking off deliverables in the phase doc.
3. **Run verification tests** at the end - do not proceed until all pass.
4. **Update this master file** with the phase status as you work.
5. **Leave notes** in each phase doc's "Notes & Learnings" section.

---

## Test Summary Across Phases

| Phase | Tests | Focus |
|-------|-------|-------|
| 0 | 21 (actual) | Structure, configs, imports |
| 1 | 12 (actual) | Data schemas, coverage, caching |
| 2 | 14 | Leakage prevention (5), feature correctness (9) |
| 3 | 12 | Model training, performance, calibration |
| 4 | 14 | Sequences, padding, gradients, determinism |
| 5 | 10 | Injury vectors, trade awareness, ablation |
| 6 | 10 | News proxy features, availability flags, aggregation logic |
| 7 | 6 | Fusion forward pass plus ensemble training coverage |
| 8 | 12 | Prediction pipeline, output format, backtest |
| 9 | 10 + 8 manual | Dashboard pages, rendering, browser |
| 10 | 8 | Publishing, manifest state, dashboard source precedence |
| 11 | 11 | Live injury/news context overlays, coverage summaries, playoff filtering |
| 12 | 8 | API health, manifest, forecast, game detail, and metrics endpoints |
| **Total** | **148 automated + 8 manual** | |

---

## Quick Reference

### Core Principles

- **Predict probabilities**, not just winners
- **No data leakage** - only pre-tip-off information
- **Anonymous team IDs** - no name memorization
- **LLM is a feature extractor**, not the predictor
- **Reproducibility** - pin seeds, version snapshots, log everything

### Key Technical Decisions

| Decision | Choice |
|----------|--------|
| Data source | `nba_api` behind `DataProvider` interface |
| Historical news | None - zero vector + `news_available=0` pre-2023-24 |
| Sequence padding | Zero-pad + binary mask |
| LLM model | GPT-4o-mini (temp=0), Haiku/Flash fallback |
| LLM schema | 7 core fields MVP, 8 extended deferred |
| Normalization | Per-season StandardScaler, train-only fit |
| Scope | Regular season only (V1) |
| Meta-model | Logistic regression ensemble |
