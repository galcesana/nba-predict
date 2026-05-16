# Next-Generation Model Roadmap

> **Active forward roadmap for making the model materially stronger.**
> This document replaces the old pre-Phase-0 planning doc as the main planning source.
> Historical phase docs remain valuable, but they are now treated as implementation history rather than the active plan.

---

## 1. Why a New Roadmap Exists

The current system is a solid V1 forecasting product, but it still feels simpler than it should for three reasons:

1. The core predictor is still mostly **team-level**, not **player-level** or **lineup-level**.
2. The model architecture is multi-stream, but two of the streams are still comparatively shallow:
   - injury impact is still largely heuristic when full live player value is unavailable
   - news/sentiment is still sparse and often falls back to simple defaults
3. The product explanations are only partially learned:
   - the displayed “top factors” are still rule-based heuristics rather than full model attribution

The result is a system that is useful and honest, but not yet convincingly “smart.”

This roadmap is designed to fix that in the right order:

1. improve the **representation**
2. improve the **regime handling**
3. improve the **decision layer and calibration**
4. improve the **explanations and uncertainty**

---

## 2. Guiding Principles For The Next Iteration

### 2.1 Data and representation matter more than making the MLP deeper

The biggest expected gains will come from:

- player-aware availability
- lineup-aware matchup context
- regime-specific behavior

not from simply adding more hidden layers to the existing fusion head.

### 2.2 Preserve leakage safety

For every upgrade in this roadmap:

- only use information known before tip-off
- never let post-game rotation, box-score, or article information leak into pregame features
- preserve time-based splits for all experiments

### 2.3 Keep a strong baseline culture

Every major neural upgrade must be compared against strong tabular baselines:

- CatBoost
- LightGBM
- XGBoost
- current ensemble

If a more complex architecture is not clearly better on log loss and calibration, it should not replace the simpler model.

### 2.4 Build for both model quality and product quality

A model that improves log loss by 0.003 but still cannot explain itself well may not feel more impressive.

Future work should improve:

- forecast quality
- calibration
- reliability under missing context
- explanation quality
- visible intelligence in the UI/API

---

## 3. Summary Of The Current Starting Point

The current production system already has:

- historical game and team-log pipeline
- rolling and schedule features
- Elo + tabular baselines
- GRU-based team sequence encoder
- injury/news/context fusion model
- calibrated ensemble
- live publishing
- Streamlit dashboard
- FastAPI service

The current system summary is documented in:

- [current_system_implementation_summary.md](current_system_implementation_summary.md)
- [m1_player_lineup_foundation.md](m1_player_lineup_foundation.md) for the first concrete implementation milestone

That summary should be treated as the canonical “how the existing system works” reference before new work begins.

---

## 4. Active Workstreams

The roadmap is split into five workstreams.

### Workstream A. Player + Lineup Data Foundation

**Goal:** Move from a team-level forecast to a roster- and lineup-aware forecast.

Why this matters:

- NBA outcomes swing heavily on who actually plays
- team-level aggregates blur important matchup structure
- a lineup-aware model will feel much smarter even before architecture changes

Primary sources to add:

- official NBA team rosters
- official injury report entities resolved to players
- game rotation data
- lineup usage / net rating data
- player advanced box-score data
- player tracking data where feasible

Expected data interfaces:

- `data/processed/player_game_logs/`
- `data/processed/projected_availability/`
- `data/processed/lineup_features/`
- `data/processed/player_value_features/`

Key modeling objects to build:

- player-value estimate per player-game
- expected availability state per player:
  - available
  - questionable
  - probable
  - doubtful
  - out
- expected top-8 rotation representation
- expected starting lineup representation
- lineup continuity / familiarity features
- bench strength and fallback depth features

Key deliverables:

- player entity resolution and stable player IDs
- resolved injury-to-player pipeline
- player-value feature builder
- lineup aggregation builder
- rotation stability and synergy features
- training-ready matchup rows with player/lineup context

Acceptance criteria:

- every target game has a reproducible pregame player/lineup snapshot
- live forecasts can distinguish “star out” from “two role players out”
- the UI/API can surface expected lineup strength and missing player value

Major risks:

- lineup availability before tip-off is noisy
- injury statuses change close to game time
- NBA endpoints can be brittle

Mitigation:

- record forecast timestamp and snapshot provenance
- keep projected-availability confidence
- preserve fallback path when projection coverage is partial

---

### Workstream B. Regime-Aware Modeling

**Goal:** Stop treating all NBA games as if they come from one homogeneous regime.

Why this matters:

- playoffs behave differently from regular season
- early season behaves differently from mid-season
- post-trade-deadline teams often shift identity
- back-to-backs, travel density, and rotation shortening interact differently by regime

Regimes to model explicitly:

- regular season
- playoffs / play-in
- early season
- post-All-Star
- post-trade-deadline
- low-context vs high-context live forecast windows

Approach options:

- regime features only
- mixture-of-experts with gating
- separate heads on top of shared representation
- separate calibrators per regime

Recommended order:

1. start with explicit regime features + regime-specific calibration
2. evaluate a shared trunk with regime-specific heads
3. only add a full mixture-of-experts if the simpler version earns it

Key deliverables:

- regime annotation pipeline for all games
- playoff-series-state features
- series fatigue and familiarity features
- regime-aware validation reports
- regime-specific calibration curves

Acceptance criteria:

- performance is reported by regime, not only overall
- playoff reliability improves without harming regular-season reliability
- publish-time forecasts clearly identify which regime model/head was used

---

### Workstream C. Stronger Representation Learning

**Goal:** Replace or augment the current “GRU + small MLP fusion” design with representations that can actually exploit richer inputs.

Why this matters:

- the current fusion model is reasonable but structurally simple
- once lineup-aware inputs exist, the architecture should reflect that structure

Recommended architecture ladder:

#### C1. Stronger tabular baseline stack

Before replacing the neural model, add stronger comparison models:

- CatBoost
- tuned LightGBM
- tuned XGBoost refresh
- FT-Transformer benchmark

Reason:

- tabular tasks still reward strong tree methods
- this gives a much tougher baseline and better ensemble ingredients

#### C2. Set-based roster / lineup encoders

Add permutation-invariant encoders for player sets and lineups:

- Deep Sets style encoder
- Set Transformer style encoder

Target use cases:

- expected starters set
- top-8 rotation set
- injured-player set
- lineup cluster summaries

#### C3. Improved sequence backbone

After player/lineup features exist, upgrade temporal modeling:

- TCN benchmark
- attention-based sequence model benchmark
- Temporal Fusion Transformer style architecture if justified

Important note:

Do not jump straight to a heavy Transformer on current team-level inputs. That is likely to add complexity without enough new signal.

#### C4. Smarter ensemble

Upgrade the ensemble from a simple static stack to a stronger forecast combiner:

- regime-aware stacking
- context-aware weighting
- reliability-aware weighting when injury/news coverage is weak

Key deliverables:

- benchmark matrix across model families
- roster-set encoder prototype
- sequence-backbone comparison report
- upgraded ensemble selection report

Acceptance criteria:

- better log loss than current ensemble on held-out future seasons
- no degradation in calibration quality
- performance gains hold across regimes, not only overall

---

### Workstream D. Explanations, Attribution, and Uncertainty

**Goal:** Make the model feel meaningfully smarter to users, not just numerically better.

Why this matters:

- current “top factors” are partially rule-based
- users want to know why the forecast moved
- the app should communicate uncertainty honestly

Targets:

- model-derived local feature attribution
- player-level counterfactuals
- lineup-level counterfactuals
- uncertainty flags when context coverage is weak
- better calibration and coverage diagnostics

Features to add:

- per-game attribution summary
- “biggest swing players” summary
- “if player X is ruled out, home win probability changes by Y points”
- context completeness score
- stale-data or low-confidence warning level
- optional conformal or interval-style uncertainty band around the probability estimate

Key deliverables:

- learned attribution pipeline for the winning model
- counterfactual simulator for lineup/injury changes
- uncertainty metadata in published JSON and API
- dashboard cards for attribution and confidence quality

Acceptance criteria:

- every live forecast can explain itself in model-grounded terms
- weak-context forecasts are visibly marked as such
- the app can show what changed between two forecast snapshots

---

### Workstream E. Experimentation, Evaluation, and Monitoring

**Goal:** Make model iteration faster and more trustworthy.

Why this matters:

- a richer model stack needs stronger evaluation discipline
- otherwise the project will become more complex without becoming better

Needed upgrades:

- experiment registry for model runs
- per-regime scorecards
- ablation harness for each new stream
- feature coverage reports
- drift / calibration monitoring over time
- publish-time validation checks before deployment artifacts are committed

Key deliverables:

- experiment report schema
- model comparison board
- rolling calibration monitor
- pre-publish validation checklist
- automated publish blocker for clearly stale or malformed context

Acceptance criteria:

- every new model run produces comparable artifacts
- every publish run can be audited
- regressions are caught before they reach the dashboard

---

## 5. Recommended Execution Order

This is the recommended order of implementation.

### Stage 1. Build the new data foundation

Do first:

- player identity and roster data
- injury-to-player resolution
- projected availability snapshots
- lineup and rotation feature tables

Do not start a major new architecture before this stage exists.

### Stage 2. Upgrade the benchmark stack

Do next:

- CatBoost
- LightGBM refresh
- FT-Transformer benchmark
- stronger ensemble comparison

This stage answers an important question:

“Are we missing performance because of the architecture, or because of the data representation?”

### Stage 3. Add regime awareness

Once the new data exists:

- annotate regimes
- report performance by regime
- add regime-specific calibration
- test regime-aware heads

### Stage 4. Add set-based and improved sequence models

Only after Stages 1 to 3:

- lineup set encoder
- rotation set encoder
- stronger temporal backbone
- smarter ensemble

### Stage 5. Add attribution and uncertainty

Then:

- learned explanations
- counterfactuals
- context-confidence reporting
- richer UI/API forecasting intelligence

---

## 6. Proposed New Milestones

These replace the old future-phase mindset.

### Milestone M1. Player + Lineup Foundation

Outcome:

- complete pregame player and lineup data layer

Success metrics:

- reliable projected-availability coverage
- usable lineup features on historical and live slates

### Milestone M2. Strong Baselines Refresh

Outcome:

- clear performance map across tree, tabular deep, and current fusion approaches

Success metrics:

- stronger baseline leaderboard
- confirmed target architecture direction

### Milestone M3. Regime-Aware Forecasting

Outcome:

- explicit handling of playoffs and season sub-regimes

Success metrics:

- improved playoff log loss and calibration

### Milestone M4. Next-Generation Representation

Outcome:

- player/lineup-aware model replacing or augmenting the current GRU fusion core

Success metrics:

- meaningful held-out improvement over the current ensemble

### Milestone M5. Explainable Live Forecasting

Outcome:

- production forecasts that feel smarter because they can explain movement and uncertainty

Success metrics:

- attribution, counterfactuals, and context confidence visible in dashboard/API

---

## 7. Immediate Next Build Recommendation

If only one milestone is started next, it should be:

## **M1. Player + Lineup Foundation**

Rationale:

- this is the highest-leverage structural improvement
- it strengthens both the model and the user-facing product
- it unlocks better regime handling, better representations, and better explanations later

Concrete immediate scope:

1. add player-level historical tables
2. add injury entity resolution to player IDs
3. add projected availability snapshots
4. add lineup continuity and rotation depth features
5. update historical matchup rows to include the new player/lineup features

---

## 8. What This Roadmap Explicitly De-Prioritizes

These are not forbidden, but they are not the first best next move:

- simply making the current fusion MLP wider or deeper
- replacing everything with a large Transformer before new data exists
- betting-style market feature integration
- adding more UI polish without improving explanation quality
- adding more endpoints before the model itself becomes stronger

---

## 9. Definition Of Success For “A Stronger Model”

The next-generation system should not be judged only by raw accuracy.

It should be considered stronger if it delivers all of the following:

- lower log loss on future held-out seasons
- equal or better calibration
- better playoff behavior
- stronger handling of injury-driven shifts
- materially richer explanations
- visible confidence handling when the live context is incomplete

If those do not improve together, the extra complexity is probably not worth it.
