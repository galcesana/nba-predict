# M1 - Player + Lineup Foundation

> **First active implementation milestone under the new roadmap.**
> This milestone is the highest-leverage next step for making the model materially stronger.

---

## 1. Objective

Move the project from a mostly team-level forecasting system to a **player-aware, lineup-aware pregame forecasting foundation**.

This milestone does **not** aim to deliver the final next-generation model yet.

It aims to build the data and feature layer that future stronger models will depend on:

- player identities
- roster snapshots
- resolved injury entities
- projected availability
- expected lineups
- rotation and depth features
- player-value summaries

At the end of this milestone, the project should be able to say:

> “Here is the best pregame estimate of who is expected to play, what lineups each team is likely to use, how much value is missing, and how stable each team’s rotation is.”

---

## 2. Why This Is The Right First Milestone

The current model’s biggest limitation is not the lack of a bigger neural network.

It is that the representation is still too coarse:

- team-level injury summaries blur important player differences
- current live news and injury overlays do not tell the model enough about specific expected rotations
- playoff and late-season basketball are heavily driven by rotation tightening and matchup-specific lineups

Without this milestone:

- a stronger model architecture will still be learning from shallow inputs
- explanations will remain vague
- counterfactuals like “what if player X is ruled out?” will remain weak

---

## 3. Scope

### In scope

- player metadata foundation
- historical player-game table
- roster snapshot handling
- injury report entity resolution to player IDs
- projected availability table for pregame snapshots
- lineup and rotation feature engineering
- player-value estimation features
- integration into matchup-level training rows
- documentation, tests, and reproducible artifacts

### Out of scope

- final replacement of the current fusion model
- full playoff-specific model head
- attribution/counterfactual UI
- major dashboard redesign
- probabilistic uncertainty bands

Those belong to later milestones and depend on this one.

---

## 4. Design Principles

### 4.1 Preserve leakage safety

For a game on date `D`, all player and lineup features must be derived from information available before tip-off.

Forbidden:

- using postgame minutes from the target game
- using lineups that only became known after the game started
- using updated injury statuses that were published after tip-off

### 4.2 Distinguish observed from projected

The system must keep separate concepts for:

- historically observed player participation
- pregame projected availability
- live last-known status

We should never blur projected pregame information with known postgame reality.

### 4.3 Keep provenance explicit

Every projected availability or lineup feature row should be traceable to:

- source timestamp
- source type
- confidence level
- whether it was directly observed, heuristically inferred, or fallback-generated

### 4.4 Build for partial coverage

The milestone should support:

- complete lineup confidence
- partial injury confidence
- sparse late-breaking context

The system must degrade gracefully instead of requiring perfect coverage.

---

## 5. Deliverables

### D1. Player metadata foundation

New expected outputs:

- `data/mappings/player_to_idx.json`
- normalized player identity helper(s)
- player alias resolution utilities

Requirements:

- stable player IDs across seasons
- support for common name variants and suffix normalization
- mapping from official injury report names to internal player IDs

### D2. Historical player-game table

New expected outputs:

- `data/processed/player_game_logs/player_game_logs.parquet`

Required columns should include:

- `game_id`
- `date`
- `season`
- `team_idx`
- `player_id`
- `player_name`
- `started`
- `minutes`
- `points`
- `rebounds`
- `assists`
- `usage_proxy`
- `plus_minus` or net-impact proxy
- `available_for_game`

Purpose:

- support player-value modeling
- support lineup stability / rotation role inference

### D3. Player-value feature layer

New expected outputs:

- `data/processed/player_value_features/player_value_features.parquet`

Requirements:

- estimate player-level pregame value using historical-only data
- support multiple proxy families, such as:
  - rolling minutes
  - rolling usage
  - rolling on/off or plus-minus proxy
  - prior missed-game team drop-off / replacement-risk proxy
  - starter indicator
  - recent role stability

This does not need to be perfect player RAPM. It needs to be leakage-safe and materially better than “all outs count the same.”

### D4. Injury entity resolution

New expected outputs:

- normalized injury-report rows with resolved player IDs
- unresolved-entity audit table

Requirements:

- parse official injury report player names
- resolve to tracked player IDs
- record confidence / match type
- never silently collapse unmatched names

### D5. Projected availability table

New expected outputs:

- `data/processed/projected_availability/projected_availability.parquet`

Required fields should include:

- `game_id`
- `team_idx`
- `player_id`
- `status`
- `availability_score`
- `source_type`
- `availability_model_version`
- `source_timestamp`
- `projection_confidence`

Projection idea:

- `OUT` near zero
- `DOUBTFUL` very low
- `QUESTIONABLE` partial
- `PROJECTED_ABSENT` low, but not zero, when a rotation player missed prior team games
- `PROBABLE` high
- `AVAILABLE` full

This allows the model to reason over uncertainty instead of only binary availability.

Implementation note:

- `historical_absence_proxy_v1` is a leakage-safe proxy, not an official inactive feed.
- It uses only team games before the target date and never inspects whether the player appeared in the target game.
- It exists to create measurable historical missing-player coverage until richer official inactive history is available.
- `replacement_risk_v1` extends this by comparing team net rating or point differential in prior
  games when a player played versus prior games he missed, then carrying a risk-weighted missing
  value into projected availability.

### D6. Lineup and rotation features

New expected outputs:

- `data/processed/lineup_features/lineup_features.parquet`

Feature families should include:

- expected starter continuity
- expected top-8 continuity
- projected minutes concentration
- bench depth quality
- rotation stability over last N games
- lineup familiarity / shared minutes proxy
- expected missing starter value
- expected missing rotation value
- expected missing replacement risk

### D7. Matchup-row integration

Existing matchup/training rows should be extended with player/lineup-aware features.

Expected outputs:

- updated matchup dataset or parallel enriched dataset
- training-ready tensors/tables for future model work

This milestone should not destroy the current pipeline. It should create a clean path to compare:

- current team-level representation
- enriched player/lineup representation

### D8. Tests and validation

New expected tests:

- player identity resolution tests
- leakage-safety tests for player/lineup features
- projected availability logic tests
- lineup feature schema tests
- historical/backfill robustness tests

---

## 6. Implementation Breakdown

### Stage A. Foundations

Build first:

1. player metadata utilities
2. player ID mapping
3. normalized player naming and alias resolution

Files likely needed:

- `src/data/player_metadata.py`
- `src/anonymization/player_mapping.py`
- `tests/test_player_mapping.py`

### Stage B. Historical player-game ingestion

Build next:

1. fetch or derive player-level historical logs
2. store normalized player-game tables
3. join player rows to team/game structure

Likely outputs:

- `src/data/fetch_player_logs.py`
- `data/processed/player_game_logs/player_game_logs.parquet`
- `tests/test_player_logs.py`

### Stage C. Player-value estimation

Build next:

1. rolling player role/value features
2. starter/closer role indicators
3. recent value snapshots before each game

Likely outputs:

- `src/features/player_value_features.py`
- `tests/test_player_value_features.py`

### Stage D. Pregame projected availability

Build next:

1. resolve injury report names to players
2. convert statuses into probabilistic availability scores
3. preserve source timestamps and confidence
4. discount likely availability for rotation players who missed prior team games

Likely outputs:

- `src/features/projected_availability.py`
- `tests/test_projected_availability.py`

### Stage E. Lineup and rotation modeling

Build next:

1. infer expected starters and top-8 rotation
2. aggregate missing value by projected lineup role
3. compute continuity and familiarity metrics

Likely outputs:

- `src/features/lineup_features.py`
- `tests/test_lineup_features.py`

### Stage F. Matchup integration

Build last for this milestone:

1. join projected availability and lineup features into matchup rows
2. preserve current training path
3. produce enriched rows/tensors for model experiments

Likely outputs:

- updates to matchup feature builders
- updated inference scaffolding for future milestones
- `tests/test_enriched_matchup_rows.py`

---

## 7. Data Contracts

### Player identity contract

Each player used by new features must have:

- stable `player_id`
- normalized `player_name`
- `team_idx`
- optional alias list

### Availability contract

Availability should be represented numerically and categorically:

- categorical status
- continuous availability score
- source confidence

This is important because “questionable” is not the same as “out.”

### Lineup contract

Each team-game should support:

- expected starters
- expected top-8 rotation
- projected missing value
- continuity / stability features

---

## 8. Model-Readiness Goals

This milestone is successful if it unlocks the following later model upgrades:

- player-set encoders
- lineup-set encoders
- better injury impact modeling
- player-level counterfactuals
- playoff rotation-aware modeling

If the milestone only creates more tables but does not make those future upgrades easier, it has not done its job.

---

## 9. Verification Plan

Required checks:

1. entity resolution accuracy spot checks
2. no-leakage tests on player and lineup features
3. schema and null-rate reports
4. historical backfill sanity checks
5. live snapshot reproducibility checks

Suggested commands:

```bash
pytest tests/test_player_mapping.py -q
pytest tests/test_player_logs.py -q
pytest tests/test_player_value_features.py -q
pytest tests/test_projected_availability.py -q
pytest tests/test_lineup_features.py -q
pytest tests -q
ruff check src/ tests/
```

---

## 10. Definition Of Done

This milestone is done when:

- player IDs are stable and usable
- injury rows resolve to players with auditability
- projected availability exists pregame
- lineup and rotation features exist for historical and live paths
- enriched matchup rows are produced without leakage
- the current system can quantify missing player value better than simple team-level counts
- tests pass and docs are updated

---

## 11. Success Criteria

We should consider the milestone successful if it produces:

- better input realism
- better live forecast context
- better future model readiness
- better user-facing explanations later

This milestone does not need to improve the deployed model immediately.

It needs to make future improvements actually possible.
