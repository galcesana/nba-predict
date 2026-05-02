# NBA Game Outcome Prediction Model — Comprehensive Project Plan

**Project goal:** Build an ambitious NBA game-outcome prediction system that predicts calibrated win probabilities using historical performance, recent form, injuries, schedule context, and LLM-extracted news/team-spirit signals.

**Core prediction task:** For a scheduled game, output:

```json
{
  "game_id": "0022500451",
  "date": "2026-01-15",
  "home_team_idx": 4,
  "away_team_idx": 17,
  "home_win_probability": 0.64,
  "away_win_probability": 0.36,
  "predicted_winner": "home_team_idx_4",
  "confidence": "medium",
  "model_version": "team_sequence_news_fusion_v1"
}
```

This project should be treated as a **sports analytics and forecasting project**, not a betting project.

---

## 1. Guiding Principles

### 1.1 Predict probabilities, not just winners

The model should output a calibrated probability:

```text
P(home team wins)
```

Not merely:

```text
home wins / away wins
```

A good model should be able to say:

```text
Home team is favored, but only 56%.
```

That is more honest and more useful than a hard prediction.

---

### 1.2 Avoid data leakage

For a game on date `D`, the model may only use information that would have been known **before tip-off**.

Allowed:

```text
Previous games before D
Team stats before D
Player availability known before tip-off
News published before tip-off
Schedule context known before D
Historical team/player data before D
```

Forbidden:

```text
Final score of the target game
Box score of the target game
Season averages calculated after the target game
Standings after the target game
Articles published after the game
Post-game injury reports
Future schedule outcomes
```

This is the most important rule in the entire project.

---

### 1.3 Keep team names anonymous inside the model

Use real team names only for:

```text
Data collection
Entity resolution
Debugging
Human-facing UI
```

Internally, map teams to anonymous IDs:

```json
{
  "ATL": 0,
  "BOS": 1,
  "BKN": 2,
  "CHA": 3
}
```

The model should mainly learn from:

```text
Team features
Team recent-game sequences
Injury impact
News/team-spirit signals
Schedule context
```

Not from team names.

Important distinction:

```text
Good:
  team_idx used for joins and tracking

Risky:
  raw team_idx fed directly into the model
```

A raw index can become a memorization shortcut. If team identity is tested later, use it carefully as a categorical embedding and compare it against a no-identity model.

---

### 1.4 The LLM is not the final predictor

The LLM should **not** be asked:

```text
Who will win this game?
```

Instead, the LLM should be used as a structured feature extractor:

```text
Recent articles/interviews/injury notes
        ↓
LLM extraction
        ↓
Structured sentiment/team-state features
        ↓
Main prediction model
```

The main prediction model should be responsible for learning whether these LLM features actually help.

---

### 1.5 Ensure reproducibility

Every experiment should be reproducible:

```text
Pin random seeds: Python, NumPy, PyTorch
Version data snapshots with download dates
Store feature configs as versioned YAML files
Log git commit hash with every experiment run
Use deterministic operations where possible
```

Data reproducibility:

```text
Cache all raw API responses as Parquet snapshots.
Never re-fetch historical data that has already been downloaded.
Tag each data snapshot with a date and source version.
```

Model reproducibility:

```text
Every training run should log:
  random_seed
  git_commit
  data_snapshot_id
  feature_config_version
  hyperparameters
  environment_info
```

---

## 2. High-Level System Architecture

```text
                         ┌────────────────────────┐
                         │ Historical NBA data     │
                         │ games, box scores,      │
                         │ advanced stats          │
                         └───────────┬────────────┘
                                     ↓
                            Team sequence builder
                                     ↓
                            Team performance encoder


                         ┌────────────────────────┐
                         │ Injury / availability  │
                         │ official reports,      │
                         │ player status          │
                         └───────────┬────────────┘
                                     ↓
                              Injury impact encoder


                         ┌────────────────────────┐
                         │ Recent news/articles   │
                         │ interviews, reports,   │
                         │ team mood, pressure    │
                         └───────────┬────────────┘
                                     ↓
                            LLM sentiment extractor
                                     ↓
                              News/team-spirit encoder


                         ┌────────────────────────┐
                         │ Schedule context       │
                         │ rest, travel, home,    │
                         │ back-to-back           │
                         └───────────┬────────────┘
                                     ↓
                               Context encoder


       performance state
     + injury state
     + news/team-spirit state
     + context state
     + home-away differences
                 ↓
          Matchup fusion model
                 ↓
          Calibration layer
                 ↓
          P(home team wins)
```

---

## 3. Data Sources

### 3.1 Historical games and box scores

Primary source candidates:

1. `nba_api`
   - Python client for NBA.com stats endpoints.
   - Useful for schedules, game finder, box scores, play-by-play, team/player information.

2. Basketball Reference
   - Good for validation and historical tables.
   - Useful for schedules, standings, team game logs, player game logs, advanced stats.

3. hoopR / sportsdataverse
   - Useful for ready-made basketball datasets and play-by-play style data.

Recommended first source:

```text
nba_api
```

because it integrates naturally into a Python pipeline.

However, `nba_api` is an unofficial scraper of NBA.com endpoints. These endpoints change without notice and are rate-limited. Mitigations:

```text
Abstract all data fetching behind a DataProvider interface.
First implementation wraps nba_api.
If nba_api breaks, swap in a Basketball Reference scraper or manual CSV import.
Cache all raw API responses as Parquet snapshots.
Historical data should never need re-fetching.
```

Rate limiting and retry strategy:

```text
Maximum 1 request per second to NBA.com endpoints.
Exponential backoff on 429/5xx errors: 2s, 4s, 8s, 16s, max 60s.
Maximum 3 retries per request.
Log all failed requests for manual review.
Validate response schemas before saving.
```

DataProvider interface:

```python
class DataProvider(ABC):
    def fetch_games(self, season: str) -> pd.DataFrame: ...
    def fetch_team_game_logs(self, season: str) -> pd.DataFrame: ...
    def fetch_box_scores(self, game_id: str) -> pd.DataFrame: ...
    def fetch_player_info(self, season: str) -> pd.DataFrame: ...
```

Define this interface in Phase 0. Implement the `nba_api` version first.

---

### 3.2 Play-by-play and possession-level data

Useful later, not required for MVP.

Possible sources:

```text
nba_api PlayByPlayV2
pbpstats
sportsdataverse / hoopR
```

Why this matters:

```text
Possession-level data can produce better features:
- lineup performance
- garbage-time filtering
- pace-adjusted strength
- clutch context
- shot-quality approximations
```

This should be Phase 3 or Phase 4, not the first milestone.

---

### 3.3 Injury and availability data

Structured injury data should be treated separately from news sentiment.

Hard injury features should come from official or structured reports:

```text
Official NBA Injury Report
ESPN injuries
CBS injuries
RotoWire or similar injury aggregators, if legally and technically accessible
```

For this project, injury features should describe the **basketball impact** of missing/limited players:

```text
players_out_count
players_questionable_count
star_player_out
top_3_usage_players_out
starter_out_count
rotation_minutes_missing
usage_missing
estimated_plus_minus_missing
minutes_restriction_flag
```

---

### 3.4 Recent news and sentiment data

Potential sources:

```text
GDELT
NewsAPI
ESPN NBA
NBA.com
CBS Sports
Yahoo Sports
team official sites
RSS feeds
local beat reporters where accessible
```

The news layer should collect articles before a game and convert them into structured numerical features using an LLM.

Important:

```text
Article collection should be timestamped.
Only articles published before game start may be used.
```

#### Historical news backfill strategy

Decision: do not attempt to backfill news articles for old seasons.

```text
Rationale:
  Retroactive scraping is legally questionable and unreliable.
  Synthetic backfill would introduce fake signal.
  Old article archives have inconsistent availability.

Approach:
  For seasons before 2023-24: use a zero news vector + news_available=0 flag.
  The model must learn to predict without news features when unavailable.
  News features should improve predictions when available, not break them when absent.
  Train the news encoder only on seasons where real articles were collected.
  The fusion model receives a news_available flag per team.

News feature availability by season:
  2014-15 through 2022-23: no news features (zero vector + news_available=0)
  2023-24 onward: real news features collected and processed
```

This means news encoder ablation tests should focus on recent-season validation only.

---

## 4. Data Model

### 4.1 Core entities

```text
Team
Player
Game
TeamGameLog
PlayerGameLog
InjuryReport
NewsArticle
ArticleSentimentScore
TeamNewsAggregate
PlayerTeamAssignment
MatchupTrainingRow
Prediction
```

---

### 4.2 Team mapping

Store team identity separately:

```text
data/mappings/team_to_idx.json
data/mappings/player_to_idx.json
data/mappings/team_idx_to_display_name.json
```

Example:

```json
{
  "ATL": 0,
  "BOS": 1,
  "BKN": 2,
  "CHA": 3
}
```

The model training table should avoid team names.

---

### 4.3 Game table

```text
game_id
date
season
home_team_idx
away_team_idx
home_score
away_score
home_win
arena
city
neutral_site
start_time_utc
```

Target:

```text
home_win = 1 if home team wins, else 0
```

---

### 4.4 Team game log table

One row per team per game.

```text
game_id
date
season
team_idx
opponent_team_idx
is_home
won
points_for
points_against
point_diff
possessions
off_rating
def_rating
net_rating
pace
efg_pct
ts_pct
turnover_pct
off_rebound_pct
def_rebound_pct
free_throw_rate
assist_pct
steal_pct
block_pct
rest_days_before_game
back_to_back
games_last_7_days
games_last_14_days
opponent_strength_before_game
```

This is the base unit for team sequence modeling.

---

### 4.5 Player-team assignment table

To handle mid-season trades correctly:

```text
player_id
team_idx
start_date
end_date
source
```

Example:

```json
{
  "player_id": "203954",
  "team_idx": 4,
  "start_date": "2025-10-22",
  "end_date": "2026-02-06",
  "source": "nba_api"
}
```

When computing injury features for a game on date D, only consider players whose team assignment includes date D:

```text
start_date <= D AND (end_date IS NULL OR end_date >= D)
```

This prevents counting a traded player's injury against their former team.

---

## 5. Feature Engineering

### 5.1 Numerical performance features

For each team before a game:

```text
season_win_pct_before_game
last_5_win_pct
last_10_win_pct
season_point_diff
last_5_point_diff
last_10_point_diff
season_net_rating
last_5_net_rating
last_10_net_rating
season_off_rating
season_def_rating
last_10_off_rating
last_10_def_rating
pace
turnover_pct
rebound_pct
true_shooting_pct
effective_fg_pct
free_throw_rate
assist_pct
```

Use both raw team features and home-away differences:

```text
home_last_10_net_rating
away_last_10_net_rating
last_10_net_rating_diff
```

Difference features help the model reason relationally.

Feature normalization strategy:

```text
Use per-season StandardScaler for all numerical features.
Fit scalers on training data only. Never fit on validation or test data.
Store fitted scalers as artifacts alongside model checkpoints.
For rolling features, normalize after computing rolling windows.
Binary flags (back_to_back, etc.) do not need normalization.
```

MVP core feature set (start with these ~35 features before expanding):

```text
season_win_pct_before_game
last_5_win_pct
last_10_win_pct
season_point_diff
last_10_point_diff
season_net_rating
last_10_net_rating
season_off_rating
season_def_rating
last_10_off_rating
last_10_def_rating
pace
turnover_pct
true_shooting_pct
effective_fg_pct
free_throw_rate
assist_pct
rebound_pct
```

Expand to the full feature set listed above in Phase 4 after baselines are established.

---

### 5.2 Schedule and fatigue features

```text
home_rest_days
away_rest_days
rest_diff
home_back_to_back
away_back_to_back
home_3_games_in_4_nights
away_3_games_in_4_nights
home_games_last_7
away_games_last_7
home_games_last_14
away_games_last_14
home_away_streak
away_away_streak
home_travel_distance_since_last_game
away_travel_distance_since_last_game
timezone_change
```

Start simple:

```text
rest days
back-to-back flag
games in last 7 days
games in last 14 days
```

Add travel later.

---

### 5.3 Injury impact features

Hard injury data:

```text
players_out_count
players_doubtful_count
players_questionable_count
star_player_out
starter_out_count
rotation_player_out_count
top_3_usage_players_out
top_5_minutes_players_out
minutes_missing
usage_missing
estimated_value_missing
probable_players_count
questionable_but_expected_to_play_count
```

Possible formula for missing value:

```text
player_value_missing =
  recent_minutes_per_game
  × usage_rate
  × estimated_plus_minus_or_box_score_value
```

Team-level injury vector:

```json
{
  "players_out_count": 2,
  "players_questionable_count": 1,
  "starter_out_count": 1,
  "top_3_usage_players_out": 1,
  "minutes_missing": 42.5,
  "usage_missing": 0.31,
  "estimated_value_missing": 4.8
}
```

---

### 5.4 LLM news/team-spirit features

The LLM should output structured scores, not prose.

Possible team-news features:

```text
overall_sentiment
morale
confidence
pressure
distraction
injury_concern
team_cohesion
coach_player_tension
motivation
fatigue_mentions
returning_player_optimism
minutes_restriction_concern
media_noise
article_volume
negative_article_ratio
positive_article_ratio
sentiment_volatility
average_llm_confidence
```

Score range:

```text
-1.0 = strongly negative
 0.0 = neutral / no clear signal
+1.0 = strongly positive
```

For some features, use `0` to `1`:

```text
injury_concern: 0 to 1
distraction: 0 to 1
pressure: 0 to 1
llm_confidence: 0 to 1
```

---

## 6. LLM Sentiment Pipeline

### 6.1 News collection

For each scheduled game:

```text
For each team:
  collect articles from previous 24h, 72h, and 7d
  filter articles mentioning team, players, coach, or matchup
  remove duplicates
  store article metadata
```

Article metadata:

```text
article_id
source
url
title
published_at
retrieved_at
team_idx
game_id
matched_entities
text_hash
language
relevance_score
```

---

### 6.2 Entity resolution

Use real names only to map text to anonymous entities.

Example:

```text
"Boston Celtics" → TEAM_A
"Jayson Tatum" → PLAYER_A
"Joe Mazzulla" → COACH_A
```

Anonymized article:

```text
PLAYER_A said TEAM_A is locked in after two disappointing losses.
```

The LLM sees the anonymized version where possible.

Goal:

```text
Reduce franchise/name reputation bias.
Score the content, not the brand.
```

---

### 6.3 LLM extraction prompt

The prompt should ask for strict JSON only.

Example concept:

```text
You are extracting structured pre-game team-state signals from sports news.

Do not predict the game winner.
Do not rely on franchise reputation, player fame, or historical assumptions.
Only score the information explicitly present in the text.

Return JSON following the schema exactly.
Scores must be in the required ranges.
If there is no evidence for a field, return 0 for neutral fields and low confidence.
```

---

### 6.4 Article-level LLM output schema

Start with a core schema of 7 fields. Expand only after correlation analysis shows the extra fields add independent signal.

Core schema (MVP):

```json
{
  "team_idx": 12,
  "article_id": "abc123",
  "article_relevance": 0.82,
  "overall_sentiment": 0.20,
  "injury_concern": 0.60,
  "pressure": 0.25,
  "team_cohesion": 0.15,
  "motivation": 0.30,
  "llm_confidence": 0.78,
  "evidence_summary": "Short non-predictive explanation for debugging only."
}
```

Extended schema (add after validating core fields help):

```text
morale
confidence
distraction
coach_player_tension
fatigue_mentions
returning_player_optimism
minutes_restriction_concern
```

Many of the extended fields are likely correlated with the core fields (morale ↔ overall_sentiment, confidence ↔ motivation). Run PCA or correlation analysis before adding them as independent features.

For training features, use the numeric fields. Keep `evidence_summary` for debugging, not model input.

---

### 6.5 Aggregating article sentiment

Each article should have a weight:

```text
article_weight =
  source_quality_weight
  × recency_weight
  × relevance_score
  × llm_confidence
```

Example recency weights:

```text
0-12 hours before game: high
12-24 hours before game: medium-high
24-72 hours before game: medium
3-7 days before game: low
```

Aggregate per team per game:

```text
weighted_avg_sentiment_24h
weighted_avg_sentiment_72h
weighted_avg_morale_24h
weighted_avg_morale_72h
weighted_avg_injury_concern_24h
article_volume_24h
negative_article_ratio_72h
sentiment_volatility_72h
avg_llm_confidence
```

---

### 6.6 LLM cost estimation

Estimated token usage per extraction:

```text
Input: ~500 tokens per article (title + truncated body)
Output: ~200 tokens per structured JSON response
Total per article: ~700 tokens
```

Estimated volume:

```text
Articles per team per game: ~10-20
Teams per game: 2
Games per season: ~1,230
Articles per season: ~25,000 to 50,000
Tokens per season: ~17M to 35M tokens
```

Cost control strategy:

```text
Use a cost-efficient model: GPT-4o-mini, Claude Haiku, or similar.
At ~$0.15 per 1M input tokens and ~$0.60 per 1M output tokens:
  Input cost per season: ~$1.90 to $3.75
  Output cost per season: ~$3.00 to $6.00
  Total per season: ~$5 to $10

For 3 seasons of collection (2023-24, 2024-25, 2025-26): ~$15 to $30 total.
This is affordable. Budget $50 for the full project including retries and experiments.
```

Cost reduction options if needed:

```text
Batch API calls where available (50% discount on most providers).
Filter low-relevance articles before LLM extraction.
Cache LLM responses keyed by article text hash.
Use shorter article truncation.
```

---

## 7. Model Architecture

### 7.1 Baseline models

Even though the project is ambitious, baselines are mandatory.

Baseline models:

```text
Home-team baseline
Elo model
Logistic regression on basic features
XGBoost/LightGBM tabular model
```

Purpose:

```text
Prove the advanced model is actually better.
Detect data leakage.
Establish minimum performance.
```

XGBoost should not be the final ambition, but it is a strong benchmark.

---

### 7.2 Powerful model: team sequence + injury + news fusion

Recommended main architecture:

```text
Home recent game sequence
        ↓
Shared Team Performance Encoder
        ↓
home_performance_state


Away recent game sequence
        ↓
Shared Team Performance Encoder
        ↓
away_performance_state


Home injury vector
        ↓
Shared Injury Encoder
        ↓
home_injury_state


Away injury vector
        ↓
Shared Injury Encoder
        ↓
away_injury_state


Home news vector
        ↓
Shared News Encoder
        ↓
home_news_state


Away news vector
        ↓
Shared News Encoder
        ↓
away_news_state


Game context vector
        ↓
Context Encoder
        ↓
context_state


Concatenate:
[
  home_performance_state,
  away_performance_state,
  home_performance_state - away_performance_state,
  home_performance_state * away_performance_state,

  home_injury_state,
  away_injury_state,
  home_injury_state - away_injury_state,

  home_news_state,
  away_news_state,
  home_news_state - away_news_state,

  context_state
]

        ↓
Matchup Fusion Network
        ↓
Sigmoid
        ↓
P(home win)
```

---

### 7.3 Team performance encoder options

#### Option A: GRU encoder

```text
last N games → GRU → team_state_vector
```

Pros:

```text
Good for time series
Less likely to overfit than Transformer
Easier to debug
Good first neural model
```

#### Option B: Temporal Convolutional Network

```text
last N games → 1D temporal convolutions → team_state_vector
```

Pros:

```text
Fast
Stable
Strong on smaller time-series datasets
```

#### Option C: Small Transformer encoder

```text
last N games → Transformer Encoder → attention pooling → team_state_vector
```

Pros:

```text
Learns which past games matter most
Can capture non-local sequence patterns
Most flexible
```

Recommended path:

```text
Start with GRU or TCN.
Then upgrade to a small Transformer.
```

Do not start with a huge Transformer. NBA game data is not big enough to justify a massive architecture.

---

### 7.3.1 Sequence padding and masking

Not all teams have N previous games available:

```text
Season openers: 0 previous games
Early season (game 5): only 4 previous games available
All-Star break: gap in schedule but not in game count
```

Padding strategy:

```text
Use zero-padding for missing game slots.
Provide a binary mask tensor alongside the sequence.
GRU: mask is applied to ignore padded timesteps.
Transformer: mask prevents attention to padded positions.
```

Sequence construction:

```text
For a team with K < N previous games:
  Sequence = [zero_pad] * (N - K) + [game_K, game_K-1, ..., game_1]
  Mask = [0] * (N - K) + [1] * K

For a team with K >= N previous games:
  Sequence = [game_N, game_N-1, ..., game_1]
  Mask = [1] * N
```

Optional warm-start: for season openers, use the last M games from the previous season with a `cross_season_flag=1` indicator. Test whether this helps versus pure zero-padding.

---

### 7.4 Shared weights for home and away teams

Use the same encoder for both teams:

```python
home_state = TeamEncoder(home_recent_games)
away_state = TeamEncoder(away_recent_games)
```

Do not use separate encoders:

```python
home_state = HomeEncoder(home_recent_games)
away_state = AwayEncoder(away_recent_games)
```

Reason:

```text
A team’s quality should be represented consistently.
Home-court advantage should be captured in context features.
```

---

### 7.5 Optional anonymous team embeddings

Later, test identity-aware embeddings:

```text
team_idx → embedding vector
```

But make this an experiment, not the default.

Models to compare:

```text
Model A: no team identity
Model B: anonymous team embedding
```

Keep the embedding small and regularized:

```text
embedding_dim: 4 to 8
dropout: yes
weight decay: yes
```

If identity embeddings improve future-season validation without hurting calibration, keep them.

---

## 8. Input and Output Schemas

### 8.1 Training row

```json
{
  "game_id": "0022500451",
  "date": "2026-01-15",
  "season": "2025-26",
  "home_team_idx": 4,
  "away_team_idx": 17,

  "home_recent_games_sequence": [[...], [...], "..."],
  "away_recent_games_sequence": [[...], [...], "..."],

  "home_injury_vector": [...],
  "away_injury_vector": [...],

  "home_news_vector": [...],
  "away_news_vector": [...],

  "context_vector": [...],

  "target_home_win": 1
}
```

Shapes:

```text
home_recent_games_sequence: [N, F_game]
away_recent_games_sequence: [N, F_game]
home_injury_vector: [F_injury]
away_injury_vector: [F_injury]
home_news_vector: [F_news]
away_news_vector: [F_news]
context_vector: [F_context]
```

Recommended initial values:

```text
N = 20 previous games
F_game = 20-40
F_injury = 8-20
F_news = 8-20
F_context = 8-20
```

---

### 8.2 Prediction input

```json
{
  "game_id": "0022500451",
  "date": "2026-01-15T19:30:00Z",
  "home_team_idx": 4,
  "away_team_idx": 17,

  "home_recent_games_sequence": [],
  "away_recent_games_sequence": [],

  "home_injury_vector": {},
  "away_injury_vector": {},

  "home_news_vector": {},
  "away_news_vector": {},

  "context_vector": {}
}
```

---

### 8.3 Prediction output

```json
{
  "game_id": "0022500451",
  "generated_at": "2026-01-15T15:00:00Z",
  "model_version": "team_sequence_news_fusion_v1",

  "home_team_idx": 4,
  "away_team_idx": 17,

  "home_win_probability": 0.64,
  "away_win_probability": 0.36,
  "predicted_winner": "home_team_idx_4",
  "confidence_bucket": "medium",

  "component_outputs": {
    "elo_probability": 0.58,
    "tabular_probability": 0.61,
    "sequence_model_probability": 0.66,
    "final_probability": 0.64
  },

  "top_model_factors": [
    "home team has stronger recent net rating",
    "away team is on short rest",
    "away injury concern is elevated",
    "home team news sentiment is slightly positive"
  ]
}
```

The `top_model_factors` field is generated using:

```text
Tabular models: SHAP values (TreeExplainer for XGBoost/LightGBM).
Neural models: input gradient attribution or attention weights.
Ensemble: weighted combination of component-level attributions.

Fallback for MVP: rule-based templates.
  Compare feature values to season averages.
  Flag features more than 1 standard deviation from mean.
  Example: if rest_diff > 1σ → "away team is on short rest"
```

Start with the rule-based fallback. Add SHAP in Phase 8 when building the daily prediction system.

---

## 9. Training and Evaluation

### 9.1 Use time-based splits

Do not use random train/test split.

Example:

```text
Train:
2014-15 through 2021-22

Validation:
2022-23

Test:
2023-24 and 2024-25
```

Better: rolling validation.

```text
Train 2014-2019 → validate 2020
Train 2014-2020 → validate 2021
Train 2014-2021 → validate 2022
Train 2014-2022 → validate 2023
```

This simulates real forecasting.

#### Playoff vs regular season

```text
Decision: V1 is regular season only.

Rationale:
  Playoff games have different dynamics (7-game series, higher intensity,
  longer rest, different rotations).
  Mixing playoff and regular season data can confuse the model.
  Regular season provides ~1,230 games/season vs ~80 playoff games.

Future option (Phase 10+):
  Train a separate playoff model or add a playoff_flag feature.
  Test whether regular-season-trained model generalizes to playoffs.
```

#### Performance by season phase

Track and report metrics broken down by month:

```text
October/November: small sample, roster experimentation
December-February: mid-season stability
February trade deadline: roster disruption
March-April: tanking, resting starters, playoff positioning
```

A good model should know its own weak spots across the season.

---

### 9.2 Metrics

Use several metrics:

```text
Accuracy
Log loss
Brier score
ROC-AUC
Calibration error
Expected calibration error
Performance by confidence bucket
Performance on close games
Performance on injury-uncertain games
Performance on high-news-volume games
```

Accuracy is not enough.

Most important:

```text
Log loss
Brier score
Calibration
```

Because this is a probability model.

---

### 9.3 Calibration

After training, calibrate probabilities.

Options:

```text
Platt scaling
Isotonic regression
Temperature scaling for neural model
```

Check whether:

```text
Games predicted at 60% actually win around 60% of the time.
Games predicted at 70% actually win around 70% of the time.
```

---

### 9.4 Ablation testing

This is critical for proving whether the LLM/news layer helps.

Train and compare:

```text
Model A: stats only
Model B: stats + schedule
Model C: stats + schedule + injuries
Model D: stats + schedule + news sentiment
Model E: stats + schedule + injuries + news sentiment
Model F: full ensemble
```

Questions to answer:

```text
Does news sentiment improve log loss?
Does injury data improve calibration?
Does the sequence model beat tabular features?
Does the LLM layer help mostly in upset games?
Does it help mostly when injury uncertainty is high?
```

If sentiment sounds impressive but does not improve validation metrics, downweight it or remove it.

---

## 10. Ensemble Strategy

The strongest final system should probably be an ensemble:

```text
Elo model
Tabular model
Neural sequence model
Injury-aware model
News/sentiment-aware model
```

Meta-model input:

```text
elo_probability
tabular_probability
sequence_probability
injury_features_summary
news_features_summary
model_confidence_features
```

Meta-model options:

```text
Logistic regression
Small MLP
Calibrated gradient boosting
```

Recommended first ensemble:

```text
Logistic regression meta-model
```

Reason:

```text
Simple
Interpretable
Harder to overfit
Good for combining probabilities
```

---

## 11. Repository Structure

```text
nba-outcome-model/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── .gitignore
│
├── configs/
│   ├── data_sources.yaml
│   ├── model_config.yaml
│   ├── feature_config.yaml
│   └── llm_sentiment_schema.json
│
├── data/
│   ├── raw/
│   │   ├── nba_api/
│   │   ├── injuries/
│   │   └── news/
│   │
│   ├── interim/
│   │   ├── cleaned_games/
│   │   ├── cleaned_boxscores/
│   │   └── anonymized_articles/
│   │
│   ├── processed/
│   │   ├── team_game_logs/
│   │   ├── injury_features/
│   │   ├── news_features/
│   │   └── matchup_rows/
│   │
│   └── mappings/
│       ├── team_to_idx.json
│       ├── player_to_idx.json
│       └── source_quality_weights.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_elo_baseline.ipynb
│   ├── 03_sequence_dataset_debug.ipynb
│   ├── 04_news_sentiment_debug.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── providers/
│   │   │   ├── base.py
│   │   │   └── nba_api_provider.py
│   │   ├── fetch_games.py
│   │   ├── fetch_boxscores.py
│   │   ├── fetch_play_by_play.py
│   │   ├── fetch_injuries.py
│   │   ├── fetch_news.py
│   │   └── clean_data.py
│   │
│   ├── anonymization/
│   │   ├── team_mapping.py
│   │   ├── player_mapping.py
│   │   └── anonymize_articles.py
│   │
│   ├── nlp/
│   │   ├── article_filtering.py
│   │   ├── extract_sentiment.py
│   │   ├── sentiment_schema.py
│   │   └── aggregate_team_news.py
│   │
│   ├── features/
│   │   ├── build_team_game_logs.py
│   │   ├── rolling_features.py
│   │   ├── schedule_features.py
│   │   ├── injury_features.py
│   │   ├── news_features.py
│   │   ├── sequence_builder.py
│   │   └── build_matchup_dataset.py
│   │
│   ├── models/
│   │   ├── elo.py
│   │   ├── tabular_model.py
│   │   ├── team_encoder.py
│   │   ├── injury_encoder.py
│   │   ├── news_encoder.py
│   │   ├── matchup_fusion_model.py
│   │   ├── ensemble.py
│   │   ├── calibrate.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── api.py
│   │
│   └── utils/
│       ├── time.py
│       ├── logging.py
│       ├── validation.py
│       └── paths.py
│
├── models/
│   ├── baselines/
│   ├── neural/
│   ├── calibrators/
│   └── ensembles/
│
├── predictions/
│   ├── daily/
│   └── historical_backtests/
│
├── tests/
│   ├── test_no_leakage.py
│   ├── test_sequence_builder.py
│   ├── test_anonymization.py
│   ├── test_sentiment_schema.py
│   └── test_prediction_shapes.py
│
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    ├── leakage_rules.md
    ├── llm_sentiment_design.md
    └── evaluation_plan.md
```

Makefile targets:

```makefile
fetch-data:     # Download raw data from nba_api
build-features: # Build leakage-safe feature tables
train-baseline: # Train Elo + tabular baselines
train-model:    # Train neural sequence model
evaluate:       # Run full evaluation suite
predict-today:  # Generate today's predictions
test:           # Run all tests including leakage checks
lint:           # Run linting and type checks
```

---

## 12. Implementation Roadmap

### Phase 0 — Project setup [S — 1 to 2 days]

Deliverables:

```text
Repository structure
Python environment
Config files
DataProvider interface + nba_api implementation
Data folders
Logging setup
Makefile
Basic tests
README
```

Goal:

```text
A clean, professional codebase from day one.
```

---

### Phase 1 — Historical data foundation [M — 3 to 5 days]

Deliverables:

```text
Fetch historical games (2014-15 through current)
Fetch team box scores
Normalize team IDs
Create anonymous team mapping
Build player-team assignment table
Build game table
Build team game logs
Cache all raw responses as Parquet snapshots
```

Success condition:

```text
Can generate one clean historical dataset with one row per team per game.
```

---

### Phase 2 — Leakage-safe feature table [M — 3 to 5 days]

Deliverables:

```text
Rolling features before each game
Rest/back-to-back features
Home/away splits
Basic context features
Matchup training rows
```

Success condition:

```text
For every game, all features are computed only from prior games.
```

Add tests:

```text
test_no_future_games_used
test_no_target_boxscore_used
test_rolling_features_shifted_correctly
```

---

### Phase 3 — Baselines [M — 3 to 5 days]

Deliverables:

```text
Home-team baseline
Elo baseline
Logistic regression baseline
XGBoost/LightGBM benchmark
Evaluation script
Calibration plots
```

Success condition:

```text
Reliable benchmark metrics on time-based validation.
```

---

### Phase 4 — Team sequence model [L — 1 to 2 weeks]

Deliverables:

```text
Sequence dataset builder
GRU or TCN team encoder
Matchup fusion head
Training loop
Evaluation against baselines
```

Success condition:

```text
Neural sequence model is competitive with or better than tabular baseline.
```

Do not worry if it does not beat XGBoost immediately. The first goal is a correct architecture.

---

### Phase 5 — Injury features [M — 3 to 5 days]

Deliverables:

```text
Injury data ingestion
Player availability table
Player value estimation
Team injury vector
Injury encoder
Ablation test with and without injuries
```

Success condition:

```text
Injury features improve log loss or calibration, especially in games with missing key players.
```

---

### Phase 6 — LLM news/sentiment layer [XL — 2 to 3 weeks]

Deliverables:

```text
News article collector
Entity resolver
Article anonymizer
LLM structured extraction
Article-level sentiment table
Team-level sentiment aggregation
News feature vector
Ablation test
```

Success condition:

```text
News sentiment features improve validation metrics or specific subsets like upset games, high-pressure games, or injury-uncertain games.
```

---

### Phase 7 — Full fusion model [L — 1 to 2 weeks]

Deliverables:

```text
Performance encoder
Injury encoder
News encoder
Context encoder
Fusion model
Calibration layer
Model comparison
```

Success condition:

```text
Full model beats baselines on log loss/Brier score without becoming miscalibrated.
```

---

### Phase 8 — Daily prediction system [M — 3 to 5 days]

Deliverables:

```text
predict_today.py
daily feature generation
daily predictions JSON
prediction storage
basic dashboard
```

Output:

```text
Today’s games
Home win probability
Away win probability
Confidence bucket
Top model factors
```

---

### Phase 9 — Product/dashboard [L — 1 to 2 weeks]

Deliverables:

```text
Streamlit dashboard
Historical prediction archive
Model performance view
Calibration charts
Game detail pages
Source/debug view for news sentiment
```

Dashboard sections:

```text
Today’s predictions
Model confidence
Team form
Injury impact
News/team-spirit signal
Backtest performance
```

---

## 13. Technical Stack

Recommended stack:

```text
Python
pandas
numpy
scikit-learn
PyTorch
LightGBM or XGBoost for benchmarks
nba_api
requests/httpx
pydantic
polars optional
MLflow or Weights & Biases optional
Streamlit
FastAPI optional
SQLite/Postgres optional
```

For LLM structured extraction:

```text
Primary model: GPT-4o-mini (best cost/quality ratio for structured extraction).
Fallback: Claude 3.5 Haiku or Gemini Flash.
Use strict JSON mode / structured output where available.
Temperature: 0 for deterministic, reproducible extraction.
Validate every response with Pydantic.
Retry invalid outputs up to 3 times.
Store raw and parsed outputs.
Log model name + version alongside every extraction.
```

LLM consistency validation:

```text
Create a small human-labeled validation set (~50 articles).
Score LLM extraction quality against human labels.
Re-run validation when switching models or updating prompts.
Track inter-run agreement on the same articles.
```

---

## 14. Database / Storage Design

For MVP:

```text
Parquet files
CSV only for exports
JSON for mappings and configs
```

Suggested:

```text
data/processed/team_game_logs.parquet
data/processed/matchup_rows.parquet
data/processed/injury_features.parquet
data/processed/news_features.parquet
data/processed/article_sentiment_scores.parquet
```

Later:

```text
PostgreSQL
```

Tables:

```text
games
teams
players
team_game_logs
player_game_logs
injury_reports
news_articles
article_sentiment_scores
team_news_aggregates
predictions
model_runs
```

---

## 15. Experiment Tracking

Every model run should store:

```text
model_version
git_commit
training_start_date
training_end_date
validation_period
test_period
feature_config
data_snapshot
hyperparameters
metrics
calibration_metrics
notes
```

This prevents confusion when experiments multiply.

Suggested tools:

```text
MLflow
Weights & Biases
simple JSON experiment logs for MVP
```

---

## 16. Risk Register

### 16.1 Data leakage

Risk:

```text
Model accidentally uses future information.
```

Mitigation:

```text
Strict feature timestamping
Unit tests
Time-based validation
Shift all rolling features by one game
```

---

### 16.2 LLM sentiment is noisy

Risk:

```text
LLM-generated scores sound useful but do not improve predictions.
```

Mitigation:

```text
Ablation tests
Confidence-weighted aggregation
Small number of interpretable sentiment features
Do not let LLM predict winner
```

---

### 16.3 Team identity memorization

Risk:

```text
Model memorizes anonymous team IDs.
```

Mitigation:

```text
Do not feed team_idx into V1 model
Test identity embeddings separately
Use future-season validation
```

---

### 16.4 Overfitting

Risk:

```text
Powerful model learns noise from a limited dataset.
```

Mitigation:

```text
Small neural models first
Dropout
Weight decay
Early stopping
Rolling validation
Baselines
Calibration checks
```

---

### 16.5 News source bias

Risk:

```text
Some teams receive more coverage, creating volume bias.
```

Mitigation:

```text
Use article volume as a feature
Normalize by expected coverage
Use source weights
Compare large-market vs small-market performance
```

---

### 16.6 Missing injury/news data

Risk:

```text
Some games have incomplete external data.
```

Mitigation:

```text
Use missingness indicators
Fallback to neutral sentiment
Fallback to no-injury-known state
Track data availability
```

---

## 17. First MVP Specification

The first serious MVP should include:

```text
Anonymous team IDs
Historical game data
Team game logs
Rolling performance features
Rest/back-to-back features
Elo baseline
Tabular benchmark
Sequence model with GRU or TCN
Time-based validation
Basic calibrated probabilities
```

Do not include news sentiment in the very first MVP. Add it once the base pipeline is stable.

MVP output:

```json
{
  "game_id": "0022500451",
  "home_team_idx": 4,
  "away_team_idx": 17,
  "home_win_probability": 0.61,
  "away_win_probability": 0.39,
  "confidence": "medium",
  "model_version": "sequence_mvp_v1"
}
```

---

## 18. Second MVP Specification: LLM Sentiment

After the numerical model works, add:

```text
News collection
Article anonymization
LLM structured extraction
Team-level news aggregation
News feature vector
Ablation testing
```

Goal:

```text
Prove whether team-spirit/news sentiment adds predictive value.
```

The LLM layer should be judged by validation performance, not vibes.

---

## 19. Definition of Success

A strong project result would be:

```text
Clean data pipeline
No-leakage validation
Model beats home-team and Elo baselines
Model produces calibrated probabilities
Injury layer improves uncertainty handling
LLM sentiment layer is measurable through ablation tests
Daily predictions can be generated automatically
Dashboard explains model output
```

Target realistic performance:

```text
Accuracy: roughly mid-60s can already be solid
Log loss: should beat Elo/tabular baselines
Calibration: predicted probabilities should match empirical win rates
```

Do not chase unrealistic accuracy. In sports, a well-calibrated 60-65% model can be genuinely strong.

---

## 20. Immediate Next Steps

Recommended next technical steps:

```text
1. Create repository structure.
2. Build team/team_idx mapping.
3. Fetch historical schedules and results.
4. Build team game log table.
5. Implement leakage-safe rolling features.
6. Train Elo baseline.
7. Train tabular benchmark.
8. Build sequence dataset.
9. Train first GRU/TCN model.
10. Evaluate with time-based split.
```

After that:

```text
11. Add structured injury features.
12. Add news collection.
13. Add LLM sentiment extraction.
14. Run ablation tests.
15. Build dashboard.
```

---

## 21. Architecture Summary

Final intended architecture:

```text
Historical team sequences
        +
Schedule context
        +
Structured injury impact
        +
LLM-extracted news/team-spirit features
        ↓
Multi-encoder neural fusion model
        ↓
Calibrated ensemble
        ↓
Home/away win probabilities
```

Core philosophy:

```text
The model should learn what a strong anonymous team-state looks like,
who is available tonight,
what the recent emotional/news environment looks like,
and how all of that compares to the opponent.
```

That is the project.
