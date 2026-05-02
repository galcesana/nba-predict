# Phase 6 — LLM News/Sentiment Layer

| Field | Value |
|-------|-------|
| **Size** | XL (2–3 weeks) |
| **Status** | `[ ]` Not Started |
| **Depends on** | Phase 4 |
| **Unlocks** | Phase 7 |

---

## Goal

Build the news collection, LLM-based structured sentiment extraction, and team-level aggregation pipeline. Prove via ablation whether news/team-spirit signals add predictive value.

---

## Deliverables Checklist

- [ ] `src/data/fetch_news.py` — news article collector
- [ ] `src/anonymization/anonymize_articles.py` — entity replacement
- [ ] `src/nlp/article_filtering.py` — relevance filtering
- [ ] `src/nlp/sentiment_schema.py` — Pydantic schema for LLM output
- [ ] `src/nlp/extract_sentiment.py` — LLM structured extraction
- [ ] `src/nlp/aggregate_team_news.py` — weighted aggregation
- [ ] `src/features/news_features.py` — news feature vector builder
- [ ] Article-level scores saved to `data/processed/`
- [ ] Team-level aggregates saved to `data/processed/news_features/`
- [ ] LLM extraction validation set (~50 articles, human-labeled)
- [ ] Ablation: model with news vs model without news
- [ ] Cost tracking log
- [ ] All verification tests pass

---

## Key Implementation Details

### News sources (start with 1-2, expand later)

```text
MVP: NewsAPI (limited but structured)
Later: ESPN RSS, GDELT, CBS Sports, Yahoo Sports
```

### Core LLM schema (7 fields — MVP)

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
  "evidence_summary": "debugging only"
}
```

### LLM configuration

```text
Model:       GPT-4o-mini (primary), Claude Haiku / Gemini Flash (fallback)
Temperature: 0 (deterministic)
Max retries: 3
Output:      Strict JSON mode / structured output
Cost budget: $50 total for project
```

### Weighted aggregation

```text
article_weight = source_quality × recency_weight × relevance × llm_confidence

Recency weights:
  0-12h before game:  1.0
  12-24h:             0.8
  24-72h:             0.5
  3-7d:               0.2

Aggregate per team per game:
  weighted_avg_sentiment_24h, weighted_avg_sentiment_72h,
  article_volume_24h, negative_article_ratio_72h,
  sentiment_volatility_72h, avg_llm_confidence
```

### Historical backfill decision

```text
No backfill for pre-2023-24 seasons.
Use zero vector + news_available=0 flag.
News ablation tests use 2023-24+ validation only.
```

---

## Verification Tests

Run: `pytest tests/test_sentiment.py -v`

```python
# tests/test_sentiment.py

def test_sentiment_schema_valid():
    """Pydantic schema validates correct JSON and rejects bad JSON."""

def test_schema_score_ranges():
    """overall_sentiment in [-1, 1], pressure/injury_concern in [0, 1],
    llm_confidence in [0, 1]."""

def test_article_collector_runs():
    """fetch_news.py can collect at least 1 article for a known team/date."""

def test_anonymization_replaces_names():
    """Team and player names are replaced with TEAM_A, PLAYER_A etc."""

def test_anonymization_preserves_meaning():
    """Anonymized text still contains key contextual words."""

def test_llm_extraction_returns_valid_json():
    """Run extraction on 3 test articles. Output matches Pydantic schema."""

def test_llm_extraction_deterministic():
    """Same article extracted twice produces identical scores (temp=0)."""

def test_aggregation_weights_sum():
    """Aggregation weights are > 0 and produce finite averaged scores."""

def test_news_features_shape():
    """News feature vector has expected number of columns per team."""

def test_news_available_flag():
    """Pre-2023-24 games have news_available=0 and zero news vectors."""

def test_recent_games_have_news():
    """2023-24+ games have news_available=1 and non-zero news vectors
    (for at least some games)."""

def test_no_future_articles():
    """No article has published_at after game start_time."""

def test_cost_under_budget():
    """Total LLM API cost logged and under $50."""

def test_validation_set_agreement():
    """LLM scores on 50-article validation set correlate with human labels
    (Pearson r > 0.5 on overall_sentiment)."""
```

**Expected: 14/14 pass.**

---

## Definition of Done

- [ ] All 14 verification tests pass
- [ ] News features saved to `data/processed/news_features/`
- [ ] LLM cost < $50
- [ ] Ablation results documented
- [ ] Human validation set created and scored

---

## Notes & Learnings

```
Ablation results (2023-24+ validation only):
  Model without news: log_loss=___  brier=___
  Model with news:    log_loss=___  brier=___
  Improvement:        Δ log_loss=___

LLM extraction quality:
  Human-LLM correlation (overall_sentiment): r=___
  Most reliable field: ___
  Least reliable field: ___

Total LLM cost: $___
Articles collected: ___
```
