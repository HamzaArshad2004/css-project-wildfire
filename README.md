# LA Wildfire Crisis Analysis — FCA + LLM Rule Evaluation

Formal Concept Analysis (FCA) of mobility and online emotional behavior during the **January 2025 LA wildfires**, with GPT-4o-mini batch selection of the most novel and policy-relevant cross-domain association rules.

---

## Research Question

> Can daily Reddit sentiment patterns (fear, anger, solidarity) predict or co-occur with physical mobility signals (traffic congestion, evacuation mentions) during an acute wildfire crisis?

---

## Pipeline

```
collect_mobility_data.py   ─┐
collect_reddit_data.py      ├─► preprocess_and_features.py ─► fca_analysis.py
                            │                                        │
                            └────────────────────────────────────────┘
                                                                     │
                                                            evaluate_rules_llm.py
                                                                     │
                                                            visualize_results.py
```

---

## Dataset

| Source | Description | Period |
|--------|-------------|--------|
| LA/CA Caltrans PeMS | Daily VMT, VHT, TTI, vehicle delay | Jan 2025 (33 days) |
| Reddit r/LosAngeles, r/wildfires | Wildfire posts scraped by PRAW | Jan 2025 (33 days) |

---

## Feature Definitions

### Mobility Features
| Feature | Definition |
|---------|-----------|
| `traffic_congestion_detected` | TTI or VMT deviation above baseline threshold |
| `evacuation_mentioned` | Reddit posts contain evacuation-related keywords |

### Emotion / Social Features
| Feature | Definition |
|---------|-----------|
| `high_negative_sentiment` | Daily VADER compound score < −0.1 |
| `high_positive_sentiment` | Daily VADER compound score > 0.1 |
| `fear_keywords_present` | ≥1 post contains fear-related keywords (flee, terrified, etc.) |
| `anger_mentioned` | ≥1 post contains anger-related keywords |
| `anxiety_keywords_present` | ≥1 post contains anxiety-related keywords |
| `sadness_keywords_present` | ≥1 post contains sadness/grief keywords |
| `dominant_emotion_fear` | Fear keywords appear more than any other emotion category |
| `solidarity_messages` | ≥1 post contains solidarity/support keywords |
| `policy_governance_discussion` | ≥1 post discusses government response, resources, or policy |
| `mixed_emotions` | Multiple distinct emotion categories present on same day |
| `sentiment_shift_detected` | Compound score shifts > 0.15 from previous day |
| `sentiment_improved` | Compound score increases > 0.15 |
| `sentiment_worsened` | Compound score decreases > 0.15 |

### Composite (Derived) Features
| Feature | Definition |
|---------|-----------|
| `emotion_with_mobility_signal` | Any emotion feature AND any mobility feature both active |
| `emotion_mobility_mismatch` | Any emotion feature active but NO mobility feature |
| `low_emotion_low_mobility_signal` | Neither emotion nor mobility features active |
| `weekend` | Day of week is Saturday or Sunday |

---

## Methods

### 1. Formal Concept Analysis
- Binary matrix: 33 days × 19 features
- **`concepts`** library builds the Galois lattice
- Association rules extracted with: `min_support=0.10`, `min_confidence=0.80`, `min_lift=1.05`, `max_premise_size=2`, `max_conclusion_prevalence=0.75`
- Tautological rules filtered via `TAUTOLOGY_DEFINITIONS` (rules whose conclusions follow definitionally from composite feature premises)

### 2. LLM Batch Evaluation
- All 47 rules sent to `gpt-4o-mini` in a single call
- System prompt instructs the LLM to select the top 20 most:
  - **Novel** (would surprise an emergency manager)
  - **Cross-domain** (mobility ↔ emotion)
  - **Policy-actionable**
- LLM returns `novelty_score`, `policy_score`, `reasoning`, `policy_recommendation` for each selected rule

---

## Key Findings

| Rule | Support | Confidence | Lift | Novelty | Policy |
|------|---------|-----------|------|---------|--------|
| `traffic_congestion + anger → fear_keywords` | 5 days | 100% | 2.54 | 8/10 | 7/10 |
| `traffic_congestion + sadness → policy_governance_discussion` | 6 days | 83% | 1.72 | 7/10 | 8/10 |
| `traffic_congestion + anger → policy_governance_discussion` | 5 days | 80% | 1.65 | 7/10 | 8/10 |
| `fear + anger → traffic_congestion` | 5 days | 100% | 2.30 | 6/10 | 6/10 |
| `anger + policy_governance → traffic_congestion` | 5 days | 100% | 2.30 | 6/10 | 6/10 |

> Cross-domain rules reveal that traffic congestion co-occurs with escalating emotional distress (anger → fear, sadness → policy demands), suggesting mobility stress as an early-warning signal for community sentiment deterioration.

---

## Setup

### 1. Environment
```bash
git clone https://github.com/HamzaArshad2004/capstone-project.git
cd capstone-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Credentials
```bash
cp .env.example .env
# Edit .env — add your Reddit API credentials and OpenAI key:
#   REDDIT_CLIENT_ID=...
#   REDDIT_CLIENT_SECRET=...
#   REDDIT_USER_AGENT=...
#   OPENAI_API_KEY=sk-...
```

### 3. Run full pipeline
```bash
# Data collection (optional — raw data already in data/raw/)
python scripts/collect_mobility_data.py
python scripts/collect_reddit_data.py

# Feature engineering
python scripts/preprocess_and_features.py

# FCA + association rules
python scripts/fca_analysis.py

# LLM evaluation (requires OPENAI_API_KEY)
python scripts/evaluate_rules_llm.py --llm

# Visualizations + summary report
python scripts/visualize_results.py
```

Or run everything at once:
```bash
make all
```

---

## Project Structure

```
capstone-project/
├── data/
│   ├── raw/                        # LA mobility CSV + Reddit JSON
│   └── processed/                  # FCA binary matrix, features CSV
├── results/
│   ├── fca/
│   │   ├── association_rules.csv           # All mined rules
│   │   ├── association_rules_evaluated.csv # Rules with LLM scores
│   │   ├── top_cross_domain_rules.txt      # Human-readable top rules
│   │   ├── formal_concepts.csv
│   │   └── galois_lattice.png
│   ├── visualizations/             # Mobility, sentiment, heatmap plots
│   └── summary_report.txt          # Auto-generated analysis summary
├── scripts/
│   ├── collect_mobility_data.py
│   ├── collect_reddit_data.py
│   ├── preprocess_and_features.py
│   ├── fca_analysis.py             # Core FCA + tautology filtering
│   ├── evaluate_rules_llm.py       # LLM batch evaluation
│   └── visualize_results.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `concepts` (FCA lattice)
- `praw` (Reddit scraping)
- `vaderSentiment` (sentiment scoring)
- `openai>=1.0.0` (LLM evaluation — requires `OPENAI_API_KEY`)
