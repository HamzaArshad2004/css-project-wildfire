# LLM Rule Evaluation Guide

The evaluation script uses an LLM to critically score association rules and surface the most novel and crisis-relevant findings.

## Quick Start

### Without LLM (default, free)
```bash
python scripts/evaluate_rules_llm.py
```
- Rules ranked by **cross-domain priority** (mobility + emotion interactions first)
- Then by statistical strength (lift, confidence, support)
- Good for quick ranking, but doesn't filter out trivial patterns

### With LLM (recommended)
```bash
python scripts/evaluate_rules_llm.py --llm
```
- **Requires:** OpenAI API key
- Rules scored on **novelty** (1-10: how surprising) and **crisis-relevance** (1-10: how actionable)
- Primary sort: **LLM composite score** (average of novelty + crisis_relevance)
- Falls back to cross-domain priority for ties
- **Filters out** tautological and obvious rules

---

## Setup (One-time)

### 1. Get an OpenAI API key
- Go to: https://platform.openai.com/api-keys
- Create a new API key
- Copy it (you'll use it once)

### 2. Add to `.env`
```bash
# Edit .env in the project root
OPENAI_API_KEY="sk-your-key-here"
```

### 3. Verify it works
```bash
python scripts/evaluate_rules_llm.py --llm --top-n 5
```
Should see API calls starting, asking the LLM to score each rule.

---

## What the LLM Does

The LLM gets a critical prompt that tells it to:

**Score LOW (< 5) for:**
- Tautological patterns: `"sentiment_improved → sentiment_shift_detected"` (if improved, it shifted)
- Meta-feature consequences: `"emotion + mobility signal → emotion_with_mobility_signal"` (by definition)
- Feature containment: `"sentiment_worsened, high_negative_sentiment → ..."` (worsened implies negative)
- Obvious one-way implications

**Score HIGH (8-10) for:**
- Unexpected cross-domain links: emotion predicting mobility in non-obvious ways
- Temporal/contextual insights: e.g., weekend effects, timing patterns
- Policy-relevant findings that would surprise an emergency manager
- Low-support but high-confidence novel discoveries

---

## Output Comparison

### Without LLM
```
Ranking by cross-domain priority + statistical strength...

Rule 1:  [CROSS-DOMAIN]
  IF:   traffic_congestion_detected, anger_mentioned
  THEN: fear_keywords_present
  Support: 5 days (15.2%)  Confidence: 100.0%  Lift: 2.54
```
→ Just marked as cross-domain; no quality judgment.

### With LLM
```
Ranking by LLM scores (novelty + crisis_relevance)...

Rule 1:  [CROSS-DOMAIN | ★ HIGH-VALUE]
  IF:   traffic_congestion_detected, anger_mentioned
  THEN: fear_keywords_present
  Support: 5 days (15.2%)  Confidence: 100.0%  Lift: 2.54
  LLM:  novelty=8/10  crisis_relevance=7/10  (avg=7.5)
        → Traffic congestion + public anger → fear escalation. Actionable for 
          emergency communication strategies.
```
→ Scored, prioritized, and reasoned. Rules with `avg < 4` marked `⚠ LOW-VALUE`.

---

## Running Options

```bash
# Basic LLM evaluation
python scripts/evaluate_rules_llm.py --llm

# Specify model (default: gpt-4o-mini, cheap & fast)
python scripts/evaluate_rules_llm.py --llm --model gpt-4o-mini
python scripts/evaluate_rules_llm.py --llm --model gpt-4-turbo

# Change number of rules in summary (default: 15)
python scripts/evaluate_rules_llm.py --llm --top-n 20

# All together
python scripts/evaluate_rules_llm.py --llm --model gpt-4o-mini --top-n 20
```

---

## Costs

- **gpt-4o-mini** (recommended): ~$0.01-0.05 per run for 50-100 rules
- **gpt-4-turbo**: ~$0.10-0.20 per run (higher quality, slower)
- No cost if you use the default (non-LLM) ranking

---

## Integration with Pipeline

The full pipeline will ask you if you want LLM evaluation:

```bash
python scripts/run_full_pipeline.py

Enable LLM evaluation? (y/n, default: n): y
✓ OPENAI_API_KEY detected. LLM evaluation enabled.
```

Then at the end of all analysis steps, the LLM-ranked rules are generated automatically.

---

## Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure `.env` file exists in the project root
- Check that `OPENAI_API_KEY="sk-..."` is in the file (with the real key)
- Or set it as an environment variable: `export OPENAI_API_KEY="sk-..."`

### "openai package not installed"
```bash
pip install openai
```

### API errors (rate limiting, quota exceeded)
- Check your OpenAI account: https://platform.openai.com/account/usage/overview
- Ensure your API key is active (not revoked)
- Reduce rule count with `--top-n 5` to do a quick test

### Slow evaluation
- Using `gpt-4-turbo` is slower; switch to `gpt-4o-mini` (default)
- With ~60 rules and gpt-4o-mini, evaluation typically takes 2-3 minutes

---

## Output Files

After running (with or without LLM):

```
results/fca/
  ├── association_rules_evaluated.csv
  │   └── Contains: premise, conclusion, support, confidence, lift, 
  │       cross_domain, novelty_score, crisis_relevance, llm_composite, llm_reasoning
  │       (novelty/crisis columns are NaN if not using LLM)
  │
  └── top_cross_domain_rules.txt
      └── Human-readable summary with rankings and LLM reasoning
```

**Use `association_rules_evaluated.csv` for further analysis** (filtering by composite score, etc.)

**Use `top_cross_domain_rules.txt` for reporting** (formatted for stakeholders).
