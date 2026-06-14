"""
LLM-based rule annotation (NOT selection or ranking).

Loads association rules from FCA. Rule validity is determined ENTIRELY by the
statistical pipeline (support, confidence, lift) and the tautology filter in
fca_analysis.py. The optional LLM step is a post-hoc QUALITATIVE ANNOTATION
layer only: it scores each already-valid rule for policy relevance and writes a
plain-language rationale and a named-agency recommendation. It never selects,
ranks, or validates rules.

Usage (without LLM -- statistical ordering only):
    python scripts/evaluate_rules_llm.py

Usage (with LLM annotation -- requires OPENAI_API_KEY in .env or environment):
    python scripts/evaluate_rules_llm.py --llm

Output:
    results/fca/association_rules_evaluated.csv   (all rules, annotated if --llm)
    results/fca/top_cross_domain_rules.txt        (human-readable summary)
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules.csv"
OUT_CSV = PROJECT_ROOT / "results" / "fca" / "association_rules_evaluated.csv"
OUT_TXT = PROJECT_ROOT / "results" / "fca" / "top_cross_domain_rules.txt"

# ---------------------------------------------------------------------------
# Feature domain classification (must stay in sync with fca_analysis.py)
# ---------------------------------------------------------------------------
MOBILITY_FEATURES = frozenset({
    "traffic_congestion_detected",
    "evacuation_mentioned",
})

EMOTION_FEATURES = frozenset({
    "high_negative_sentiment",
    "dominant_emotion_fear",
    "fear_keywords_present",
    "anger_mentioned",
    "anxiety_keywords_present",
    "sadness_keywords_present",
    "high_positive_sentiment",
    "mixed_emotions",
    "solidarity_messages",
    "sentiment_worsened",
    "sentiment_improved",
    "sentiment_shift_detected",
    "policy_governance_discussion",
})

ANNOTATE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert data scientist and emergency management policy advisor
    annotating association rules mined from LA Wildfire 2025 crisis behavior data
    (January 3 – February 4, 2025). The dataset covers 33 days and combines
    daily LA traffic metrics (VMT, TTI) with Reddit sentiment signals from
    local subreddits.

    IMPORTANT: These rules have ALREADY been validated by a statistical pipeline
    (support, confidence, lift thresholds) and a tautology filter. Your job is
    NOT to select, rank, validate, or reject any rule. Every rule you are given
    is statistically valid. You provide ONLY a qualitative annotation for each:
    a policy-relevance score, a short rationale, and a named-agency recommendation.
    Score and annotate EVERY rule provided. Do not omit any.

    ── LA WILDFIRE 2025 CRISIS TIMELINE ──────────────────────────────────────
    Phase 1 — Pre-ignition (Jan 3–6):
      Baseline conditions; routine traffic; no elevated fear signals.
    Phase 2 — Ignition & rapid spread (Jan 7–11):
      Palisades and Eaton fires ignite Jan 7; mandatory evacuation orders;
      traffic congestion spikes; fear and anger peak on Reddit.
    Phase 3 — Peak crisis (Jan 12–20):
      Largest active footprint; ~180,000 residents under evacuation orders;
      TTI and VHT at maximum; solidarity messaging surges.
    Phase 4 — Containment & recovery (Jan 21–Feb 4):
      Progressive containment; return-home orders issued; VMT normalises;
      positive sentiment begins to recover.
    ──────────────────────────────────────────────────────────────────────────

    MOBILITY features: traffic_congestion_detected, evacuation_mentioned
    SOCIAL/EMOTION features: fear_keywords_present, anger_mentioned,
      anxiety_keywords_present, sadness_keywords_present, dominant_emotion_fear,
      high_negative_sentiment, high_positive_sentiment, solidarity_messages,
      policy_governance_discussion, sentiment_improved, sentiment_worsened,
      sentiment_shift_detected, mixed_emotions

    ── POLICY RELEVANCE SCORING RUBRIC (1–10) ────────────────────────────────
    Score on whether a named LA agency could act on this rule. This score is an
    annotation aid for the reader; it does NOT determine whether the rule is
    kept (the statistical pipeline already did that).

    10 — Directly actionable by a named LA agency (LAFD, CAL FIRE,
         LA County OES, LAPD, LA DPW) with a specific trigger and action.
     8–9 — Operationally useful; named agency and action obvious but needs
         one additional validation step.
     6–7 — Useful monitoring signal; further validation required.
     4–5 — Situational awareness only; no clear intervention opportunity.
     1–3 — Too generic or too rare to drive agency action.
    ──────────────────────────────────────────────────────────────────────────

    For cross-domain rules (mobility × emotion/discourse), note in your rationale
    whether the direction (emotion→mobility or mobility→emotion) is operationally
    interesting and which crisis phase it most relates to.

    IMPORTANT: Return EXACTLY ONE object per input rule, with the same rule_id.
    The number of objects MUST equal the number of rules provided. Do not omit,
    filter, deduplicate, or reorder by importance.

    Return ONLY a JSON array with one object per rule (any order):
    [
      {
        "rule_id": <int>,
        "policy_score": <int 1-10>,
        "reasoning": "<2-3 sentences: name the LA crisis phase, note whether the cross-domain direction is operationally interesting, and what it reveals>",
        "policy_recommendation": "<2 sentences: name a specific LA agency, the monitoring trigger, and the recommended action>"
      },
      ...
    ]
    No markdown fences. No extra text. Valid JSON only.
""")


def _load_env() -> None:
    """Load variables from .env into os.environ (no python-dotenv dependency)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _feature_set(rule: pd.Series) -> frozenset:
    """Return the full set of features referenced in a rule (premise + conclusion)."""
    parts = [p.strip() for p in rule["premise"].split(",")]
    parts.append(rule["conclusion"].strip())
    return frozenset(parts)


def tag_cross_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Add / refresh the cross_domain flag."""
    def _is_cross(row: pd.Series) -> bool:
        fs = _feature_set(row)
        return bool(fs & MOBILITY_FEATURES) and bool(fs & EMOTION_FEATURES)
    df = df.copy()
    df["cross_domain"] = df.apply(_is_cross, axis=1)
    return df


def _build_candidate_block(df: pd.DataFrame) -> str:
    """Render rules as a numbered text block for the LLM prompt."""
    lines = []
    has_conviction = "conviction" in df.columns
    for _, row in df.iterrows():
        cross = "YES" if row.get("cross_domain") else "NO"
        conviction_str = ""
        if has_conviction and pd.notna(row.get("conviction")):
            conviction_str = f" conviction={float(row['conviction']):.2f}"
        lines.append(
            f"[id={int(row['rule_id'])}] IF: {row['premise']} THEN: {row['conclusion']} "
            f"| support={int(row['support'])}d ({float(row['support_pct']):.1f}%) "
            f"confidence={float(row['confidence']):.1f}% lift={float(row['lift']):.2f}"
            f"{conviction_str} cross_domain={cross}"
        )
    return "\n".join(lines)


def annotate_all_rules_with_llm(
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Annotate EVERY rule with the LLM (policy relevance + rationale), dropping none.

    This is an annotation layer, not a selection or ranking step. Rule validity is
    already established by the statistical pipeline upstream. Every rule passed in
    is scored; nothing is filtered. Ordering of the output is by statistics, set
    by the caller — not by any LLM score.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.  Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY not found. Add it to .env or set it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = df.copy().reset_index(drop=True)
    df["rule_id"] = df.index

    # Send ALL rules — no stratified subset, no candidate cap, no top-N.
    candidate_block = _build_candidate_block(df)
    n = len(df)
    user_msg = (
        f"Annotate ALL {n} of the following validated rules. Return exactly {n} "
        f"objects, one per rule_id:\n\n" + candidate_block
    )

    print(f"  Sending ALL {n} rules to LLM for annotation ...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANNOTATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        annotations = json.loads(raw)
    except json.JSONDecodeError:
        print(f"WARNING: LLM returned non-JSON response:\n{raw}", file=sys.stderr)
        print("Proceeding with statistical ordering and no annotations.", file=sys.stderr)
        return df
    if not isinstance(annotations, list):
        print("WARNING: LLM response was not a JSON array; skipping annotations.", file=sys.stderr)
        return df

    df["policy_score"] = None
    df["llm_reasoning"] = ""
    df["llm_policy_recommendation"] = ""

    valid_ids = set(df["rule_id"].tolist())
    seen = set()
    for entry in annotations:
        rid = entry.get("rule_id")
        if rid not in valid_ids:
            continue
        idx = df.index[df["rule_id"] == rid][0]
        df.at[idx, "policy_score"] = entry.get("policy_score")
        df.at[idx, "llm_reasoning"] = entry.get("reasoning", "")
        df.at[idx, "llm_policy_recommendation"] = entry.get("policy_recommendation", "")
        seen.add(rid)

    missing = valid_ids - seen
    if missing:
        print(
            f"  WARNING: LLM did not annotate {len(missing)} rule(s): ids {sorted(missing)}. "
            f"They are retained with null annotation.",
            file=sys.stderr,
        )

    print(f"  LLM annotated {df['policy_score'].notna().sum()} / {n} rules.")
    return df


def _statistical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Order rules by objective statistics only: cross-domain first, then lift,
    then confidence. This is the ONLY ordering used; the LLM never reorders."""
    sort_cols = [c for c in ["cross_domain", "lift", "confidence"] if c in df.columns]
    asc = [False] * len(sort_cols)
    return df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Human-readable summary writer
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, out_path: Path, top_n: int | None = None) -> None:
    """Write a text summary of rules ordered by statistics.

    Rules are ordered by objective statistics (cross-domain, lift, confidence).
    If the LLM annotation step ran, each rule additionally shows its policy
    annotation — but the ORDER and INCLUSION are statistical, never LLM-driven.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = "policy_score" in df.columns and df["policy_score"].notna().any()

    ranked = _statistical_order(df)
    if top_n is not None:
        ranked = ranked.head(top_n)

    total_rules = len(df)
    cross_domain_count = int(df["cross_domain"].sum()) if "cross_domain" in df.columns else 0
    avg_lift = df["lift"].mean() if "lift" in df.columns else None
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None

    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  TOP RULES -- CROSS-DOMAIN MOBILITY x EMOTION ANALYSIS\n")
        f.write("  LA WILDFIRES 2025 (Jan 3 – Feb 4, 2025 | 33 days)\n")
        f.write("  Ordered by statistics (cross-domain, lift, confidence).\n")
        if annotated:
            f.write("  LLM provides policy annotation only; it does not rank or select.\n")
        f.write("=" * 72 + "\n\n")

        f.write("── RULE PIPELINE QUALITY SUMMARY ───────────────────────────────\n")
        f.write(f"  Total rules (statistically valid): {total_rules}\n")
        f.write(f"  Cross-domain rules               : {cross_domain_count}"
                f" ({100 * cross_domain_count / max(total_rules, 1):.1f}%)\n")
        if avg_lift is not None:
            f.write(f"  Mean lift                        : {avg_lift:.2f}\n")
        if avg_conf is not None:
            f.write(f"  Mean confidence                  : {avg_conf:.1f}%\n")
        f.write("─" * 72 + "\n\n")

        for i, (_, rule) in enumerate(ranked.iterrows(), start=1):
            cross = " [CROSS-DOMAIN]" if rule.get("cross_domain") else ""
            f.write(f"Rule {i}:{cross}\n")
            f.write(f"  IF:   {rule['premise']}\n")
            f.write(f"  THEN: {rule['conclusion']}\n")
            f.write(
                f"  Support: {int(rule['support'])} days ({float(rule['support_pct']):.1f}%)  "
                f"Confidence: {float(rule['confidence']):.1f}%  Lift: {float(rule['lift']):.2f}\n"
            )
            if annotated:
                policy = rule.get("policy_score")
                if pd.notna(policy):
                    f.write(f"  Policy relevance (LLM annotation): {int(policy)}/10\n")
                reasoning = rule.get("llm_reasoning", "")
                if reasoning and str(reasoning).strip():
                    f.write(f"  Insight: {reasoning}\n")
                rec = rule.get("llm_policy_recommendation", "")
                if rec and str(rec).strip():
                    f.write(f"  Recommendation: {rec}\n")
            f.write("\n")

    print(f"Summary written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Annotate FCA rules with optional LLM policy commentary (no selection/ranking)."
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM annotation via OpenAI API (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optionally limit the text summary to the top-N rules by statistics "
             "(default: show all).",
    )
    args = parser.parse_args()

    if not RULES_FILE.exists():
        print(f"ERROR: Rules file not found: {RULES_FILE}", file=sys.stderr)
        print("Run scripts/fca_analysis.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading rules from {RULES_FILE} ...")
    df = pd.read_csv(RULES_FILE)
    print(f"  Loaded {len(df)} rules.")

    df = tag_cross_domain(df)
    cross_count = df["cross_domain"].sum()
    print(f"  Cross-domain rules: {cross_count} / {len(df)}")

    if args.llm:
        print(f"Running LLM annotation (model={args.model}) on ALL rules ...")
        df = annotate_all_rules_with_llm(df, model=args.model)
    else:
        print("Skipping LLM step (use --llm to enable).")

    # Always store in statistical order so the CSV itself reflects the
    # objective ranking, not any LLM influence.
    df = _statistical_order(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Evaluated (annotated) rules saved to {OUT_CSV}")

    write_summary(df, OUT_TXT, top_n=args.top_n)


if __name__ == "__main__":
    main()