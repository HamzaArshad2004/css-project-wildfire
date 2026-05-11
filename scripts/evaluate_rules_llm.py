"""
LLM-based rule evaluation and prioritization.

Loads association rules from FCA, then optionally uses an OpenAI-compatible
LLM to identify the top 20 most novel and policy-relevant rules in a single
batch call.  Cross-domain rules are always flagged and sorted to the top
regardless of whether the LLM step is run.

Usage (without LLM -- statistical ranking only):
    python scripts/evaluate_rules_llm.py

Usage (with LLM selection -- requires OPENAI_API_KEY in .env or environment):
    python scripts/evaluate_rules_llm.py --llm

Output:
    results/fca/association_rules_evaluated.csv   (all rules with LLM scores if run)
    results/fca/top_cross_domain_rules.txt        (top N human-readable summary)
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

CANDIDATE_POOL = 100
TOP_N_LLM = 20

BATCH_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert data scientist and emergency management policy advisor
    reviewing association rules mined from LA Wildfire 2025 crisis behavior data
    (January 3 – February 4, 2025). The dataset covers 33 days and combines
    daily LA traffic metrics (VMT, TTI) with Reddit sentiment signals from
    local subreddits.

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

    COMPOSITE features (do NOT select rules that simply unpack these):
      emotion_with_mobility_signal = (any emotion) AND (any mobility signal)
      emotion_mobility_mismatch    = (any emotion) AND NOT (any mobility signal)
      low_emotion_low_mobility_signal = NOT(emotion) AND NOT(mobility)

    ── NOVELTY SCORING RUBRIC (1–10) ─────────────────────────────────────────
    10 — Cross-domain rule that reveals a counter-intuitive direction
         (e.g. solidarity messaging predicts traffic reduction, or fear
         keywords predict evacuation mentions with a specific crisis phase link).
     8–9 — Cross-domain with a non-obvious direction; timing aligns with a
         known phase but the association would surprise an emergency manager.
     6–7 — Single-domain but reveals non-trivial co-occurrence within the
         33-day crisis window.
     4–5 — Expected co-occurrence (e.g. fear + high negative sentiment).
     1–3 — Near-tautological or definitionally implied.
    ──────────────────────────────────────────────────────────────────────────

    ── POLICY RELEVANCE SCORING RUBRIC (1–10) ────────────────────────────────
    10 — Finding is directly actionable by a named LA agency (LAFD, CAL FIRE,
         LA County OES, LAPD, LA DPW) with a specific trigger and action.
     8–9 — Operationally useful; named agency and action obvious but needs
         one additional validation step.
     6–7 — Useful monitoring signal; further validation required.
     4–5 — Situational awareness only; no clear intervention opportunity.
     1–3 — Too generic or too rare to drive agency action.
    ──────────────────────────────────────────────────────────────────────────

    MANDATORY EXCLUSIONS — do NOT select rules where:
    - emotion_with_mobility_signal or emotion_mobility_mismatch is in the
      PREMISE and traffic/evacuation is the CONCLUSION (tautological unpacking).
    - Rules concluding sentiment_shift_detected when sentiment_improved or
      sentiment_worsened is in the premise (definitional).
    - Both premise and conclusion are purely within one domain with no
      cross-domain element.
    - More than 3 rules share the same conclusion.

    PREFER:
    - Emotion → mobility direction (social signal predicts physical response).
    - Mobility → emotion direction (physical disruption predicts sentiment shift).
    - Solidarity or policy discourse as surprising predictors.
    - Rules with lift > 1.5 and support across multiple days.

    YOUR TASK: select and rank the TOP 20 most novel and policy-relevant rules.

    Return ONLY a JSON array of exactly 20 objects (fewer if fewer qualify),
    ordered best-first:
    [
      {
        "rule_id": <int>,
        "novelty_score": <int 1-10>,
        "policy_score": <int 1-10>,
        "reasoning": "<2-3 sentences: name the LA crisis phase, explain why the direction is non-obvious, and what it reveals about crisis behaviour>",
        "policy_recommendation": "<2 sentences: name a specific LA agency, state the exact monitoring trigger and the recommended action>"
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
    """Render candidate rules as a numbered text block for the LLM prompt."""
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


def _stratified_candidates(df: pd.DataFrame, candidate_pool: int) -> pd.DataFrame:
    """
    Return a stratified candidate pool so the LLM sees both rule directions.
    Split evenly between rules concluding a SOCIAL/EMOTION feature and those
    concluding a MOBILITY feature. Fill any shortfall from the other stratum.
    """
    mob = MOBILITY_FEATURES
    half = candidate_pool // 2

    social_conclusion = (
        df[df["cross_domain"] & ~df["conclusion"].isin(mob)]
        .sort_values(["lift", "confidence"], ascending=False)
        .head(half)
    )
    mob_conclusion = (
        df[df["cross_domain"] & df["conclusion"].isin(mob)]
        .sort_values(["lift", "confidence"], ascending=False)
        .head(half)
    )
    combined = pd.concat([social_conclusion, mob_conclusion]).drop_duplicates()
    if len(combined) < candidate_pool:
        remaining = (
            df[~df.index.isin(combined.index)]
            .sort_values(["lift", "confidence"], ascending=False)
            .head(candidate_pool - len(combined))
        )
        combined = pd.concat([combined, remaining])
    return combined.head(candidate_pool)


def select_top_rules_with_llm(
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    candidate_pool: int = CANDIDATE_POOL,
    top_n: int = TOP_N_LLM,
) -> pd.DataFrame:
    """
    Send a stratified pool of candidate rules to the LLM in one batch call and
    ask it to select and rank the top_n most novel and policy-relevant ones.
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

    candidates = _stratified_candidates(df, candidate_pool)
    candidate_block = _build_candidate_block(candidates)
    user_msg = (
        f"Select the top {top_n} rules from these {len(candidates)} candidates:\n\n"
        + candidate_block
    )

    print(f"  Sending {len(candidates)} candidate rules to LLM for batch selection ...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2400,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        selections = json.loads(raw)
    except json.JSONDecodeError:
        print(f"WARNING: LLM returned non-JSON response:\n{raw}", file=sys.stderr)
        print("Falling back to statistical ranking.", file=sys.stderr)
        return df

    if not isinstance(selections, list):
        print("WARNING: LLM response was not a JSON array; falling back to statistical ranking.",
              file=sys.stderr)
        return df

    df["novelty_score"] = None
    df["policy_score"] = None
    df["llm_reasoning"] = ""
    df["llm_policy_recommendation"] = ""
    df["llm_rank"] = None

    valid_ids = set(df["rule_id"].tolist())
    for rank, entry in enumerate(selections[:top_n], start=1):
        rid = entry.get("rule_id")
        if rid not in valid_ids:
            continue
        idx = df.index[df["rule_id"] == rid][0]
        df.at[idx, "novelty_score"] = entry.get("novelty_score")
        df.at[idx, "policy_score"] = entry.get("policy_score")
        df.at[idx, "llm_reasoning"] = entry.get("reasoning", "")
        df.at[idx, "llm_policy_recommendation"] = entry.get("policy_recommendation", "")
        df.at[idx, "llm_rank"] = rank

    df["llm_composite"] = df[["novelty_score", "policy_score"]].mean(axis=1)
    print(f"  LLM selected {df['llm_rank'].notna().sum()} rules.")
    return df


# ---------------------------------------------------------------------------
# Human-readable summary writer
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, out_path: Path, top_n: int = TOP_N_LLM) -> None:
    """Write a text summary of the top rules."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm_selected = "llm_rank" in df.columns and df["llm_rank"].notna().any()

    if llm_selected:
        ranked = (
            df[df["llm_rank"].notna()]
            .sort_values("llm_rank")
            .head(top_n)
        )
        header_note = f"  LLM-SELECTED: Top {top_n} Most Novel & Policy-Relevant Rules\n"
    else:
        ranked = (
            df.sort_values(
                ["cross_domain", "lift", "confidence"],
                ascending=[False, False, False],
            ).head(top_n)
        )
        header_note = f"  Top {top_n} Rules by Statistical Ranking (cross-domain priority)\n"

    total_rules = len(df)
    cross_domain_count = int(df["cross_domain"].sum()) if "cross_domain" in df.columns else "N/A"
    avg_lift = df["lift"].mean() if "lift" in df.columns else None
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None

    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  TOP RULES -- CROSS-DOMAIN MOBILITY x EMOTION ANALYSIS\n")
        f.write("  LA WILDFIRES 2025 (Jan 3 – Feb 4, 2025 | 33 days)\n")
        f.write(header_note)
        f.write("=" * 72 + "\n\n")

        # Pipeline quality summary
        f.write("── RULE PIPELINE QUALITY SUMMARY ───────────────────────────────\n")
        f.write(f"  Total rules mined   : {total_rules}\n")
        f.write(f"  Cross-domain rules  : {cross_domain_count}"
                f" ({100 * cross_domain_count / max(total_rules, 1):.1f}%)\n")
        if avg_lift is not None:
            f.write(f"  Mean lift           : {avg_lift:.2f}\n")
        if avg_conf is not None:
            f.write(f"  Mean confidence     : {avg_conf:.1f}%\n")
        if llm_selected:
            avg_novelty = ranked["novelty_score"].mean()
            avg_policy = ranked["policy_score"].mean()
            f.write(f"  LLM avg novelty     : {avg_novelty:.1f}/10\n")
            f.write(f"  LLM avg policy      : {avg_policy:.1f}/10\n")
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
            if llm_selected and pd.notna(rule.get("llm_rank")):
                novelty = rule.get("novelty_score")
                policy = rule.get("policy_score")
                if pd.notna(novelty) and pd.notna(policy):
                    f.write(f"  LLM: novelty={int(novelty)}/10  policy_relevance={int(policy)}/10\n")
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

    parser = argparse.ArgumentParser(description="Evaluate FCA rules with optional LLM scoring.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM batch selection via OpenAI API (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_LLM,
        help=f"Number of top rules to select (default: {TOP_N_LLM}).",
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
        print(f"Running LLM batch selection (model={args.model}, top_n={args.top_n}) ...")
        df = select_top_rules_with_llm(df, model=args.model, top_n=args.top_n)
    else:
        print("Skipping LLM step (use --llm to enable).")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Evaluated rules saved to {OUT_CSV}")

    write_summary(df, OUT_TXT, top_n=args.top_n)


if __name__ == "__main__":
    main()
