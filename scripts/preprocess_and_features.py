"""
Feature Engineering Script
Creates binary features for FCA from mobility and social media data
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILITY_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_data_processed.csv"
BASELINES_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_baselines.json"
REDDIT_JSON_FILE = PROJECT_ROOT / "data" / "raw" / "reddit_wildfire_posts_by_day.json"

OUTPUT_SENTIMENT_FILE = PROJECT_ROOT / "data" / "processed" / "reddit_sentiment_features.csv"
OUTPUT_FCA_FILE = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
# ==========================

# FCA feature quality filters
MIN_FEATURE_SUPPORT_DAYS = 2
MAX_FEATURE_PREVALENCE = 0.90

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ========= KEYWORD DICTIONARIES =========
FIRE_WORDS = ["fire", "wildfire", "wild fire", "blaze", "flames", "smoke", "ash", "embers", "burn"]
EVAC_WORDS = ["evacuate", "evacuation", "shelter", "shelter in place", "leave now", "mandatory evacuation"]
POLICY_WORDS = ["trump", "biden", "newsom", "governor", "mayor", "fema", "president", "gop", "democrats", "republicans", "policy", "funding", "aid", "mismanagement", "negligence", "failure", "blame"]
SOLIDARITY_WORDS = ["donate", "donation", "fundraiser", "charity", "volunteers", "volunteering", "stay safe", "prayers", "thoughts", "support", "help", "relief", "shelter", "food", "clothes", "supplies"]
MISINFO_WORDS = ["fake", "hoax", "misinformation", "disinformation", "conspiracy", "rumor", "not true", "debunked", "fact-check", "fact check"]
INFO_GAP_WORDS = ["any info", "any information", "what's happening", "whats happening", "does anyone know", "can someone explain", "is it safe", "unclear", "confusing"]
FEAR_WORDS = ["afraid", "scared", "terrified", "fear", "panic", "panicking", "worried", "alarming", "danger", "threat", "risk"]
ANGER_WORDS = ["anger", "angry", "rage", "furious", "pissed", "blame", "failure", "incompetent", "negligence", "mad"]
SADNESS_WORDS = ["sad", "devastated", "heartbroken", "tragic", "loss", "lost", "crying", "tears", "hopeless", "despair"]
ANXIETY_WORDS = ["anxious", "anxiety", "nervous", "uncertain", "unsure", "what if", "worried", "stress", "stressful", "overwhelmed"]


def contains_keywords(text, keywords):
    """Check if text contains any keywords"""
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)


def emotion_intensity(text, keywords):
    """Count how many emotion words appear"""
    text_lower = text.lower()
    return sum(1 for w in keywords if w in text_lower)


def calculate_volatility(scores, window=3):
    """Calculate rolling standard deviation"""
    if len(scores) < window:
        return 0
    return float(np.std(scores[-window:]))


def get_binary_feature_columns(df, exclude=None):
    """Return columns that are cleanly coercible to binary 0/1 values."""
    exclude = exclude or set()
    binary_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        raw_non_null = df[col].notna().sum()
        converted_non_null = numeric_values.notna().sum()
        if raw_non_null == 0 or converted_non_null != raw_non_null:
            continue
        unique_values = set(numeric_values.dropna().unique())
        if unique_values.issubset({0, 1}):
            binary_cols.append(col)
    return binary_cols


def process_sentiment_features():
    """Process Reddit data to create sentiment features"""
    print("📊 Processing sentiment features...")

    with open(REDDIT_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    posts_by_day = data.get('posts_by_day', data)
    print(f"  Loaded posts for {len(posts_by_day)} days")

    rows = []
    prev_avg = None
    sentiment_history = []

    for day in sorted(posts_by_day.keys()):
        posts = posts_by_day[day]
        if not posts:
            continue

        scores = []
        neg_count = pos_count = 0
        fear_count = anger_count = sadness_count = anxiety_count = 0
        fear_intensity_val = anger_intensity_val = 0

        for p in posts:
            text = f"{p.get('title', '')} {p.get('text', '')}"
            vs = analyzer.polarity_scores(text)
            scores.append(vs["compound"])

            if vs["compound"] <= -0.3:
                neg_count += 1
            elif vs["compound"] >= 0.3:
                pos_count += 1

            if contains_keywords(text, FEAR_WORDS):
                fear_count += 1
                fear_intensity_val += emotion_intensity(text, FEAR_WORDS)
            if contains_keywords(text, ANGER_WORDS):
                anger_count += 1
                anger_intensity_val += emotion_intensity(text, ANGER_WORDS)
            if contains_keywords(text, SADNESS_WORDS):
                sadness_count += 1
            if contains_keywords(text, ANXIETY_WORDS):
                anxiety_count += 1

        avg_compound = sum(scores) / len(scores)
        neg_fraction = neg_count / len(posts)
        pos_fraction = pos_count / len(posts)

        high_negative_sentiment = 1 if avg_compound < -0.3 else 0
        high_positive_sentiment = 1 if avg_compound > 0.3 else 0

        if prev_avg is None:
            sentiment_shift_detected = 0
            sentiment_improved = 0
            sentiment_worsened = 0
        else:
            shift = avg_compound - prev_avg
            sentiment_shift_detected = 1 if abs(shift) > 0.2 else 0
            sentiment_improved = 1 if shift > 0.2 else 0
            sentiment_worsened = 1 if shift < -0.2 else 0

        prev_avg = avg_compound
        sentiment_history.append(avg_compound)
        sentiment_volatility = calculate_volatility(sentiment_history)

        fear_keywords_present = 1 if fear_count > 0 else 0
        anger_mentioned = 1 if anger_count > 0 else 0
        sadness_keywords_present = 1 if sadness_count > 0 else 0
        anxiety_keywords_present = 1 if anxiety_count > 0 else 0

        dominant_emotion_fear = 1 if (fear_count > anger_count and fear_count / len(posts) > 0.3) else 0

        emotion_counts = {
            'fear': fear_count,
            'anger': anger_count,
            'sadness': sadness_count,
            'anxiety': anxiety_count,
        }
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        emotions_present = sum(1 for v in emotion_counts.values() if v > 0)
        mixed_emotions = 1 if emotions_present >= 3 else 0

        fear_intensity_norm = fear_intensity_val / len(posts)
        anger_intensity_norm = anger_intensity_val / len(posts)

        rows.append({
            "date": day,
            "num_posts": len(posts),
            "avg_compound": avg_compound,
            "neg_fraction": neg_fraction,
            "pos_fraction": pos_fraction,
            "high_negative_sentiment": high_negative_sentiment,
            "high_positive_sentiment": high_positive_sentiment,
            "sentiment_shift_detected": sentiment_shift_detected,
            "sentiment_improved": sentiment_improved,
            "sentiment_worsened": sentiment_worsened,
            "sentiment_volatility": sentiment_volatility,
            "fear_keywords_present": fear_keywords_present,
            "anger_mentioned": anger_mentioned,
            "sadness_keywords_present": sadness_keywords_present,
            "anxiety_keywords_present": anxiety_keywords_present,
            "dominant_emotion_fear": dominant_emotion_fear,
            "dominant_emotion": dominant_emotion,
            "mixed_emotions": mixed_emotions,
            "fear_intensity_norm": fear_intensity_norm,
            "anger_intensity_norm": anger_intensity_norm,
        })

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.to_csv(OUTPUT_SENTIMENT_FILE, index=False)
    print(f"  ✓ Saved sentiment features to {OUTPUT_SENTIMENT_FILE}")

    return df


def process_topic_features():
    """Extract topic features from Reddit data"""
    print("📑 Processing topic features...")

    with open(REDDIT_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    posts_by_day = data.get('posts_by_day', data)
    topic_rows = []

    for day_str, posts in posts_by_day.items():
        day_date = datetime.fromisoformat(day_str).date()

        fire_count = evac_count = policy_count = 0
        solid_count = misinfo_count = info_gap_count = 0

        for p in posts:
            text = f"{p.get('title', '')} {p.get('text', '')}"
            if contains_keywords(text, FIRE_WORDS):
                fire_count += 1
            if contains_keywords(text, EVAC_WORDS):
                evac_count += 1
            if contains_keywords(text, POLICY_WORDS):
                policy_count += 1
            if contains_keywords(text, SOLIDARITY_WORDS):
                solid_count += 1
            if contains_keywords(text, MISINFO_WORDS):
                misinfo_count += 1
            if contains_keywords(text, INFO_GAP_WORDS):
                info_gap_count += 1

        if len(posts) == 0:
            continue

        topic_rows.append({
            "date": day_date,
            "fire_topic_detected": int(fire_count > 0),
            "evacuation_mentioned": int(evac_count > 0),
            "policy_governance_discussion": int(policy_count > 0),
            "solidarity_messages": int(solid_count > 0),
            "misinformation_mentions": int(misinfo_count > 0),
            "information_request_uncertainty": int(info_gap_count > 0),
        })

    df = pd.DataFrame(topic_rows)
    print(f"  ✓ Created topic features for {len(df)} days")

    return df


def process_mobility_features():
    """Create binary mobility features"""
    print("🚗 Processing mobility features...")

    mobility = pd.read_csv(MOBILITY_FILE)
    mobility['Date'] = pd.to_datetime(mobility['Date']).dt.date

    with open(BASELINES_FILE, 'r') as f:
        baselines = json.load(f)

    print(f"  Loaded {len(mobility)} days of mobility data")
    print(f"  Using baselines: VMT={baselines['VMT']['mean']:.0f}, TTI={baselines['TTI']['mean']:.3f}")

    if "Delay (V_t=35) (Veh-Hours)" in mobility.columns:
        delay_col = "Delay (V_t=35) (Veh-Hours)"
    elif "Delay (V_t=35)" in mobility.columns:
        delay_col = "Delay (V_t=35)"
    else:
        delay_col = None

    features = pd.DataFrame()
    features['date'] = mobility['Date']

    baseline_VMT = baselines['VMT']['mean']
    baseline_Delay = baselines.get('Delay', {}).get('mean', 0)

    features['mobility_drop_traffic'] = (
        mobility['VMT (Veh-Miles)'] < 0.7 * baseline_VMT
    ).astype(int)

    if delay_col:
        features['traffic_congestion_detected'] = (
            (mobility['TTI'] > 1.15) |
            (mobility[delay_col] > 1.3 * baseline_Delay)
        ).astype(int)
    else:
        features['traffic_congestion_detected'] = (
            mobility['TTI'] > 1.15
        ).astype(int)

    features['evacuation_pattern_observed'] = (
        (features['mobility_drop_traffic'] == 1) &
        (features['traffic_congestion_detected'] == 1)
    ).astype(int)

    print("  ✓ Created mobility features")
    return features


def create_combined_features():
    """Combine all features into final FCA matrix"""
    print("\n🔗 Combining all features...")

    sentiment_df = pd.read_csv(OUTPUT_SENTIMENT_FILE)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

    topic_df = process_topic_features()
    mobility_df = process_mobility_features()

    df = mobility_df.merge(sentiment_df, on='date', how='outer')
    df = df.merge(topic_df, on='date', how='outer')

    df = df.fillna(0)

    df['date_dt'] = pd.to_datetime(df['date'])
    df['weekend'] = (df['date_dt'].dt.dayofweek >= 5).astype(int)

    def behavioral_flags(row):
        high_neg = int(row.get('high_negative_sentiment', 0))
        fear_dom = int(row.get('dominant_emotion_fear', 0))
        fear_present = int(row.get('fear_keywords_present', 0))

        mob_cols = [
            'mobility_drop_traffic',
            'traffic_congestion_detected',
            'evacuation_pattern_observed',
        ]

        mob_flags = [int(row.get(c, 0)) for c in mob_cols]
        any_mob = int(any(bool(m) for m in mob_flags))

        emotion_mobility_mismatch = int((high_neg or fear_dom or fear_present) and not any_mob)
        low_emotion_low_mobility_signal = int((high_neg == 0) and (fear_dom == 0) and (fear_present == 0) and not any_mob)
        emotion_with_mobility_signal = int((fear_dom or fear_present or high_neg) and any_mob)

        return pd.Series({
            'emotion_mobility_mismatch': emotion_mobility_mismatch,
            'low_emotion_low_mobility_signal': low_emotion_low_mobility_signal,
            'emotion_with_mobility_signal': emotion_with_mobility_signal,
        })

    behav = df.apply(behavioral_flags, axis=1)
    df = pd.concat([df, behav], axis=1)

    df = df.drop(['date_dt'], axis=1)
    df = df.rename(columns={'date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    binary_cols = get_binary_feature_columns(df, exclude={'Date'})
    filtered_out = []
    for col in binary_cols:
        series = pd.to_numeric(df[col], errors='coerce').fillna(0)
        support = int(series.sum())
        prevalence = support / len(df)
        if support < MIN_FEATURE_SUPPORT_DAYS or support == 0 or support == len(df) or prevalence > MAX_FEATURE_PREVALENCE:
            filtered_out.append(col)

    if filtered_out:
        df = df.drop(columns=filtered_out)
        print(f"  ✓ Removed {len(filtered_out)} low-value binary features")
        print(f"    Removed: {', '.join(sorted(filtered_out))}")

    df.to_csv(OUTPUT_FCA_FILE, index=False)
    print(f"  ✓ Saved FCA matrix to {OUTPUT_FCA_FILE}")

    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    binary_cols = get_binary_feature_columns(
        df,
        exclude={
            'Date',
            'num_posts',
            'avg_compound',
            'neg_fraction',
            'pos_fraction',
            'sentiment_volatility',
            'fear_intensity_norm',
            'anger_intensity_norm',
        },
    )

    print(f"Total days: {len(df)}")
    print(f"Total binary features for FCA: {len(binary_cols)}")
    print("\nFeature Activation Rates:")
    for col in sorted(binary_cols):
        activation = (pd.to_numeric(df[col], errors='coerce').fillna(0).sum() / len(df)) * 100
        if activation > 0:
            print(f"  {col:40s} {activation:5.1f}%")


def main():
    print("=" * 70)
    print("FEATURE ENGINEERING - FCA BINARY MATRIX")
    print("=" * 70)

    process_sentiment_features()
    create_combined_features()

    print("\n" + "=" * 70)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
