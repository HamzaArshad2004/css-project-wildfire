"""
Results Visualization Script
Creates comprehensive visualizations of mobility, sentiment, and FCA results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILITY_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_data_processed.csv"
FCA_FILE = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
REDDIT_JSON = PROJECT_ROOT / "data" / "raw" / "reddit_wildfire_posts_by_day.json"
RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules.csv"
EVALUATED_RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules_evaluated.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"
# ==========================

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class ResultsVisualizer:
    def __init__(self):
        """Load all data files"""
        print("📂 Loading data files...")

        self.mobility_df = pd.read_csv(MOBILITY_FILE)
        self.mobility_df['Date'] = pd.to_datetime(self.mobility_df['Date'])

        self.features_df = pd.read_csv(FCA_FILE)
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])

        with open(REDDIT_JSON, 'r') as f:
            data = json.load(f)
        posts_by_day = data.get('posts_by_day', data)

        daily_rows = []
        for day, posts in posts_by_day.items():
            if posts:
                daily_rows.append({
                    'Date': pd.to_datetime(day),
                    'num_posts': len(posts),
                    'avg_score': np.mean([p.get('score', 0) for p in posts]),
                })
        self.reddit_daily = pd.DataFrame(daily_rows)

        print(f"  ✓ Loaded {len(self.mobility_df)} days of mobility data")
        print(f"  ✓ Loaded {len(self.features_df)} days of features")
        print(f"  ✓ Loaded {len(self.reddit_daily)} days of Reddit data")

    @staticmethod
    def binary_columns(df, exclude=None):
        exclude = exclude or set()
        cols = []
        for col in df.columns:
            if col in exclude:
                continue
            vals = pd.to_numeric(df[col], errors='coerce')
            if vals.notna().sum() != df[col].notna().sum() or vals.notna().sum() == 0:
                continue
            unique_vals = set(vals.dropna().unique())
            if unique_vals.issubset({0, 1}):
                cols.append(col)
        return cols

    def plot_mobility_trends(self):
        """Plot mobility metrics over time"""
        print("\n📈 Creating mobility trends plot...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Mobility Trends During LA Wildfire Crisis', fontsize=16, fontweight='bold')

        delay_col = None
        for col in ["Delay (V_t=35) (Veh-Hours)", "Delay (V_t=35)"]:
            if col in self.mobility_df.columns:
                delay_col = col
                break

        metrics = [
            ('VMT (Veh-Miles)', '#1f77b4', 'VMT'),
            ('TTI', '#ff7f0e', 'TTI'),
            ('VHT (Veh-Hours)', '#2ca02c', 'VHT'),
            (delay_col if delay_col else 'TTI', '#d62728', 'Delay'),
        ]

        for idx, (metric, color, label) in enumerate(metrics):
            if metric and metric in self.mobility_df.columns:
                ax = axes[idx // 2, idx % 2]
                ax.plot(self.mobility_df['Date'], self.mobility_df[metric], color=color, linewidth=2, marker='o', markersize=4)
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'mobility_trends.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_sentiment_timeline(self):
        """Plot sentiment over time"""
        print("💭 Creating sentiment timeline...")

        fig, ax = plt.subplots(figsize=(14, 6))

        if 'avg_compound' in self.features_df.columns:
            ax.plot(self.features_df['Date'], self.features_df['avg_compound'], label='Compound Sentiment', linewidth=2.5, marker='o', color='purple', markersize=5)

            if 'neg_fraction' in self.features_df.columns:
                ax.plot(self.features_df['Date'], self.features_df['neg_fraction'], label='Negative Fraction', linewidth=2, marker='s', color='red', alpha=0.7, markersize=4)

            if 'pos_fraction' in self.features_df.columns:
                ax.plot(self.features_df['Date'], self.features_df['pos_fraction'], label='Positive Fraction', linewidth=2, marker='^', color='green', alpha=0.7, markersize=4)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Daily Sentiment from Reddit Posts', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'sentiment_timeline.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_features_heatmap(self):
        """Create heatmap of binary features"""
        print("🔥 Creating features heatmap...")

        binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})

        if len(binary_cols) == 0:
            print("  ⚠️  No binary columns found, skipping heatmap")
            return

        preferred_order = [
            'high_negative_sentiment', 'fear_keywords_present', 'anger_mentioned',
            'traffic_congestion_detected', 'mobility_drop_traffic',
            'evacuation_pattern_observed', 'policy_governance_discussion',
            'solidarity_messages', 'sentiment_shift_detected',
            'emotion_with_mobility_signal', 'emotion_mobility_mismatch',
            'evacuation_mentioned', 'weekend',
        ]
        binary_cols = [c for c in preferred_order if c in binary_cols]

        if len(binary_cols) == 0:
            print("  ⚠️  No selected binary columns found, skipping heatmap")
            return

        binary_matrix = self.features_df[binary_cols]

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            binary_matrix.T,
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Feature Active'},
            yticklabels=binary_cols,
            xticklabels=[f"D{i + 1}" for i in range(len(binary_matrix))],
        )

        plt.title('Binary Feature Activation Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()

        output_file = OUTPUT_DIR / 'features_heatmap.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_combined_mobility_sentiment(self):
        """Plot mobility and sentiment together"""
        print("🔗 Creating combined mobility-sentiment plot...")

        merged = self.features_df[['Date', 'avg_compound']].merge(
            self.mobility_df[['Date', 'VMT (Veh-Miles)']],
            on='Date',
            how='outer',
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        color1 = '#1f77b4'
        ax1.plot(merged['Date'], merged['VMT (Veh-Miles)'], color=color1, linewidth=2.5, marker='o', markersize=5)
        ax1.set_ylabel('Vehicle Miles Traveled (VMT)', color=color1, fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Mobility vs Sentiment During Crisis', fontsize=14, fontweight='bold')

        color2 = '#d62728'
        ax2.plot(merged['Date'], merged['avg_compound'], color=color2, linewidth=2.5, marker='s', markersize=5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Sentiment (Compound)', color=color2, fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        output_file = OUTPUT_DIR / 'combined_analysis.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def create_summary_report(self):
        """Generate text summary"""
        print("📝 Creating summary report...")

        output_file = OUTPUT_DIR.parent / 'summary_report.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prefer evaluated CSV (with LLM scores) over raw rules
        rules_df = pd.DataFrame()
        for path in (EVALUATED_RULES_FILE, RULES_FILE):
            try:
                rules_df = pd.read_csv(path)
                break
            except Exception:
                continue

        llm_available = "llm_rank" in rules_df.columns and rules_df["llm_rank"].notna().any()

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CRISIS BEHAVIORAL ANALYSIS - SUMMARY REPORT\n")
            f.write("LA WILDFIRES - JANUARY 2025\n")
            f.write("=" * 80 + "\n\n")

            f.write("DATA OVERVIEW:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Analysis Period: {self.features_df['Date'].min().date()} to {self.features_df['Date'].max().date()}\n")
            f.write(f"Total Days Analyzed: {len(self.features_df)}\n")
            binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
            f.write(f"Binary Features: {len(binary_cols)}\n\n")

            f.write("FEATURE ACTIVATION SUMMARY:\n")
            f.write("-" * 80 + "\n")
            for col in sorted(binary_cols):
                activation_pct = (pd.to_numeric(self.features_df[col], errors='coerce').fillna(0).sum() / len(self.features_df)) * 100
                if activation_pct > 0:
                    f.write(f"  {col:45s} {activation_pct:5.1f}%\n")

            if len(rules_df) > 0:
                if llm_available:
                    top_rules = (
                        rules_df[rules_df["llm_rank"].notna()]
                        .sort_values("llm_rank")
                        .head(20)
                    )
                    f.write("\n\nTOP 20 LLM-SELECTED ASSOCIATION RULES:\n")
                else:
                    top_rules = (
                        rules_df.sort_values(
                            ["cross_domain", "lift", "confidence"] if "cross_domain" in rules_df.columns
                            else ["lift", "confidence"],
                            ascending=False,
                        ).head(10)
                    )
                    f.write("\n\nTOP 10 ASSOCIATION RULES:\n")
                f.write("-" * 80 + "\n")

                for i, (_, row) in enumerate(top_rules.iterrows(), start=1):
                    cross = " [CROSS-DOMAIN]" if row.get("cross_domain") else ""
                    f.write(f"\nRule {i}:{cross}\n")
                    f.write(f"  IF:   {row['premise']}\n")
                    f.write(f"  THEN: {row['conclusion']}\n")
                    f.write(f"  Support: {row['support']} days ({float(row['support_pct']):.1f}%)\n")
                    if 'confidence' in row:
                        f.write(f"  Confidence: {float(row['confidence']):.1f}%\n")
                    if 'lift' in row:
                        f.write(f"  Lift: {float(row['lift']):.2f}\n")
                    if llm_available and pd.notna(row.get("llm_rank")):
                        novelty = row.get("novelty_score")
                        policy = row.get("policy_score")
                        if pd.notna(novelty) and pd.notna(policy):
                            f.write(f"  LLM: novelty={int(novelty)}/10  policy_relevance={int(policy)}/10\n")
                        reasoning = row.get("llm_reasoning", "")
                        if reasoning and str(reasoning).strip():
                            f.write(f"  Why it matters: {reasoning}\n")
                        rec = row.get("llm_policy_recommendation", "")
                        if rec and str(rec).strip():
                            f.write(f"  Policy recommendation: {rec}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"  ✓ Saved to {output_file}")


def main():
    print("=" * 70)
    print("RESULTS VISUALIZATION")
    print("=" * 70)

    visualizer = ResultsVisualizer()
    visualizer.plot_mobility_trends()
    visualizer.plot_sentiment_timeline()
    visualizer.plot_features_heatmap()
    visualizer.plot_combined_mobility_sentiment()
    visualizer.create_summary_report()

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
