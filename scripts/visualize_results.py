"""
Results Visualization Script
Creates comprehensive visualizations of mobility, sentiment, and FCA results
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# LA Wildfire 2025 crisis phase boundaries
_PHASES = [
    ("2025-01-03", "2025-01-06", "#fff3cd", "Pre-ignition"),
    ("2025-01-07", "2025-01-11", "#ffd6d6", "Ignition & spread"),
    ("2025-01-12", "2025-01-20", "#f4b8b8", "Peak crisis"),
    ("2025-01-21", "2025-02-04", "#d6eaff", "Containment"),
]

_CRISIS_EVENTS = [
    ("2025-01-07", "Fires ignite", "red"),
    ("2025-01-12", "Peak evacuations", "darkred"),
    ("2025-01-30", "80% contained", "steelblue"),
]


def _add_phase_bands(ax, alpha: float = 0.22) -> None:
    """Shade LA wildfire crisis phases on a date-axis Axes."""
    for start, end, color, label in _PHASES:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   color=color, alpha=alpha, zorder=0)


def _add_event_lines(ax, fontsize: float = 6.5) -> None:
    """Draw vertical lines for key crisis events."""
    for date_str, label, color in _CRISIS_EVENTS:
        d = pd.to_datetime(date_str)
        ax.axvline(d, color=color, linewidth=1.2, linestyle='--', alpha=0.7, zorder=1)
        ax.text(d, ax.get_ylim()[1] * 0.97, label,
                rotation=90, fontsize=fontsize, color=color, va='top', ha='right', alpha=0.85)


def _date_locator(ax, labelsize: float = 12) -> None:
    """Weekly major ticks, daily minor ticks for a ~33-day range."""
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis='x', rotation=30, labelsize=labelsize)

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
        """Plot LA mobility metrics with 3-day rolling average and crisis phase shading."""
        print("\n📈 Creating mobility trends plot...")

        delay_col = next(
            (c for c in ["Delay (V_t=35) (Veh-Hours)", "Delay (V_t=35)"]
             if c in self.mobility_df.columns), None
        )
        metrics = [
            ('VMT (Veh-Miles)', '#1f77b4', 'VMT (Vehicle Miles Traveled)'),
            ('TTI', '#ff7f0e', 'TTI (Travel Time Index)'),
            ('VHT (Veh-Hours)', '#2ca02c', 'VHT (Vehicle Hours Traveled)'),
            (delay_col, '#d62728', 'Delay (Veh-Hours)'),
        ]
        metrics = [(m, c, l) for m, c, l in metrics if m and m in self.mobility_df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes_flat = axes.flatten()
        fig.suptitle('LA Wildfire 2025 — Traffic Mobility Metrics (Jan–Feb 2025)',
                     fontsize=20, fontweight='bold', y=0.99)

        for ax_i, (metric, color, label) in enumerate(metrics[:4]):
            ax = axes_flat[ax_i]
            series = self.mobility_df.set_index('Date')[metric].sort_index()
            roll3 = series.rolling(3, center=True, min_periods=1).mean()

            _add_phase_bands(ax)
            ax.plot(series.index, series.values,
                    color=color, linewidth=0.7, alpha=0.4, label='Daily')
            ax.plot(roll3.index, roll3.values,
                    color=color, linewidth=2.2, label='3-day avg')
            _add_event_lines(ax, fontsize=9)

            ax.set_title(label, fontsize=15, fontweight='bold')
            ax.set_ylabel(label.split('(')[0].strip(), fontsize=13)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True, alpha=0.25, axis='y')
            ax.tick_params(axis='y', labelsize=12)
            _date_locator(ax, labelsize=12)

        # Hide unused subplots
        for ax_i in range(len(metrics), 4):
            axes_flat[ax_i].set_visible(False)

        phase_patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
                         for _, _, c, l in _PHASES]
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.legend(handles=phase_patches, loc='lower center', ncol=4,
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

        output_file = OUTPUT_DIR / 'mobility_trends.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_sentiment_timeline(self):
        """Sentiment timeline with 3-day rolling average and crisis phase shading."""
        print("💭 Creating sentiment timeline...")

        if 'avg_compound' not in self.features_df.columns:
            print("  ⚠️  avg_compound column not found; skipping")
            return

        df = self.features_df.set_index('Date').sort_index()

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.suptitle('Reddit Sentiment During LA Wildfire Crisis (Jan–Feb 2025)',
                     fontsize=20, fontweight='bold')

        # Panel 1: Compound sentiment
        ax1 = axes[0]
        compound = df['avg_compound']
        roll3 = compound.rolling(3, center=True, min_periods=1).mean()

        _add_phase_bands(ax1)
        ax1.fill_between(compound.index, compound.values, 0,
                         where=compound.values >= 0, alpha=0.15, color='green')
        ax1.fill_between(compound.index, compound.values, 0,
                         where=compound.values < 0, alpha=0.15, color='red')
        ax1.plot(compound.index, compound.values,
                 color='slategray', linewidth=0.6, alpha=0.4, label='Daily')
        ax1.plot(roll3.index, roll3.values,
                 color='purple', linewidth=2.2, label='3-day avg')
        ax1.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.5)
        _add_event_lines(ax1, fontsize=9)
        ax1.set_ylabel('Compound Score', fontsize=14)
        ax1.set_ylim(-1.1, 1.1)
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.set_title('Compound Sentiment', fontsize=16)
        ax1.tick_params(axis='y', labelsize=12)

        # Panel 2: Positive vs Negative fractions
        ax2 = axes[1]
        _add_phase_bands(ax2)
        if 'pos_fraction' in df.columns:
            pos_r = df['pos_fraction'].rolling(3, center=True, min_periods=1).mean()
            ax2.plot(df.index, df['pos_fraction'].values,
                     color='green', linewidth=0.5, alpha=0.3)
            ax2.plot(pos_r.index, pos_r.values,
                     color='green', linewidth=2, label='Positive (3-day avg)')
        if 'neg_fraction' in df.columns:
            neg_r = df['neg_fraction'].rolling(3, center=True, min_periods=1).mean()
            ax2.plot(df.index, df['neg_fraction'].values,
                     color='red', linewidth=0.5, alpha=0.3)
            ax2.plot(neg_r.index, neg_r.values,
                     color='red', linewidth=2, label='Negative (3-day avg)')
        _add_event_lines(ax2, fontsize=9)
        ax2.set_ylabel('Fraction of Posts', fontsize=14)
        ax2.legend(fontsize=12, loc='upper right')
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.set_title('Positive vs Negative Post Fraction', fontsize=16)
        ax2.tick_params(axis='y', labelsize=12)

        _date_locator(ax2, labelsize=12)
        ax2.set_xlabel('Date', fontsize=14)

        phase_patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
                         for _, _, c, l in _PHASES]
        plt.tight_layout(rect=[0, 0.035, 1, 1])
        fig.legend(handles=phase_patches, loc='lower center', ncol=4,
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

        output_file = OUTPUT_DIR / 'sentiment_timeline.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_features_heatmap(self):
        """Heatmap of binary features with daily resolution (33 days is readable)."""
        print("🔥 Creating features heatmap...")

        binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
        if not binary_cols:
            print("  ⚠️  No binary columns found, skipping heatmap")
            return

        preferred_order = [
            'traffic_congestion_detected', 'evacuation_mentioned',
            'emotion_with_mobility_signal', 'emotion_mobility_mismatch',
            'low_emotion_low_mobility_signal',
            'dominant_emotion_fear', 'high_negative_sentiment',
            'fear_keywords_present', 'anger_mentioned',
            'anxiety_keywords_present', 'sadness_keywords_present',
            'sentiment_shift_detected', 'sentiment_worsened', 'sentiment_improved',
            'high_positive_sentiment', 'mixed_emotions',
            'solidarity_messages', 'policy_governance_discussion',
            'weekend',
        ]
        ordered_cols = [c for c in preferred_order if c in binary_cols]
        remaining = [c for c in binary_cols if c not in ordered_cols]
        ordered_cols = ordered_cols + remaining

        df_indexed = self.features_df.set_index('Date')[ordered_cols].sort_index()

        fig, ax = plt.subplots(figsize=(max(14, len(df_indexed) * 0.45), 8))
        im = ax.imshow(
            df_indexed.T.values,
            aspect='auto',
            cmap='RdYlGn_r',
            vmin=0, vmax=1,
            interpolation='nearest',
        )

        # Y: feature names
        ax.set_yticks(range(len(ordered_cols)))
        ax.set_yticklabels(ordered_cols, fontsize=12)

        # X: daily dates — show every 3rd
        dates = df_indexed.index.tolist()
        tick_pos = list(range(0, len(dates), 3))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([dates[i].strftime('%b %d') for i in tick_pos],
                           rotation=35, fontsize=11)

        # Phase boundary lines
        for start_str, _, _, _ in _PHASES[1:]:
            phase_ts = pd.Timestamp(start_str)
            diffs = [(abs((d - phase_ts).days), i) for i, d in enumerate(dates)]
            closest_i = min(diffs)[1]
            ax.axvline(closest_i - 0.5, color='white', linewidth=1.8, alpha=0.8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label('Feature active (1) / inactive (0)', fontsize=12)
        ax.set_title('Feature Activation Heatmap — LA Wildfire 2025 (daily)',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'features_heatmap.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_combined_mobility_sentiment(self):
        """Three-panel: VMT, compound sentiment, key binary signals with phase shading."""
        print("🔗 Creating combined mobility-sentiment plot...")

        mob = self.mobility_df.set_index('Date').sort_index()
        feat = self.features_df.set_index('Date').sort_index()

        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
        fig.suptitle('LA Wildfire 2025 — Mobility, Sentiment & Key Signals',
                     fontsize=18, fontweight='bold')

        # Panel 1: VMT
        ax1 = axes[0]
        if 'VMT (Veh-Miles)' in mob.columns:
            vmt = mob['VMT (Veh-Miles)']
            roll3 = vmt.rolling(3, center=True, min_periods=1).mean()
            _add_phase_bands(ax1)
            ax1.plot(vmt.index, vmt.values, color='#1f77b4', linewidth=0.6, alpha=0.35)
            ax1.plot(roll3.index, roll3.values, color='#1f77b4', linewidth=2.2,
                     label='VMT (3-day avg)')
            _add_event_lines(ax1, fontsize=10)
        ax1.set_ylabel('Veh-Miles', fontsize=14)
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.set_title('Vehicle Miles Traveled', fontsize=15)
        ax1.tick_params(axis='y', labelsize=12)

        # Panel 2: Compound sentiment
        ax2 = axes[1]
        if 'avg_compound' in feat.columns:
            compound = feat['avg_compound']
            roll3 = compound.rolling(3, center=True, min_periods=1).mean()
            _add_phase_bands(ax2)
            ax2.fill_between(compound.index, compound.values, 0,
                             where=compound.values >= 0, alpha=0.12, color='green')
            ax2.fill_between(compound.index, compound.values, 0,
                             where=compound.values < 0, alpha=0.12, color='red')
            ax2.plot(compound.index, compound.values,
                     color='slategray', linewidth=0.5, alpha=0.35)
            ax2.plot(roll3.index, roll3.values, color='purple', linewidth=2.2,
                     label='Compound sentiment (3-day avg)')
            ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
            _add_event_lines(ax2, fontsize=10)
        ax2.set_ylabel('Sentiment score', fontsize=14)
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend(fontsize=12, loc='lower right')
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.set_title('Reddit Compound Sentiment', fontsize=15)
        ax2.tick_params(axis='y', labelsize=12)

        # Panel 3: Binary signals
        ax3 = axes[2]
        _add_phase_bands(ax3)
        binary_signals = [
            ('traffic_congestion_detected', '#e07b39', 'Traffic congestion'),
            ('evacuation_mentioned', '#d62728', 'Evacuation mentioned'),
            ('solidarity_messages', '#2ca02c', 'Solidarity messages'),
        ]
        offsets = [0.75, 0.45, 0.15]
        for (col, color, label), offset in zip(binary_signals, offsets):
            if col in feat.columns:
                active_days = feat.index[feat[col] == 1]
                ax3.scatter(active_days, [offset] * len(active_days),
                            color=color, s=60, alpha=0.8, label=label, zorder=3)
        ax3.set_yticks(offsets)
        ax3.set_yticklabels(['Traffic\ncongestion', 'Evacuation\nmentioned', 'Solidarity\nmessages'],
                            fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=12, loc='lower right')
        ax3.grid(True, alpha=0.2, axis='y')
        ax3.set_title('Key Binary Signals', fontsize=15)
        _add_event_lines(ax3, fontsize=10)

        _date_locator(ax3, labelsize=12)
        ax3.set_xlabel('Date', fontsize=14)

        phase_patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
                         for _, _, c, l in _PHASES]
        fig.legend(handles=phase_patches, loc='lower center', ncol=4,
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'combined_analysis.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_rules_overview(self):
        """Bubble chart: support vs confidence, bubble size = lift², colour = cross-domain."""
        print("🔵 Creating rules overview chart...")

        rules_path_options = [
            EVALUATED_RULES_FILE,
            RULES_FILE,
        ]
        rules_df = None
        for p in rules_path_options:
            if p.exists():
                rules_df = pd.read_csv(p)
                break
        if rules_df is None or len(rules_df) == 0:
            print("  ⚠️  No rules file found; skipping rules overview")
            return

        fig, ax = plt.subplots(figsize=(11, 7))
        is_cross = rules_df['cross_domain'].astype(str).str.lower().isin(['true', '1', 'yes'])
        for cross_domain, group in rules_df.groupby(is_cross):
            color = '#e07b39' if cross_domain else '#5b9bd5'
            label = 'Cross-domain' if cross_domain else 'Same-domain'
            sizes = (group['lift'].clip(lower=1) ** 2) * 60
            x = group['support_pct'] if 'support_pct' in group.columns else group['support'] * 100
            ax.scatter(x, group['confidence'],
                       s=sizes, c=color, alpha=0.75,
                       edgecolors='white', linewidths=0.5, label=label)

        # Annotate top rules by lift
        top = rules_df.nlargest(min(6, len(rules_df)), 'lift')
        for _, row in top.iterrows():
            x = row.get('support_pct', row['support'] * 100)
            premise_short = str(row['premise'])[:35] + ('…' if len(str(row['premise'])) > 35 else '')
            ax.annotate(premise_short, (x, row['confidence']),
                        fontsize=9, ha='left', va='bottom',
                        xytext=(4, 3), textcoords='offset points', color='#333')

        ax.set_xlabel('Support (%)', fontsize=14)
        ax.set_ylabel('Confidence (%)', fontsize=14)
        ax.set_title('FCA Association Rules — Support vs Confidence\n(bubble size = lift²)',
                     fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=12)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'rules_overview.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_feature_activation(self):
        """Horizontal bar chart of feature activation rates, colour-coded by domain."""
        print("📊 Creating feature activation chart...")

        binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
        if not binary_cols:
            print("  ⚠️  No binary columns found; skipping")
            return

        rates = self.features_df[binary_cols].mean().sort_values(ascending=True) * 100

        mobility_features = {'traffic_congestion_detected', 'evacuation_mentioned'}
        emotion_features = {
            'high_negative_sentiment', 'high_positive_sentiment', 'sentiment_improved',
            'sentiment_worsened', 'sentiment_shift_detected', 'dominant_emotion_fear',
            'mixed_emotions', 'fear_keywords_present', 'anger_mentioned',
            'anxiety_keywords_present', 'sadness_keywords_present', 'solidarity_messages',
        }
        topic_features = {'policy_governance_discussion'}
        colors = []
        for feat in rates.index:
            if feat in mobility_features:
                colors.append('#1f77b4')
            elif feat in emotion_features:
                colors.append('#d62728')
            elif feat in topic_features:
                colors.append('#2ca02c')
            else:
                colors.append('#9467bd')

        fig, ax = plt.subplots(figsize=(10, max(5, len(rates) * 0.38)))
        bars = ax.barh(rates.index, rates.values, color=colors, alpha=0.82, edgecolor='white')
        for bar, val in zip(bars, rates.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', va='center', fontsize=12)

        ax.set_xlabel('% of days active', fontsize=14)
        ax.set_title('Feature Activation Rates — LA Wildfire 2025 (33 days)',
                     fontsize=16, fontweight='bold')
        ax.set_xlim(0, 108)
        ax.grid(True, alpha=0.2, axis='x')
        ax.axvline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=12)

        legend_patches = [
            mpatches.Patch(color='#1f77b4', label='Mobility'),
            mpatches.Patch(color='#d62728', label='Emotion/Sentiment'),
            mpatches.Patch(color='#2ca02c', label='Topic/Discourse'),
            mpatches.Patch(color='#9467bd', label='Composite'),
        ]
        ax.legend(handles=legend_patches, fontsize=12, loc='lower right')

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'feature_activation.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def create_summary_report(self):
        """Generate text summary.

        Rules are ordered by OBJECTIVE STATISTICS only (cross-domain, lift,
        confidence). If the LLM annotation step ran, each rule additionally shows
        a policy-relevance annotation and rationale — but the LLM never selects,
        ranks, or validates rules. Validity comes entirely from the statistical
        pipeline and the tautology filter upstream.
        """
        print("📝 Creating summary report...")

        output_file = OUTPUT_DIR.parent / 'summary_report.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prefer evaluated CSV (with LLM annotation) over raw rules
        rules_df = pd.DataFrame()
        for path in (EVALUATED_RULES_FILE, RULES_FILE):
            try:
                rules_df = pd.read_csv(path)
                break
            except Exception:
                continue

        annotated = "policy_score" in rules_df.columns and rules_df["policy_score"].notna().any()

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
                sort_cols = (
                    ["cross_domain", "lift", "confidence"]
                    if "cross_domain" in rules_df.columns
                    else ["lift", "confidence"]
                )
                top_rules = rules_df.sort_values(sort_cols, ascending=False)
                n_rules = len(top_rules)
                f.write(f"\n\nASSOCIATION RULES (statistically ordered: cross-domain, lift, confidence) — {n_rules} total:\n")
                if annotated:
                    f.write("(LLM provides policy annotation only; it does not rank or select rules.)\n")
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
                    if annotated:
                        policy = row.get("policy_score")
                        if pd.notna(policy):
                            f.write(f"  Policy relevance (LLM annotation): {int(policy)}/10\n")
                        reasoning = row.get("llm_reasoning", "")
                        if reasoning and str(reasoning).strip():
                            f.write(f"  Insight: {reasoning}\n")
                        rec = row.get("llm_policy_recommendation", "")
                        if rec and str(rec).strip():
                            f.write(f"  Recommendation: {rec}\n")

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
    visualizer.plot_rules_overview()
    visualizer.plot_feature_activation()
    visualizer.create_summary_report()

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()