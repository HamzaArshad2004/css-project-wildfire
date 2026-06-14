"""
Script for Formal Concept Analysis and Galois Lattice generation
Requires: pip install pandas numpy concepts networkx matplotlib
"""

import math
import pandas as pd
from concepts import Context
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY_MATRIX = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "fca"

# ---------------------------------------------------------------------------
# Tautology definitions: a rule "premise → conclusion" is definitionally true
# (and should be excluded) if the premise contains at least one feature from
# EVERY component group listed for that conclusion.
# ---------------------------------------------------------------------------
TAUTOLOGY_DEFINITIONS: dict[str, list[set[str]]] = {
    # emotion_with_mobility_signal = (any negative emotion) AND (any mobility: traffic OR evacuation)
    # Tautological if premise already contains one emotion AND one mobility feature.
    "emotion_with_mobility_signal": [
        {"dominant_emotion_fear", "fear_keywords_present", "high_negative_sentiment",
         "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present"},
        {"traffic_congestion_detected", "evacuation_mentioned"},
    ],
    # emotion_mobility_mismatch = (any emotion) AND NOT (any mobility)
    # If premise has both emotion AND mobility features, the mismatch conclusion can't hold.
    "emotion_mobility_mismatch": [
        {"dominant_emotion_fear", "fear_keywords_present", "high_negative_sentiment",
         "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present"},
        {"traffic_congestion_detected", "evacuation_mentioned"},
    ],
    # emotion_with_mobility_signal already embeds a mobility signal by definition.
    # Concluding traffic or evacuation from a premise containing that composite is
    # just unpacking it — tautological.
    "traffic_congestion_detected": [
        {"emotion_with_mobility_signal"},
    ],
    "evacuation_mentioned": [
        {"emotion_with_mobility_signal"},
    ],
    # sentiment_shift_detected = sentiment_improved OR sentiment_worsened — definitional.
    "sentiment_shift_detected": [
        {"sentiment_improved", "sentiment_worsened"},
    ],
    # low_emotion_low_mobility_signal = NOT(any emotion) AND NOT(any mobility).
    # Cannot hold when any emotion or mobility feature is in the premise.
    "low_emotion_low_mobility_signal": [
        {"dominant_emotion_fear", "fear_keywords_present", "high_negative_sentiment",
         "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present",
         "traffic_congestion_detected", "evacuation_mentioned",
         "emotion_with_mobility_signal"},
    ],
}

MOBILITY_FEATURES = frozenset({
    "traffic_congestion_detected",
    "evacuation_mentioned",
})

EMOTION_FEATURES = frozenset({
    "high_negative_sentiment", "dominant_emotion_fear", "fear_keywords_present",
    "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present",
    "high_positive_sentiment", "mixed_emotions", "solidarity_messages",
    "sentiment_worsened", "sentiment_improved", "sentiment_shift_detected",
})


def _is_tautological(premise: tuple[str, ...], conclusion: str) -> bool:
    """Return True if conclusion is definitionally implied by the premise."""
    defs = TAUTOLOGY_DEFINITIONS.get(conclusion)
    if defs is None:
        return False
    premise_set = set(premise)
    return all(any(f in premise_set for f in group) for group in defs)


def _is_cross_domain(premise: tuple[str, ...], conclusion: str) -> bool:
    """Return True if the rule spans both mobility and emotion domains."""
    all_features = set(premise) | {conclusion}
    return bool(all_features & MOBILITY_FEATURES) and bool(all_features & EMOTION_FEATURES)


class FCAAnalyzer:
    def __init__(self, binary_matrix_file):
        """
        Initialize FCA analyzer

        Args:
            binary_matrix_file: CSV file with binary features (Date as first column)
        """
        self.binary_matrix_file = Path(binary_matrix_file)
        self.df = pd.read_csv(self.binary_matrix_file)
        self.date_col = self.df.iloc[:, 0]

        candidate_data = self.df.iloc[:, 1:].copy()
        binary_columns = []

        for column in candidate_data.columns:
            numeric_values = pd.to_numeric(candidate_data[column], errors='coerce')
            raw_non_null = candidate_data[column].notna().sum()
            converted_non_null = numeric_values.notna().sum()

            if raw_non_null == 0 or converted_non_null != raw_non_null:
                continue

            unique_values = set(numeric_values.dropna().unique())
            if unique_values.issubset({0, 1}):
                binary_columns.append(column)
                candidate_data[column] = numeric_values.astype(int)

        self.binary_data = candidate_data[binary_columns]
        self.context = None
        self.lattice = None

    def create_formal_context(self):
        """Create formal context for FCA"""
        objects = [f"Day_{i}" for i in range(len(self.binary_data))]
        properties = list(self.binary_data.columns)

        bools = []
        for _, row in self.binary_data.iterrows():
            bools.append(tuple(row.astype(bool)))

        self.context = Context(objects, properties, bools)
        return self.context

    def generate_lattice(self):
        """Generate Galois lattice"""
        if self.context is None:
            self.create_formal_context()

        self.lattice = self.context.lattice
        return self.lattice

    def extract_concepts(self):
        """Extract all formal concepts"""
        if self.lattice is None:
            self.generate_lattice()

        concepts = []
        for concept in self.lattice:
            concepts.append({
                'extent': concept.extent,
                'intent': concept.intent,
                'support': len(concept.extent),
            })

        return pd.DataFrame(concepts)

    def extract_implications(
        self,
        min_support=0.05,
        max_premise_size=2,
        min_confidence=0.8,
        min_lift=1.05,
        max_conclusion_prevalence=0.9,
    ):
        """
        Extract implication rules from the binary matrix with quality filters.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            max_premise_size: Max number of attributes in premise
            min_confidence: Minimum rule confidence (0.0 to 1.0)
            min_lift: Minimum rule lift (>1 means positive association)
            max_conclusion_prevalence: Ignore overly common conclusions

        Prints a short filtering summary so the rule funnel is reportable:
            candidate (premise, conclusion) pairs evaluated,
            rules passing the statistical thresholds (support/conf/lift),
            rules removed by the tautology filter, and the final count.
        Note: because the statistical gates are applied inline during mining,
        "rules passing thresholds" is the earliest countable stage — there is
        no separate ungated raw-rule list.
        """
        if self.context is None:
            self.create_formal_context()

        implications = []
        # --- Rule-funnel counters (for reporting / cross-case table) ---
        candidate_pairs = 0      # (premise, conclusion) pairs that cleared support + prevalence
        passed_thresholds = 0    # of those, how many cleared confidence + lift
        tautology_removed = 0    # of those, how many the tautology filter dropped
        min_support_count = max(2, math.ceil(min_support * len(self.binary_data)))

        feature_names = list(self.binary_data.columns)
        total_rows = len(self.binary_data)

        conclusion_prevalence = {
            col: float(self.binary_data[col].mean()) for col in feature_names
        }

        for premise_size in range(1, min(max_premise_size, len(feature_names)) + 1):
            for premise_tuple in combinations(feature_names, premise_size):
                premise_mask = self.binary_data[list(premise_tuple)].all(axis=1)
                support_count = int(premise_mask.sum())

                if support_count < min_support_count:
                    continue

                for conclusion in feature_names:
                    if conclusion in premise_tuple:
                        continue

                    p_conclusion = conclusion_prevalence[conclusion]
                    if p_conclusion == 0 or p_conclusion > max_conclusion_prevalence:
                        continue

                    candidate_pairs += 1

                    conclusion_support = int(self.binary_data.loc[premise_mask, conclusion].sum())
                    confidence = conclusion_support / support_count
                    lift = confidence / p_conclusion
                    leverage = (conclusion_support / total_rows) - (
                        (support_count / total_rows) * p_conclusion
                    )

                    if confidence >= min_confidence and lift >= min_lift:
                        passed_thresholds += 1
                        if _is_tautological(premise_tuple, conclusion):
                            tautology_removed += 1
                            continue
                        implications.append({
                            'premise': ', '.join(premise_tuple),
                            'conclusion': conclusion,
                            'support': support_count,
                            'support_pct': support_count / total_rows * 100,
                            'confidence': confidence * 100,
                            'lift': lift,
                            'leverage': leverage,
                            'cross_domain': _is_cross_domain(premise_tuple, conclusion),
                        })

        # --- Filtering summary -------------------------------------------------
        print("\n  Rule extraction summary:")
        print(f"    Candidate (premise -> conclusion) pairs evaluated : {candidate_pairs}")
        print(f"    Passed thresholds (support/conf/lift)             : {passed_thresholds}")
        print(f"    Removed by tautology filter                       : {tautology_removed}")
        print(f"    Final rules retained                              : {len(implications)}")

        return pd.DataFrame(implications)

    def visualize_lattice(self, output_file='lattice_visualization.png'):
        """Visualize the Galois lattice"""
        if self.lattice is None:
            self.generate_lattice()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        G = nx.DiGraph()
        concept_labels = {}
        for i, concept in enumerate(self.lattice):
            G.add_node(i)
            attrs = list(concept.intent)[:3]
            label = f"C{i}\\n{len(concept.extent)} days"
            if attrs:
                label += f"\\n{', '.join(attrs)}"
            concept_labels[i] = label

        concepts_list = list(self.lattice)
        for i, concept in enumerate(concepts_list):
            for j, other_concept in enumerate(concepts_list):
                if i != j:
                    if (
                        set(concept.extent).issubset(set(other_concept.extent))
                        and set(other_concept.intent).issubset(set(concept.intent))
                        and len(set(concept.extent)) < len(set(other_concept.extent))
                    ):
                        G.add_edge(j, i)

        plt.figure(figsize=(16, 12), constrained_layout=True)
        pos = nx.spring_layout(G, k=2, iterations=50)

        nx.draw(
            G,
            pos,
            labels=concept_labels,
            node_color='lightblue',
            node_size=3000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            linewidths=2,
            with_labels=True,
        )

        plt.title("Galois Lattice - Crisis Behavior Formal Concept Analysis", fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Lattice visualization saved to {output_path}")

    def export_for_galicia(self, output_file='galicia_context.cxt'):
        """Export formal context in Galicia .cxt format"""
        if self.context is None:
            self.create_formal_context()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("B\\n\\n")
            f.write(f"{len(self.context.objects)}\\n")
            f.write(f"{len(self.context.properties)}\\n\\n")

            for obj in self.context.objects:
                f.write(f"{obj}\\n")

            for prop in self.context.properties:
                f.write(f"{prop}\\n")

            for row in self.context.bools:
                row_str = ''.join(['X' if val else '.' for val in row])
                f.write(f"{row_str}\\n")

        print(f"Context exported to Galicia format: {output_path}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = FCAAnalyzer(DEFAULT_BINARY_MATRIX)

    context = analyzer.create_formal_context()
    print(f"Created formal context with {len(context.objects)} objects and {len(context.properties)} properties")

    lattice = analyzer.generate_lattice()
    print(f"Generated lattice with {len(list(lattice))} concepts")

    concepts_df = analyzer.extract_concepts()
    concepts_df.to_csv(RESULTS_DIR / 'formal_concepts.csv', index=False)
    print(f"Extracted {len(concepts_df)} formal concepts")

    rules_df = analyzer.extract_implications(
        min_support=0.10,
        max_premise_size=2,
        min_confidence=0.8,
        min_lift=1.05,
        max_conclusion_prevalence=0.75,
    )
    if not rules_df.empty:
        rules_df = rules_df.sort_values(
            ['cross_domain', 'lift', 'confidence', 'support'],
            ascending=[False, False, False, False],
        )
    rules_df.to_csv(RESULTS_DIR / 'association_rules.csv', index=False)
    print(f"\nTotal rules saved: {len(rules_df)}")
    print("\nTop 10 Association Rules:")
    print(rules_df.head(10))

    analyzer.visualize_lattice(RESULTS_DIR / 'galois_lattice.png')
    analyzer.export_for_galicia(RESULTS_DIR / 'crisis_context.cxt')