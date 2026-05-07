"""
Mobility Data Processing Script
Loads and processes PeMS data, calculates baselines
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "LA_Mobility.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_data_processed.csv"
BASELINE_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_baselines.json"

BASELINE_DAYS = 5  # First N days to use as baseline
# ==========================

class MobilityDataProcessor:
    def __init__(self):
        self.data = None
        self.baselines = {}
    
    def load_pems_data(self, filepath):
        """Load and clean PeMS data"""
        print(f"📂 Loading data from {filepath}...")
        
        try:
            self.data = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            print("Please ensure your PeMS data file is in data/raw/")
            sys.exit(1)
        
        # Normalize column names
        self.data.columns = [c.strip() for c in self.data.columns]
        
        # Find date column
        if "Date" in self.data.columns:
            date_col = "Date"
        elif "Day" in self.data.columns:
            date_col = "Day"
        else:
            print("❌ Could not find Date/Day column in PeMS file.")
            print(f"Available columns: {list(self.data.columns)}")
            sys.exit(1)
        
        # Parse date
        self.data["Date"] = pd.to_datetime(self.data[date_col])
        
        print(f"✓ Loaded {len(self.data)} days of data")
        print(f"  Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
        return self.data
    
    def clean_numeric_columns(self):
        """Remove commas, spaces, convert to float"""
        print("🧹 Cleaning numeric columns...")
        
        possible_cols = [
            "VMT (Veh-Miles)",
            "TTI",
            "VHT (Veh-Hours)",
            "Delay (V_t=35) (Veh-Hours)",
            "Delay (V_t=35)",
            "# Lane Points",
            "% Observed",
        ]
        
        for col in possible_cols:
            if col in self.data.columns:
                self.data[col] = (
                    self.data[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("\u00a0", "", regex=False)  # non-breaking space
                    .str.replace(" ", "", regex=False)
                    .replace("", np.nan)
                    .astype(float)
                )
        
        print("✓ Numeric columns cleaned")
    
    def interpolate_missing(self):
        """Fill missing values using linear interpolation"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        missing_before = self.data[numeric_cols].isna().sum().sum()
        
        if missing_before > 0:
            print(f"⚠️  Found {missing_before} missing values, interpolating...")
            self.data[numeric_cols] = self.data[numeric_cols].interpolate(method='linear')
            missing_after = self.data[numeric_cols].isna().sum().sum()
            print(f"✓ Missing values reduced to {missing_after}")
    
    def calculate_baselines(self):
        """Calculate baseline values from first N days"""
        print(f"\n📊 Calculating baselines (first {BASELINE_DAYS} days)...")
        
        baseline_data = self.data.sort_values('Date').head(BASELINE_DAYS)
        
        # Find delay column
        if "Delay (V_t=35) (Veh-Hours)" in self.data.columns:
            delay_col = "Delay (V_t=35) (Veh-Hours)"
        elif "Delay (V_t=35)" in self.data.columns:
            delay_col = "Delay (V_t=35)"
        else:
            print("⚠️  Warning: Could not find Delay column")
            delay_col = None
        
        metrics = {
            "VMT (Veh-Miles)": "VMT",
            "TTI": "TTI",
            "VHT (Veh-Hours)": "VHT"
        }
        
        if delay_col:
            metrics[delay_col] = "Delay"
        
        for col, name in metrics.items():
            if col in baseline_data.columns:
                self.baselines[name] = {
                    'mean': float(baseline_data[col].mean()),
                    'std': float(baseline_data[col].std()),
                    'median': float(baseline_data[col].median()),
                    'min': float(baseline_data[col].min()),
                    'max': float(baseline_data[col].max())
                }
                print(f"  {name:8s} → Mean: {self.baselines[name]['mean']:,.2f}")
        
        # Save baselines
        import json
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_FILE, 'w') as f:
            json.dump(self.baselines, f, indent=2)
        print(f"\n✓ Baselines saved to {BASELINE_FILE}")
        
        return self.baselines
    
    def save_processed_data(self, filename):
        """Save processed mobility data"""
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filename, index=False)
        print(f"✓ Saved processed data to {filename}")


def main():
    print("=" * 70)
    print("MOBILITY DATA PROCESSING")
    print("=" * 70)
    
    processor = MobilityDataProcessor()
    
    # Load data
    processor.load_pems_data(INPUT_FILE)
    
    # Clean and interpolate
    processor.clean_numeric_columns()
    processor.interpolate_missing()
    
    # Calculate baselines
    baselines = processor.calculate_baselines()
    
    # Save processed data
    processor.save_processed_data(OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("✓ MOBILITY DATA PROCESSING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()