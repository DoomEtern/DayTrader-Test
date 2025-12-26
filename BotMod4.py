import pandas as pd
import numpy as np
import os
import sys

# ==================== CONFIGURATION ====================
STAGE4_PATH = "stage4_forex_audit/audit_summary.csv"  # Stage 4 output
OUTPUT_DIR = "stage5_portfolio"

# FX Specific Caps
MAX_SINGLE_PAIR = 0.25       # Max 25% allocation per currency pair
MAX_CURRENCY_EXPOSURE = 0.40 # Max 40% exposure to any single currency
MIN_PAIRS = 3                # Minimum pairs for full capital deployment

# Qualification thresholds
EQUITY_THRESHOLD = 100000    # Minimum final equity to consider
WIN_RATE_THRESHOLD = 0.48    # Minimum win rate to qualify

# Cash
CASH_TICKER = "CASH_USD"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Stage5ForexPortfolio:
    def __init__(self, input_csv):
        self.input_csv = input_csv
        self.weights_df = None
        self.qualified = None

    def load_and_filter(self):
        """Load Stage 4 audit and filter high-quality signals."""
        if not os.path.exists(self.input_csv):
            sys.exit(f"FATAL: Stage 4 data missing: {self.input_csv}")

        df = pd.read_csv(self.input_csv)
        required_cols = ['Pair', 'Final_Equity', 'Win_Rate', 'Net_Pip_Gain']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            sys.exit(f"FATAL: Stage 4 CSV missing columns: {missing}")

        # Filter for quality signals
        mask = (df['Final_Equity'] >= EQUITY_THRESHOLD) & (df['Win_Rate'] >= WIN_RATE_THRESHOLD)
        self.qualified = df[mask].copy()

        if self.qualified.empty:
            print("🚨 NO QUALIFIED SIGNALS. PARKING IN CASH.")
            self._park_in_cash()
            sys.exit(0)

    def calculate_fx_weights(self):
        """Compute pair weights with caps and currency exposure limits."""
        df = self.qualified.copy()

        # Conviction score: combine equity and win rate
        df['Conviction'] = df['Final_Equity'] * df['Win_Rate']

        # Raw weights proportional to squared conviction (convex scaling)
        total_conviction = (df['Conviction'] ** 2).sum()
        df['Raw_Weight'] = (df['Conviction'] ** 2) / total_conviction

        # Apply pair and currency caps
        weights = {}
        currency_exposure = {}

        for _, row in df.iterrows():
            pair = row['Pair'].replace('=X', '')
            base, quote = pair[:3], pair[3:]

            weight = min(row['Raw_Weight'], MAX_SINGLE_PAIR)

            # Check currency exposure
            if currency_exposure.get(base, 0) + weight > MAX_CURRENCY_EXPOSURE:
                weight = max(0, MAX_CURRENCY_EXPOSURE - currency_exposure.get(base, 0))
            if currency_exposure.get(quote, 0) + weight > MAX_CURRENCY_EXPOSURE:
                weight = max(0, MAX_CURRENCY_EXPOSURE - currency_exposure.get(quote, 0))

            weights[pair] = weight
            currency_exposure[base] = currency_exposure.get(base, 0) + weight
            currency_exposure[quote] = currency_exposure.get(quote, 0) + weight

        # If not enough pairs, scale down overall risk proportionally
        total_pairs = len(weights)
        allocation_factor = 1.0
        if total_pairs < MIN_PAIRS:
            allocation_factor = total_pairs / MIN_PAIRS
            print(f"⚠️ Diversification shortfall ({total_pairs}/{MIN_PAIRS}). Scaling risk to {allocation_factor:.1%}")
            for k in weights:
                weights[k] *= allocation_factor

        # Build final DataFrame
        self.weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])

        # Add cash buffer
        total_allocated = self.weights_df['Weight'].sum()
        cash_row = pd.DataFrame([{'Ticker': CASH_TICKER, 'Weight': 1.0 - total_allocated}])
        self.weights_df = pd.concat([self.weights_df, cash_row], ignore_index=True)
        self.weights_df.to_csv(f"{OUTPUT_DIR}/portfolio_weights.csv", index=False)

    def _park_in_cash(self):
        """100% cash scenario."""
        pd.DataFrame([{'Ticker': CASH_TICKER, 'Weight': 1.0}]).to_csv(
            f"{OUTPUT_DIR}/portfolio_weights.csv", index=False
        )

    def print_summary(self):
        print("\n" + "="*40)
        print("🌍 FX PORTFOLIO ENGINE: FINAL WEIGHTS")
        print("="*40)
        print(self.weights_df.to_string(index=False))
        print("="*40)

if __name__ == "__main__":
    engine = Stage5ForexPortfolio(STAGE4_PATH)
    engine.load_and_filter()
    engine.calculate_fx_weights()
    engine.print_summary()
