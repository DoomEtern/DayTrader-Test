import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from scipy.optimize import minimize

# ==================== MASTER CONFIGURATION ====================
INPUT_PATH = "stage4_elite_audit/elite_audit_summary.csv"
OUTPUT_DIR = "stage5_portfolio"

MAX_ASSETS = 8
MIN_ASSETS = 3
MAX_ALLOCATION = 0.35
MIN_ALLOCATION = 0.05
CASH_BUFFER = 0.05

BEAR_MARKET_SHARPE = 0.2
CRISIS_MARKET_MDD = -0.15

CORRELATION_PENALTY = 0.5
TARGET_VOLATILITY = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== METRICS ====================
class AdvancedMetrics:
    @staticmethod
    def calculate_conviction(row):
        mdd_penalty = abs(row['GBM_MDD']) * 3.0
        alpha_boost = row['Alpha_Added'] * 5.0
        sharpe_base = row['GBM_Sharpe'] * 2.0
        return sharpe_base + alpha_boost - mdd_penalty

# ==================== REGIME ====================
class RegimeDetector:
    @staticmethod
    def detect_regime(df):
        top = df.sort_values('GBM_Sharpe', ascending=False).head(5)
        if top.empty:
            return "CRISIS"
        if top['GBM_MDD'].mean() < CRISIS_MARKET_MDD:
            return "CRISIS"
        if top['GBM_Sharpe'].mean() < BEAR_MARKET_SHARPE:
            return "BEAR"
        return "BULL"

# ==================== OPTIMIZER ====================
class CitadelGradeOptimizer:
    @staticmethod
    def get_correlation_matrix(tickers):
        data = {}
        for t in tickers:
            try:
                df = pd.read_csv(f"bot_ready_data/{t}_data.csv", index_col=0)
                data[t] = df['Close'].pct_change()
            except:
                continue
        returns = pd.DataFrame(data).dropna()
        return returns.corr(), returns.cov()

    @staticmethod
    def optimize_risk_parity(assets):
        tickers = assets['Ticker'].tolist()
        corr, cov = CitadelGradeOptimizer.get_correlation_matrix(tickers)

        if corr.empty:
            return assets['Apex_Score'] / assets['Apex_Score'].sum()

        n = len(tickers)
        x0 = np.array([1 / n] * n)
        bounds = [(0.02, 0.40)] * n
        cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        def objective(w):
            var = np.dot(w.T, np.dot(cov, w))
            corr_penalty = np.sum(np.dot(w.T, np.dot(corr, w))) * CORRELATION_PENALTY
            return var + corr_penalty

        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else x0

# ==================== ENGINE ====================
class ApexPortfolioEngine:
    def __init__(self):
        if not os.path.exists(INPUT_PATH):
            sys.exit(f"FATAL: Missing {INPUT_PATH}")
        self.raw_df = pd.read_csv(INPUT_PATH)
        self.regime = "UNKNOWN"
        self.portfolio = None
        self.diagnostics = []

    def log(self, msg):
        print(f"[ApexEngine] {msg}")
        self.diagnostics.append(msg)

    def run_selection_waterfall(self):
        df = self.raw_df.copy()
        self.regime = RegimeDetector.detect_regime(df)
        self.log(f"Market Regime Detected: {self.regime}")

        t1 = df[(df['GBM_Sharpe'] > 0.5) & (df['Alpha_Added'] > 0) & (df['GBM_MDD'] > -0.2)]
        t2 = df[(df['GBM_Sharpe'] > 0) & (df['GBM_MDD'] > -0.3)]
        t3 = df.sort_values('GBM_Sharpe', ascending=False).head(MIN_ASSETS)

        if self.regime == "BULL" and len(t1) >= MIN_ASSETS:
            selected = t1
        elif self.regime in ["BULL", "BEAR"] and len(t2) >= MIN_ASSETS:
            selected = t2
        else:
            selected = t3

        selected['Apex_Score'] = selected.apply(AdvancedMetrics.calculate_conviction, axis=1)
        selected = selected.sort_values('Apex_Score', ascending=False).head(MAX_ASSETS)
        self.log(f"Selected {len(selected)} assets")
        return selected

    # ===== COMBINED ALLOCATION LOGIC =====
    def allocate_weights(self, assets):
        self.log("Running Covariance-Aware Risk Parity Optimization")
        optimized = CitadelGradeOptimizer.optimize_risk_parity(assets)

        exposure = 0.95
        if self.regime == "CRISIS":
            exposure = 0.40
            self.log("CRISIS MODE: Exposure capped at 40%")
        elif self.regime == "BEAR":
            exposure = 0.70

        df = assets.copy()
        df['Final_Weight'] = optimized * exposure

        total = df['Final_Weight'].sum()
        cash = pd.DataFrame([{
            'Ticker': 'CASH_USD',
            'Final_Weight': 1 - total,
            'GBM_Sharpe': 0,
            'GBM_MDD': 0,
            'Apex_Score': 0
        }])

        self.portfolio = pd.concat([df, cash], ignore_index=True)
        self.log(f"Effective Bets: {round(1/np.sum(optimized**2), 2)}")

    def generate_report(self):
        path = f"{OUTPUT_DIR}/portfolio_report.txt"
        with open(path, "w") as f:
            f.write(f"APEX ENGINE REPORT | {datetime.now()}\n")
            f.write("="*40 + "\n")
            f.write(f"REGIME: {self.regime}\n")
            f.write("-"*40 + "\n")
            for d in self.diagnostics:
                f.write(f"> {d}\n")
            f.write("="*40 + "\n")
            f.write(self.portfolio.to_string(index=False))

        self.portfolio.to_csv(f"{OUTPUT_DIR}/final_allocations.csv", index=False)
        self.log("Report generated")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\nINITIALIZING PORTFOLIO ENGINE...")
    engine = ApexPortfolioEngine()
    assets = engine.run_selection_waterfall()
    engine.allocate_weights(assets)
    engine.generate_report()

    print("\nFINAL PORTFOLIO")
    print(engine.portfolio[['Ticker', 'Final_Weight']].to_string(index=False))
