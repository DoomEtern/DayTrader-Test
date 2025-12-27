import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==================== ELITE CONFIGURATION ====================
BLOCK_SIZE = 10       
SIMULATIONS = 10000
HORIZON = 21            # 1-Month Risk Window
CONFIDENCE_LEVEL = 0.99 
DATA_DIR = "bot_ready_data"
OUTPUT_DIR = "stage6_extreme_audit"

class InstitutionalRiskEngine:
    def __init__(self):
        self.portfolio = pd.read_csv("stage5_portfolio/final_allocations.csv")
        self.tickers = self.portfolio[self.portfolio['Ticker'] != 'CASH_USD']['Ticker'].tolist()
        self.weights = self.portfolio[self.portfolio['Ticker'] != 'CASH_USD'].set_index('Ticker')['Final_Weight']
        self.cash_weight = self.portfolio[self.portfolio['Ticker'] == 'CASH_USD']['Final_Weight'].values[0]

    def get_joint_returns(self):
        all_rets = []
        for t in self.tickers:
            df = pd.read_csv(f"{DATA_DIR}/{t}_data.csv", index_col=0, parse_dates=True)
            all_rets.append(df['Close'].pct_change().rename(t))
        joint_df = pd.concat(all_rets, axis=1).dropna()
        return joint_df

    def run_block_bootstrap(self, joint_df):
        n_obs = len(joint_df)
        sim_paths = np.zeros((HORIZON, SIMULATIONS))

        for s in range(SIMULATIONS):
            path_rets = []
            while len(path_rets) < HORIZON:
                start_idx = np.random.randint(0, n_obs - BLOCK_SIZE)
                block = joint_df.iloc[start_idx : start_idx + BLOCK_SIZE]
 
                port_block_rets = block.dot(self.weights.values)
                path_rets.extend(port_block_rets.tolist())
            
            sim_paths[:, s] = np.cumprod(1 + np.array(path_rets[:HORIZON]))

        return sim_paths

    def calculate_evt_metrics(self, final_returns):
        final_rets = final_returns - 1
        var_limit = np.percentile(final_rets, (1 - CONFIDENCE_LEVEL) * 100)

        es = final_rets[final_rets <= var_limit].mean()
        
        return var_limit, es

    def execute(self):
        print("🚀 INITIATING MULTIVARIATE BLOCK-BOOTSTRAP...")
        joint_df = self.get_joint_returns()

        paths = self.run_block_bootstrap(joint_df)
        final_returns = paths[-1, :]
        
        # Metrics
        var_99, es_99 = self.calculate_evt_metrics(final_returns)
        
        print(f"\n--- 21-DAY INSTITUTIONAL AUDIT (99% CONFIDENCE) ---")
        print(f"Horizon: {HORIZON} Trading Days")
        print(f"Joint VaR: {var_99:.2%}")
        print(f"Expected Shortfall: {es_99:.2%}")
        print(f"Max Simulated Drawdown: {(final_returns.min() - 1):.2%}")
        
        # Save visualization
        plt.figure(figsize=(10, 6))
        plt.hist(final_returns - 1, bins=100, color='navy', alpha=0.7)
        plt.axvline(var_99, color='red', linestyle='--', label=f'99% VaR ({var_99:.2%})')
        plt.title(f"Multivariate Portfolio Risk: {HORIZON}-Day Horizon")
        plt.legend()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/joint_tail_risk.png")

if __name__ == "__main__":
    engine = InstitutionalRiskEngine()
    engine.execute()
