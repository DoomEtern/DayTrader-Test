# stage4_fx_alpha_audit.py

import pandas as pd
import numpy as np
import os
import glob
from BotMod2 import ForexRealityEngine, DATA_DIR  # your Stage 3 code

# ==================== CONFIGURATION ====================
OUTPUT_DIR = "stage4_forex_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FXAlphaAudit:
    @staticmethod
    def compute_sharpe(returns, risk_free=0.0):
        """Compute annualized Sharpe ratio (daily returns)."""
        if len(returns) == 0:
            return 0.0
        mean_ret = returns.mean() - risk_free / 252
        std_ret = returns.std()
        return mean_ret / (std_ret + 1e-9) * np.sqrt(252)

    @staticmethod
    def compute_max_drawdown(equity_curve):
        """Compute max drawdown from equity curve."""
        cum_ret = np.array(equity_curve)
        peak = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - peak) / peak
        return drawdown.min()

    @staticmethod
    def alpha_vs_baseline(equity_curve, baseline_curve):
        """Alpha = difference in final cumulative return vs baseline."""
        if len(equity_curve) == 0 or len(baseline_curve) == 0:
            return 0.0
        return equity_curve[-1] - baseline_curve[-1]

    @classmethod
    def audit(cls):
        """
        Runs Stage 4 audit on all Stage 3 outputs.
        Expects CSVs in DATA_DIR from Stage 3 simulation.
        Each CSV must have columns: 'Close', 'High', 'Low'.
        """
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not files:
            print(f"🚨 No Stage 3 data found in {DATA_DIR}")
            return pd.DataFrame()

        results = []

        for f in files:
            pair = os.path.basename(f).split('_')[0]
            df_raw = pd.read_csv(f, index_col=0, parse_dates=True)
            
            # Stage 3 indicators & simulation
            df, pip_unit = ForexRealityEngine.calculate_indicators(df_raw, pair)
            df_sim, stats = ForexRealityEngine.execute_forex_sim(df, pair, pip_unit)
            
            # Daily returns for Sharpe
            returns = df_sim['Close'].pct_change().fillna(0)
            
            # Baseline: simple linear moving average strategy (50 EMA)
            df['linear_signal'] = (df['Close'] > df['Close'].ewm(span=50).mean()).astype(float)
            df['linear_ret'] = df['linear_signal'].shift(1) * df['Close'].pct_change().fillna(0)
            baseline_curve = (1 + df['linear_ret']).cumprod().values
            
            # Metrics
            sharpe = cls.compute_sharpe(returns)
            linear_sharpe = cls.compute_sharpe(df['linear_ret'])
            alpha = cls.alpha_vs_baseline(df_sim['Close'].values, baseline_curve)
            mdd = cls.compute_max_drawdown(df_sim['Close'].values)
            
            results.append({
                'Pair': pair,
                'ML_Sharpe': round(sharpe, 4),
                'Linear_Sharpe': round(linear_sharpe, 4),
                'Alpha_Added': round(alpha, 4),
                'ML_MDD': round(mdd, 4),
                'Final_Equity': float(stats['Final_Equity']),
                'Win_Rate': float(stats['Win_Rate']),
                'Net_Pip_Gain': float(stats['Net_Pip_Gain'])
            })
        
        result_df = pd.DataFrame(results).sort_values(by='Alpha_Added', ascending=False)
        output_path = os.path.join(OUTPUT_DIR, "audit_summary.csv")
        result_df.to_csv(output_path, index=False)
        print(f"✅ Stage 4 FX Alpha Audit Complete! Results saved to {output_path}")
        return result_df

if __name__ == "__main__":
    FXAlphaAudit.audit()
