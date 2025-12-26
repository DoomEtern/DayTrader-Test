import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==================== FOREX REALITY CONFIG ====================
DATA_DIR = "forex_ready_data"
OUTPUT_DIR = "forex_reality_results"
INITIAL_CAP = 100000.0

# FOREX FRICTION CONSTANTS
# 1. Spread (measured in pips)
AVG_SPREAD_PIPS = 1.5      
# 2. Swap/Rollover (Daily interest drag, ~0.005% per day)
DAILY_SWAP_RATE = 0.00005  
# 3. Leverage (e.g., 30:1)
LEVERAGE = 30.0            

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ForexRealityEngine:
    @staticmethod
    def calculate_indicators(df, ticker):
        df = df.copy()
        is_jpy = "JPY" in ticker
        pip_unit = 0.01 if is_jpy else 0.0001
        
        # Forex-Specific Indicators
        df['trend_baseline'] = df['Close'].ewm(span=100).mean() # Faster for FX
        df['atr_pips'] = (df['High'] - df['Low']).rolling(14).mean() / pip_unit
        df['resistance'] = df['High'].rolling(20).max().shift(1)
        return df.dropna(), pip_unit

    @staticmethod
    def execute_forex_sim(df, ticker, pip_unit):
        cash_balance = INITIAL_CAP
        position_units = 0
        entry_price = 0
        peak_price = 0
        
        trade_logs = []
        equity_curve = []
        
        friction_loss = 0
        wins, losses, net_pip_profit = 0, 0, 0

        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            atr_val = df['atr_pips'].iloc[i] * pip_unit
            trend = df['trend_baseline'].iloc[i]
            resistance = df['resistance'].iloc[i]

            # 1. POSITION MANAGEMENT (EXIT LOGIC)
            if position_units > 0:
                # Calculate daily swap drag
                swap_cost = (position_units * price) * DAILY_SWAP_RATE
                cash_balance -= swap_cost
                friction_loss += swap_cost
                
                # Trailing Stop based on Volatility
                trailing_stop = peak_price - (2.5 * atr_val)
                
                if low < trailing_stop or price < (trend * 0.998):
                    # APPLY SPREAD ON EXIT
                    exit_price = min(price, trailing_stop) - (AVG_SPREAD_PIPS * pip_unit)
                    
                    pip_gain = (exit_price - entry_price) / pip_unit
                    trade_result = position_units * (exit_price - entry_price)
                    
                    if trade_result > 0: wins += 1
                    else: losses += 1
                    
                    cash_balance += (position_units * exit_price)
                    net_pip_profit += pip_gain
                    friction_loss += (AVG_SPREAD_PIPS * pip_unit * position_units)
                    
                    position_units, entry_price, peak_price = 0, 0, 0
                else:
                    peak_price = max(peak_price, high)

            # 2. ENTRY LOGIC (USING LEVERAGE)
            elif position_units == 0:
                if price > resistance and price > trend:
                    # SPREAD ON ENTRY
                    entry_price = price + (AVG_SPREAD_PIPS * pip_unit)
                    
                    # Risk-based sizing: Use 2% of equity for the stop-loss distance
                    risk_amount = cash_balance * 0.02
                    stop_dist = 2.5 * atr_val
                    
                    # Units = Risk / (Stop Distance in Price)
                    position_units = risk_amount / stop_dist
                    
                    # Check margin (Ensure we aren't exceeding LEVERAGE)
                    notional_value = position_units * entry_price
                    if notional_value > (cash_balance * LEVERAGE):
                        position_units = (cash_balance * LEVERAGE) / entry_price
                    
                    cash_balance -= (position_units * entry_price)
                    peak_price = high
                    friction_loss += (AVG_SPREAD_PIPS * pip_unit * position_units)

            equity_curve.append(cash_balance + (position_units * price))

        return df, {
            'Ticker': ticker,
            'Final_Equity': round(equity_curve[-1], 2),
            'Total_Friction_Loss': round(friction_loss, 2),
            'Net_Pip_Gain': round(net_pip_profit, 1),
            'Win_Rate': round((wins/(wins+losses)*100), 2) if (wins+losses)>0 else 0
        }

def run_forex_audit():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    summaries = []

    for f in files:
        ticker = os.path.basename(f).split('_')[0]
        raw_df = pd.read_csv(f, index_col=0, parse_dates=True)
        data, pip_unit = ForexRealityEngine.calculate_indicators(raw_df, ticker)
        results, stats = ForexRealityEngine.execute_forex_sim(data, ticker, pip_unit)
        summaries.append(stats)
        
    summary_df = pd.DataFrame(summaries)
    print("\n🌍 --- FOREX REALITY AUDIT (WITH SPREAD & SWAPS) --- 🌍")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_forex_audit()
