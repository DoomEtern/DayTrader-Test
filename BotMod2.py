import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==================== REALITY-CHECK CONFIG ====================
DATA_DIR = "bot_ready_data"
OUTPUT_DIR = "reality_check_results"
INITIAL_CAP = 100000.0

# FRICTION CONSTANTS
SLIPPAGE_BPS = 0.0010      
ESTIMATED_TAX_RATE = 0.20  
MIN_TRADE_SIZE = 500      

os.makedirs(OUTPUT_DIR, exist_ok=True)

class RealityEngine:
    @staticmethod
    def calculate_indicators(df):
        df = df.copy()
        df['trend_baseline'] = df['Close'].ewm(span=150).mean()
        df['risk_buffer_atr'] = (df['High'] - df['Low']).rolling(14).mean()
        df['breakout_threshold'] = df['High'].rolling(10).max().shift(1)
        return df.dropna()

    @staticmethod
    def execute_realistic_sim(df, ticker):
        cash_balance = INITIAL_CAP
        share_count = 0
        peak_price = 0
        
        trade_logs = []
        equity_curve = []
        
        total_friction_loss = 0
        wins, losses, total_net_profit = 0, 0, 0

        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            high = df['High'].iloc[i]
            atr = df['risk_buffer_atr'].iloc[i]
            trend = df['trend_baseline'].iloc[i]
            breakout = df['breakout_threshold'].iloc[i]

            if share_count > 0:
                trailing_stop = peak_price - (3.5 * atr)
                
                if price < trailing_stop or price < (trend * 0.98):
                    executed_sell_price = price * (1 - SLIPPAGE_BPS)
                    exit_revenue = share_count * executed_sell_price
                    
                    raw_trade_profit = exit_revenue - entry_capital
                    
                    if raw_trade_profit > 0:
                        tax_bite = raw_trade_profit * ESTIMATED_TAX_RATE
                        actual_profit = raw_trade_profit - tax_bite
                        wins += 1
                    else:
                        tax_bite = 0
                        actual_profit = raw_trade_profit
                        losses += 1
                    
                    total_friction_loss += (entry_capital * SLIPPAGE_BPS) + (exit_revenue * SLIPPAGE_BPS) + tax_bite
                    total_net_profit += actual_profit
                    cash_balance += (exit_revenue - tax_bite)
                    
                    trade_logs.append({'Date': date, 'Type': 'SELL', 'Profit': actual_profit})
                    share_count, peak_price = 0, 0
                else:
                    peak_price = max(peak_price, high)

            elif share_count == 0 and cash_balance > MIN_TRADE_SIZE:
                if price > trend and price > breakout:
                    executed_buy_price = price * (1 + SLIPPAGE_BPS)
                    entry_capital = cash_balance * 0.95
                    
                    share_count = entry_capital / executed_buy_price
                    cash_balance -= entry_capital
                    peak_price = high
                    trade_logs.append({'Date': date, 'Type': 'BUY', 'Price': executed_buy_price})

            equity_curve.append(cash_balance + (share_count * price))

        df['Reality_Equity'] = equity_curve
        df['Market_Growth'] = (df['Close'] / df['Close'].iloc[0]) * INITIAL_CAP
        
        return df, {
            'Ticker': ticker,
            'Reality_Final_Cash': round(equity_curve[-1], 2),
            'Market_Final_Cash': round(df['Market_Growth'].iloc[-1], 2),
            'Net_Profit_After_Tax': round(total_net_profit, 2),
            'Friction_Loss_$': round(total_friction_loss, 2),
            'Win_Rate': round((wins/(wins+losses)*100), 2) if (wins+losses)>0 else 0
        }

def run_reality_audit():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    summaries = []

    for f in files:
        ticker = os.path.basename(f).split('_')[0]
        data = RealityEngine.calculate_indicators(pd.read_csv(f, index_col=0, parse_dates=True))
        results, stats = RealityEngine.execute_realistic_sim(data, ticker)
        summaries.append(stats)
        
        plt.figure(figsize=(12, 5))
        plt.plot(results.index, results['Reality_Equity'], label='Bot (After Tax & Slippage)', color='red')
        plt.plot(results.index, results['Market_Growth'], label='Market (Buy & Hold)', color='gray', alpha=0.5)
        plt.title(f"{ticker} REALITY CHECK")
        plt.legend(); plt.savefig(f"{OUTPUT_DIR}/{ticker}_reality.png"); plt.close()

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{OUTPUT_DIR}/REAL_WORLD_AUDIT.csv", index=False)
    print("\nCHECK SUMMARY")
    print(summary_df.to_string(index=False))
    print(f"\nTotal Portfolio Profit (Post-Tax/Slippage): ${summary_df['Net_Profit_After_Tax'].sum():,.2f}")

if __name__ == "__main__":
    run_reality_audit()
