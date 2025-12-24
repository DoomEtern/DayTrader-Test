import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler # Better for outliers than StandardScaler

# ==================== CONFIGURATION ====================
INPUT_DIR = "bot_ready_data"
OUTPUT_DIR = "stage4_elite_audit"
WALK_TRAIN = 252 * 2  # 2 years of history for "Elite" stability
WALK_TEST = 21        # 1 month test
TARGET_VOL = 0.025    # Slightly higher threshold for "Conviction" trades
FRICTION = 0.0004     # 0.04% Institutional execution cost

os.makedirs(OUTPUT_DIR, exist_ok=True)

class AlphaAuditor:
    @staticmethod
    def engineer_elite_features(df):
        df = df.copy()
        # Add 'Regime' features
        df['Vol_ZScore'] = (df['Close'].pct_change().rolling(20).std() / 
                            df['Close'].pct_change().rolling(200).std())
        df['Dist_from_High'] = (df['Close'] / df['Close'].rolling(252).max()) - 1
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def get_optimized_params(ticker, vol_regime):
        """Dynamic tuning based on Ticker Personality & Volatility."""
        # Tech giants get 'shallow' trees to prevent overfitting to noise
        if ticker in ['NVDA', 'AMZN', 'META']:
            return {'depth': 3, 'lr': 0.03, 'n': 80}
        # Stable stocks get 'deeper' trees to find subtle patterns
        if vol_regime < 1.0:
            return {'depth': 5, 'lr': 0.05, 'n': 100}
        return {'depth': 4, 'lr': 0.04, 'n': 90}

    @staticmethod
    def simulate(df, ticker):
        df = EliteAlphaAuditor.engineer_elite_features(df)
        
        # 1. Clean & Split
        drop_cols = ['Open','High','Low','Close','Volume','Target','Fwd_Ret','Ticker','News_Summary']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        
        fwd_ret = df['Close'].shift(-5) / df['Close'] - 1
        y = np.where(fwd_ret > TARGET_VOL, 1, np.where(fwd_ret < -TARGET_VOL, -1, 0))
        
        all_returns = []
        
        # 2. Advanced Walk-Forward
        for i in range(WALK_TRAIN, len(X) - WALK_TEST, WALK_TEST):
            X_tr, y_tr = X.iloc[i-WALK_TRAIN:i], y[i-WALK_TRAIN:i]
            X_te = X.iloc[i:i+WALK_TEST]
            
            # Robust Scaling (Elite handle for NVDA swings)
            scaler = RobustScaler()
            imputer = SimpleImputer(strategy='median')
            
            try:
                X_tr_s = scaler.fit_transform(imputer.fit_transform(X_tr))
                X_te_s = scaler.transform(imputer.transform(X_te))
            except: continue

            # Get Elite Params
            vol_now = df['Vol_ZScore'].iloc[i]
            p = EliteAlphaAuditor.get_optimized_params(ticker, vol_now)
            
            model = GradientBoostingClassifier(
                n_estimators=p['n'], 
                learning_rate=p['lr'], 
                max_depth=p['depth'],
                subsample=0.8 # Stochastic gradient boosting for better generalization
            )
            
            model.fit(X_tr_s, y_tr)
            
            # Probabilistic Entry (Only take high-conviction trades)
            probs = model.predict_proba(X_te_s)
            fold_rets = []
            
            for j in range(len(X_te)):
                # Bullish Signal
                if probs[j][2] > 0.60: # 60% confidence requirement
                    sig = 1
                # Bearish Signal
                elif probs[j][0] > 0.60:
                    sig = -1
                else:
                    sig = 0
                    
                actual_ret = df['Close'].iloc[i+j+1] / df['Close'].iloc[i+j] - 1
                fold_rets.append(sig * actual_ret - (abs(sig) * FRICTION))
                
            all_returns.extend(fold_rets)

        return pd.Series(all_returns)

# ==================== EXECUTION ====================
def run_elite_stage4():
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    summary = []

    for f in files:
        ticker = os.path.basename(f).split('_')[0]
        print(f"Auditing: {ticker}")
        
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        returns = EliteAlphaAuditor.simulate(df, ticker)
        
        if len(returns) < 10: continue
        
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        cum_ret = (1 + returns).cumprod()
        mdd = (cum_ret / cum_ret.cummax() - 1).min()
        
        summary.append({
            'Ticker': ticker,
            'Sharpe': round(sharpe, 2),
            'Max_Drawdown': round(mdd, 4),
            'Win_Rate': round((returns > 0).mean(), 2)
        })

    final_df = pd.DataFrame(summary).sort_values('Sharpe', ascending=False)
    print("\nAUDIT COMPLETE")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    run_elite_stage4()
