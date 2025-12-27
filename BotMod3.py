import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ==================== CONFIGURATION ====================
INPUT_DIR = "bot_ready_data"
OUTPUT_DIR = "stage4_elite_audit"
WALK_TRAIN = 252 * 2 
WALK_TEST = 21       
TARGET_VOL = 0.025   
FRICTION = 0.0004    

os.makedirs(OUTPUT_DIR, exist_ok=True)

class EliteAlphaAuditor:
    @staticmethod
    def engineer_elite_features(df):
        df = df.copy()
        df['Vol_ZScore'] = (df['Close'].pct_change().rolling(20).std() / 
                            df['Close'].pct_change().rolling(200).std())
        df['Dist_from_High'] = (df['Close'] / df['Close'].rolling(252).max()) - 1
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def get_optimized_params(ticker, vol_regime):
        if ticker in ['NVDA', 'AMZN', 'META']:
            return {'depth': 3, 'lr': 0.03, 'n': 80}
        if vol_regime < 1.0:
            return {'depth': 5, 'lr': 0.05, 'n': 100}
        return {'depth': 4, 'lr': 0.04, 'n': 90}

    @staticmethod
    def simulate(df, ticker):
        df = EliteAlphaAuditor.engineer_elite_features(df)
        drop_cols = ['Open','High','Low','Close','Volume','Target','Fwd_Ret','Ticker','News_Summary']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        
        # Benchmark for Alpha calculation
        market_returns = df['Close'].pct_change()
        
        fwd_ret = df['Close'].shift(-5) / df['Close'] - 1
        y = np.where(fwd_ret > TARGET_VOL, 1, np.where(fwd_ret < -TARGET_VOL, -1, 0))
        
        all_returns = []
        
        for i in range(WALK_TRAIN, len(X) - WALK_TEST, WALK_TEST):
            X_tr, y_tr = X.iloc[i-WALK_TRAIN:i], y[i-WALK_TRAIN:i]
            X_te = X.iloc[i:i+WALK_TEST]
            
            scaler = RobustScaler()
            imputer = SimpleImputer(strategy='median')
            
            try:
                X_tr_s = scaler.fit_transform(imputer.fit_transform(X_tr))
                X_te_s = scaler.transform(imputer.transform(X_te))
            except: continue

            vol_now = df['Vol_ZScore'].iloc[i]
            p = EliteAlphaAuditor.get_optimized_params(ticker, vol_now)
            
            model = GradientBoostingClassifier(
                n_estimators=p['n'], 
                learning_rate=p['lr'], 
                max_depth=p['depth'],
                subsample=0.8
            )
            
            model.fit(X_tr_s, y_tr)
            probs = model.predict_proba(X_te_s)
            
            for j in range(len(X_te)):
                sig = 1 if probs[j][2] > 0.60 else (-1 if probs[j][0] > 0.60 else 0)
                actual_ret = df['Close'].iloc[i+j+1] / df['Close'].iloc[i+j] - 1
                all_returns.append(sig * actual_ret - (abs(sig) * FRICTION))
                
        return pd.Series(all_returns), market_returns.loc[X.index[WALK_TRAIN:WALK_TRAIN+len(all_returns)]]

def run_elite_stage3():
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    summary = []

    for f in files:
        ticker = os.path.basename(f).split('_')[0]
        print(f"💎 Elite Auditing: {ticker}")
        
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        returns, market_rets = EliteAlphaAuditor.simulate(df, ticker)
        
        if len(returns) < 10: continue
        
        # GBM Metrics
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        cum_ret = (1 + returns).cumprod()
        mdd = (cum_ret / cum_ret.cummax() - 1).min()
        
        # Linear (Market) Metrics for Alpha Comparison
        market_sharpe = (market_rets.mean() / (market_rets.std() + 1e-9)) * np.sqrt(252)
        alpha_added = sharpe - market_sharpe
        
        summary.append({
            'Ticker': ticker,
            'GBM_Sharpe': round(sharpe, 2),
            'Linear_Sharpe': round(market_sharpe, 2),
            'Alpha_Added': round(alpha_added, 2),
            'GBM_MDD': round(mdd, 4),
            'Win_Rate': round((returns > 0).mean(), 2)
        })

    final_df = pd.DataFrame(summary).sort_values('GBM_Sharpe', ascending=False)
    # SAVE THE SYNCED FILE
    final_df.to_csv(f"{OUTPUT_DIR}/elite_audit_summary.csv", index=False)
    print("\nAUDIT COMPLETE ")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    run_elite_stage3()
