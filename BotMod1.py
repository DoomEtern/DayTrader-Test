import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==================== FOREX SETUP ====================
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sentiment_analyzer = SentimentIntensityAnalyzer()

# Major FX Pairs (Yahoo Finance suffix is '=X')
FOREX_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'EURJPY=X']
START_DATE = "2020-01-01"
OUTPUT_DIR = "forex_ready_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== FX DATA DOWNLOAD ====================
def download_forex_data(pairs):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading Forex Pairs...")
    try:
        data = yf.download(
            pairs,
            start=START_DATE,
            interval="1d",
            group_by="ticker",
            auto_adjust=True, # Critical for FX to handle roll-over adjustments
            threads=True, 
            progress=False
        )
        return data
    except Exception as e:
        print(f"Forex Download Error: {e}")
        return None

# ==================== FX SENTIMENT ====================
def get_forex_sentiment(pair):
    """
    Forex sentiment is macro-driven. 
    Searches for the base and quote currencies.
    """
    try:
        base_currency = pair[:3]
        t = yf.Ticker(pair)
        news_list = t.news[:10]
        
        if not news_list:
            return 0.0, "Neutral Macro"

        headlines = [n.get('title', '') for n in news_list]
        text_block = " ".join(headlines)
        
        score = sentiment_analyzer.polarity_scores(text_block)["compound"]
        return score, text_block[:500]
    except Exception as e:
        return 0.0, f"Sentiment Error: {e}"

# ==================== FX INDICATORS (PIP-AWARE) ====================
def compute_fx_indicators(df, ticker):
    df = df.copy()
    
    # 1. Pip Calculation Logic
    # JPY pairs use 0.01 per pip; most others use 0.0001
    is_jpy = "JPY" in ticker
    pip_unit = 0.01 if is_jpy else 0.0001
    
    # 2. Basic Returns & Pip Volatility
    df['Return'] = df['Close'].pct_change()
    df['Pip_Change'] = (df['Close'].diff() / pip_unit)
    
    # 3. Moving Averages (Institutional Levels)
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # 4. ATR (Average True Range) - THE most important FX metric
    high_low = df['High'] - df['Low']
    high_pc = np.abs(df['High'] - df['Close'].shift())
    low_pc = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['ATR_Pips'] = tr.rolling(14).mean() / pip_unit

    # 5. RSI (Wilder's Smoothing for FX Reversals)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 6. Trend Identification Logic
    # Forex is mean-reverting; we look for "Stretches" away from the MA
    df['Distance_MA200_Pips'] = (df['Close'] - df['MA200']) / pip_unit
    
    return df

# ==================== FX PROCESSOR ====================
def process_fx_pair(raw_data, ticker):
    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            pair_data = raw_data[ticker].copy()
        else:
            pair_data = raw_data.copy()

        # Forex OHLC Check
        pair_data = compute_fx_indicators(pair_data, ticker)
        
        sentiment_score, news_summary = get_forex_sentiment(ticker)
        pair_data['Latest_Macro_Sentiment'] = sentiment_score
        pair_data['Macro_Summary'] = news_summary

        pair_data.dropna(inplace=True)
        pair_data.reset_index(inplace=True)
        return pair_data

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

# ==================== PIPELINE ====================
def run_fx_pipeline():
    raw = download_forex_data(FOREX_PAIRS)
    if raw is None or raw.empty: return

    for ticker in FOREX_PAIRS:
        df = process_fx_pair(raw, ticker)
        if df is not None:
            clean_name = ticker.replace('=X', '')
            df.to_csv(f"{OUTPUT_DIR}/{clean_name}_data.csv", index=False)
            print(f"Done: {clean_name} | RSI: {df['RSI'].iloc[-1]:.2f} | ATR (Pips): {df['ATR_Pips'].iloc[-1]:.1f}")

if __name__ == "__main__":
    run_fx_pipeline()
