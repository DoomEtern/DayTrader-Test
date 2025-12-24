import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==================== SETUP ====================
# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sentiment_analyzer = SentimentIntensityAnalyzer()

TICKERS = ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','JPM','V','MA','UNH']
START_DATE = "2020-01-01"
# Get today's date for filename/logging
TODAY = datetime.now().strftime('%Y-%m-%d')

OUTPUT_DIR = "bot_ready_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== DATA DOWNLOAD ====================
def download_market_data(tickers):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading market data...")
    # 'group_by' is essential for multi-ticker downloads
    try:
        data = yf.download(
            tickers,
            start=START_DATE,
            group_by="ticker",
            auto_adjust=False, # Use False if you want unadjusted Close for true OHLC
            threads=True, 
            progress=False
        )
        return data
    except Exception as e:
        print(f"Critical Download Error: {e}")
        return None

# ==================== NEWS SENTIMENT (FIXED) ====================
def get_yfinance_sentiment(ticker):
    """
    Uses yfinance API instead of raw scraping to avoid blocking/ban.
    """
    try:
        # yf.Ticker object fetches official news stream
        t = yf.Ticker(ticker)
        news_list = t.news
        
        if not news_list:
            return 0.0, "No news found"

        # Combine titles of the latest 5-10 articles
        headlines = [n.get('title', '') for n in news_list][:10]
        text_block = " ".join(headlines)
        
        if not text_block:
            return 0.0, "Empty headlines"

        score = sentiment_analyzer.polarity_scores(text_block)["compound"]
        return score, text_block[:500] # Return snippet for verification

    except Exception as e:
        # Fallback if API changes
        print(f"Sentiment Error for {ticker}: {e}")
        return 0.0, "Error"

# ==================== INDICATORS ====================
def compute_indicators(df):
    df = df.copy()
    
    # Basic Return
    df['Return'] = df['Close'].pct_change()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # Volatility (Annualized)
    df['Volatility'] = df['Return'].rolling(20).std() * np.sqrt(252)

    # RSI (Corrected to use EWM / Wilder's Smoothing)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Use exponential moving average for RSI standard
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Liquidity proxy
    df['Liquidity'] = df['Volume'] * df['Close']

    # Trend Logic
    df['Trend_Break'] = ((df['Close'] < df['MA50']) &
                          (df['Close'].shift(1) > df['MA50'].shift(1))).astype(int)

    # Reversal Logic
    df['Reversal'] = ((df['RSI'] < 30) & (df['Return'] > 0)).astype(int)

    return df

# ==================== PROCESS ONE STOCK ====================
def process_stock(raw_data, ticker):
    # Handle yfinance MultiIndex structure safely
    try:
        # Check if the dataframe is MultiIndex (multiple tickers) or Single Index
        if isinstance(raw_data.columns, pd.MultiIndex):
            stock = raw_data[ticker].copy()
        else:
            # If only 1 ticker was downloaded, yfinance doesn't use MultiIndex
            stock = raw_data.copy()
            
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in stock.columns for col in required_cols):
            print(f"Skipping {ticker}: Missing columns.")
            return None

        # 1. Compute Technicals
        stock = compute_indicators(stock)

        # 2. Get Sentiment (CURRENT SNAPSHOT ONLY)
        # Note: Renamed column to 'Latest_Sentiment' to avoid confusion 
        # that this sentiment applies to historical rows.
        sentiment_score, news_text = get_yfinance_sentiment(ticker)
        
        # We fill the WHOLE column for plotting convenience, 
        # but in ML, use this feature with caution (Lookahead bias warning)
        stock['Latest_News_Sentiment'] = sentiment_score
        stock['News_Summary'] = news_text

        # Clean up NaNs created by rolling windows
        stock.dropna(inplace=True)
        stock.reset_index(inplace=True)

        return stock

    except KeyError:
        print(f"Ticker {ticker} not found in downloaded data.")
        return None

# ==================== PLOTTING ====================
def plot_stock(df, ticker):
    # Check if df is empty before plotting
    if df.empty:
        return

    plt.figure(figsize=(14,8))
    
    plt.plot(df['Date'], df['Close'], label="Price", linewidth=2, alpha=0.8)
    plt.plot(df['Date'], df['MA50'], label="MA50", linestyle='--')
    plt.plot(df['Date'], df['MA200'], label="MA200", linestyle='--')

    # Add Reversal markers
    reversals = df[df['Reversal'] == 1]
    if not reversals.empty:
        plt.scatter(reversals['Date'], reversals['Close'],
                    color='green', label='RSI Reversal', s=60, zorder=5)

    # Add Trend Break markers
    breaks = df[df['Trend_Break'] == 1]
    if not breaks.empty:
        plt.scatter(breaks['Date'], breaks['Close'],
                    color='red', label='Trend Break', s=60, zorder=5)

    current_sent = df['Latest_News_Sentiment'].iloc[-1]
    plt.title(f"{ticker} Analysis | Current Sentiment: {current_sent:.4f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{ticker}_analysis.png")
    plt.close() # Close figure to free memory

# ==================== PIPELINE ====================
def run_pipeline():
    raw = download_market_data(TICKERS)

    if raw is None or raw.empty:
        print("Market download failed.")
        return

    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df = process_stock(raw, ticker)

        if df is not None and not df.empty:
            df.to_csv(f"{OUTPUT_DIR}/{ticker}_data.csv", index=False)
            plot_stock(df, ticker)
            print(f"Saved {ticker} -> Sentiment: {df['Latest_News_Sentiment'].iloc[-1]:.2f}")
        else:
            print(f"Skipping {ticker} (Insufficient Data)")

# ==================== LOOP CONTROL ====================
if __name__ == "__main__":
    UPDATE_INTERVAL = 1800 # 30 mins
    
    while True:
        user = input("\n>> Press Enter to run update or type 'exit' to quit: ").lower()
        if user == "exit":
            print("Pipeline stopped.")
            break

        print(f"\n=== Pipeline Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        run_pipeline()
        
        print(f"\nPipeline finished. Sleeping for {UPDATE_INTERVAL/60} minutes...")
        # Note: In a real bot, you usually don't ask for Input AND sleep. 
        # This setup waits for the sleep to finish before letting you type 'exit'.
        # For a simpler manual trigger, remove the sleep below.
        try:
            time.sleep(UPDATE_INTERVAL)
        except KeyboardInterrupt:
            print("\nSleep interrupted by user.")
            continue