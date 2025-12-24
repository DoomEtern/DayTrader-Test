## Stage Breakdown

### Stage 1: Data Collection
- Download historical price data for a list of tickers using `yfinance`.
- Collect daily open, high, low, close, volume data.
- Placeholder for sentiment/news scraping (to be expanded in future).

### Stage 2: Feature Engineering (`BotMod1.py`)
- Compute technical indicators:
  - RSI, MACD, MACD Signal, ATR
  - Moving Averages: MA50, MA200
  - Volatility, Daily Returns
  - Price Highs & Lows
  - Trend-breaking and reversal flags
  - Market liquidity proxies
- Integrate **news sentiment scores** using NLTK's VADER or keyword-based proxies.
- Outputs **per-ticker CSV files** in `bot_ready_data/`.
- Visualizations for each ticker with multi-event markers:
  - Major Corrections
  - Volatility Expansions
  - Momentum Continuations
  - News-driven signals

### Stage 3: Signal Generation & Backtesting (`stage3_signal_backtest.py`)
- Load Stage 2 CSVs.
- Generate deterministic **long/short/neutral signals** based on multi-indicator conditions:
  - RSI oversold/overbought
  - Trend following via MA200
  - Momentum via MACD
  - Volume & liquidity thresholds
  - News sentiment integration
- Simulate portfolio performance:
  - Capital allocation rules
  - Transaction costs
  - Position tracking
  - Strategy returns
- Compute professional-grade performance metrics:
  - Total return, CAGR, volatility, Sharpe, Max Drawdown, Win Rate, Profit Factor, etc.
- Produce **audit-ready outputs**:
  - Signal-enriched CSVs
  - Strategy performance summary
  - Plots with price, buy/sell markers, equity curve
 


Install dependencies
pip install -r requirements.txt


requirements.txt includes:
yfinance
pandas
numpy
matplotlib
nltk
requests
beautifulsoup4
