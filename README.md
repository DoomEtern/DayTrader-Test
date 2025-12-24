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

### Stage 3: Signal Generation & Backtesting (`BotMod2.py`)
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

 # 📈 Stage 4 — Machine Learning Alpha Research & Validation

## Overview

**Stage 4** is the **machine-learning research and validation layer** of the trading system.

At this stage, raw market features produced in **Stage 2** are converted into **predictive signals** using **supervised machine learning**, and rigorously evaluated using **walk-forward testing** to ensure realism and prevent data leakage.

This stage answers one core question:

> **Does non-linear machine learning generate alpha beyond simple linear models?**

---

## Objectives

- Train **leak-safe ML models** on historical market data
- Compare **non-linear ML (Gradient Boosting)** against a **linear benchmark**
- Evaluate performance using **quant-grade metrics**
- Penalize trading using **realistic friction**
- Produce research artifacts (equity curves, metrics, alpha attribution)






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
