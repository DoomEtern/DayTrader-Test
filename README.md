# Quantitative Trading and Risk Management Pipeline

This repository contains a multi-stage quantitative system for market data ingestion, alpha generation using machine learning, and institutional-grade risk auditing.

## Pipeline Stages

### Stage 1: Data Ingestion and Sentiment Analysis
Downloads market data for selected tickers and performs sentiment analysis on current news headlines using the VADER lexicon. It calculates technical indicators including RSI, MACD, and moving averages.

### Stage 2: Realistic Simulation
Runs backtests that incorporate real-world friction such as slippage and estimated tax rates. It uses ATR-based trailing stops to simulate professional exit strategies.

### Stage 3: Alpha Audit
Uses a Gradient Boosting Classifier and walk-forward validation to evaluate signal quality. It compares the strategy's Sharpe ratio against the market benchmark to determine alpha.

### Stage 4: Portfolio Optimization
Detects the current market regime (Bull, Bear, or Crisis) and performs risk-parity optimization. It adjusts asset exposure based on the detected regime and minimizes covariance and correlation penalties.

### Stage 5: Extreme Risk Audit
Performs a multivariate block bootstrap to preserve cross-asset correlations and volatility clustering. It calculates 99% Value-at-Risk (VaR) and Expected Shortfall (ES) over a 21-day horizon.

## Usage
The scripts are designed to be executed in sequence:
1. BotMod1.py
2. BotMod2.py
3. BotMod3.py
4. BotMod4.py
5. BotMod5.py

## Disclaimer
This project is for research purposes only. Trading involves significant risk of loss.
