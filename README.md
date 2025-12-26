🌍 Apex Forex Trading Bot

Automated Forex portfolio engine with machine learning alpha signals, dynamic risk allocation, and MT5 integration for live and paper trading.

Overview

This system is designed to run a full quantitative Forex workflow:

Stage	Function
Stage 3 – Forex Reality	Processes raw Forex OHLC data, computes volatility-adjusted signals, and simulates trades with spreads, swaps, and leverage.
Stage 4 – FX Alpha Audit	Generates risk-adjusted alpha metrics, Sharpe ratios, and max drawdowns, comparing ML strategies to linear baselines.
Stage 5 – FX Portfolio Engine	Builds a dynamically scaled, currency-safe portfolio, applying maximum exposure caps per pair and per currency.
Stage 6 – MT5 Execution Engine	Connects to MetaTrader 5 desktop, executes trades automatically according to the Stage 5 portfolio, supports trailing stops, and risk-aware rebalancing.
System Requirements

Python 3.10+

Libraries: pandas, numpy, MetaTrader5, matplotlib (for Stage 3/visualization)

MetaTrader 5 Desktop (Windows) – Required for live/paper trading

Data Directory: OHLC CSVs for each FX pair

Installation
git clone <repo_url>
cd apex-forex-bot
python -m venv bot_env
source bot_env/bin/activate       # Linux / Mac
# bot_env\Scripts\activate       # Windows
pip install -r requirements.txt


requirements.txt example:

pandas
numpy
matplotlib
MetaTrader5


⚠️ Note: The MetaTrader5 Python package only works with the desktop terminal, not the web version.
