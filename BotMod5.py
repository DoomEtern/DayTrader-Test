# stage6_apex_fx_bot.py

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from functools import wraps
import MetaTrader5 as mt5  # Make sure MT5 is installed and running

# ==================== CONFIGURATION ====================
STAGE5_PATH = "stage5_portfolio/portfolio_weights.csv"
LOG_DIR = "stage6_logs"
PAPER_MODE = True  # Set False for live MT5 account

DRIFT_THRESHOLD = 0.02      # 2% relative drift before trade
MIN_TRADE_DOLLAR = 20.0
REBALANCE_FREQ_SECS = 60*60  # 1 hour
TRAILING_ATR_MULT = 1.5      # ATR multiplier for trailing stop
ATR_PERIOD = 14

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/apex_fx_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def retry_with_backoff(retries=5, backoff_in_seconds=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x >= retries:
                        logging.error(f"FATAL: Max retries for {func.__name__}")
                        raise e
                    sleep = backoff_in_seconds * 2 ** x
                    logging.warning(f"Retry {func.__name__} in {sleep}s due to {e}")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

# ==================== MT5 WRAPPERS ====================
class MT5Broker:
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        self.account_info = mt5.account_info()

    @retry_with_backoff()
    def get_positions(self):
        """Returns a dict {symbol: {volume, price_open, current_price}}"""
        positions = mt5.positions_get()
        data = {}
        if positions is not None:
            for p in positions:
                data[p.symbol] = {
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "current_price": mt5.symbol_info_tick(p.symbol).bid if p.volume > 0 else mt5.symbol_info_tick(p.symbol).ask
                }
        return data

    @retry_with_backoff()
    def send_order(self, symbol, volume, order_type=mt5.ORDER_TYPE_BUY):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": mt5.symbol_info_tick(symbol).ask if order_type==mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
            "deviation": 10,
            "magic": 123456,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Trade failed {symbol}: {result.retcode}")
        else:
            logging.info(f"Executed trade {symbol}: {volume} lots, type {order_type}")

    @retry_with_backoff()
    def modify_stop(self, symbol, sl_price):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        p = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": p.ticket,
            "sl": sl_price,
            "tp": p.tp,
            "deviation": 10,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Modify SL failed {symbol}: {result.retcode}")
        else:
            logging.info(f"Updated trailing stop for {symbol} to {sl_price}")

# ==================== APEX FX MANAGER ====================
class ApexFXManager:
    def __init__(self):
        self.broker = MT5Broker()
        self.weights = pd.read_csv(STAGE5_PATH).set_index('Ticker')

    def compute_atr(self, symbol, period=ATR_PERIOD):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period+1)
        if rates is None or len(rates) < period:
            return None
        highs = np.array([r.high for r in rates])
        lows = np.array([r.low for r in rates])
        closes = np.array([r.close for r in rates])
        trs = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
        atr = np.mean(trs)
        return atr

    def adjust_trailing_stops(self):
        positions = self.broker.get_positions()
        for symbol, pos in positions.items():
            atr = self.compute_atr(symbol)
            if atr is None:
                continue
            if pos['volume'] > 0:
                new_sl = max(pos['price_open'], pos['current_price'] - TRAILING_ATR_MULT*atr)
            else:
                new_sl = min(pos['price_open'], pos['current_price'] + TRAILING_ATR_MULT*atr)
            self.broker.modify_stop(symbol, new_sl)

    def rebalance(self):
        equity = self.broker.account_info.balance
        positions = self.broker.get_positions()
        total_equity = equity

        # 1. Handle sells
        for symbol, pos in positions.items():
            target_pct = self.weights.loc[symbol, 'Weight'] if symbol in self.weights.index else 0
            target_val = target_pct * total_equity
            current_val = pos['volume'] * pos['current_price']
            drift = (current_val - target_val)/(target_val if target_val>0 else 1)
            if drift > DRIFT_THRESHOLD or target_pct==0:
                volume_to_sell = abs(current_val - target_val)/pos['current_price']
                if volume_to_sell*100000 > MIN_TRADE_DOLLAR:
                    self.broker.send_order(symbol, volume_to_sell, order_type=mt5.ORDER_TYPE_SELL)

        # 2. Handle buys
        for symbol, row in self.weights.iterrows():
            if symbol == "CASH_USD": continue
            target_val = row['Weight']*total_equity
            current_val = positions.get(symbol, {'volume':0})['volume']*positions.get(symbol, {'current_price':0})['current_price']
            drift = (target_val - current_val)/(target_val if target_val>0 else 1)
            if drift > DRIFT_THRESHOLD and target_val>current_val:
                volume_to_buy = (target_val - current_val)/mt5.symbol_info_tick(symbol).ask
                if volume_to_buy*100000 > MIN_TRADE_DOLLAR:
                    self.broker.send_order(symbol, volume_to_buy, order_type=mt5.ORDER_TYPE_BUY)

    def run_forever(self):
        print(f"🛡️ APEX FX GUARDIAN ACTIVE | Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
        while True:
            try:
                self.adjust_trailing_stops()
                self.rebalance()
                time.sleep(REBALANCE_FREQ_SECS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"LOOP ERROR: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = ApexFXManager()
    bot.run_forever()
