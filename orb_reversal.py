"""
Opening Range Breakout - Liquidity Grab Reversal Strategy
============================================================
Runs against Alpaca's PAPER trading API.

STRATEGY (per ticker, per day):
1. Capture the 9:30-9:45 ET candle -> box_high / box_low. Box is valid
   until 11:15 ET (90 minutes after the opening candle closes).
2. Liquidity filter: (box_high - box_low) must be >= LIQUIDITY_PCT of the
   14-period daily ATR. If not, skip this ticker for the day.
3. Bias:
     - Opening candle BULLISH -> look for a reversal DOWN. Watch 1-min
       candles for an inverse hammer OR bearish engulfing candle that
       trades ABOVE box_high. That's the "trigger candle" -> go SHORT.
         entry = open of the next 1-min candle
         stop  = just above trigger candle's high
         target = box_low
     - Opening candle BEARISH -> look for a reversal UP. Watch 1-min
       candles for a hammer OR bullish engulfing candle that trades
       BELOW box_low. That's the trigger candle -> go LONG.
         entry = open of the next 1-min candle
         stop  = just below trigger candle's low
         target = box_high
4. Only one trade per ticker per day. Pattern search stops at 11:15 ET.

SETUP
-----
1. pip install alpaca-py pytz --break-system-packages
2. Get PAPER keys from https://app.alpaca.markets (Paper Trading tab)
3. Set environment variables (don't hardcode keys):
     export ALPACA_API_KEY="your_key"
     export ALPACA_SECRET_KEY="your_secret"
4. Edit TICKERS and POSITION_SIZE_USD below.
5. Run it (see "SCHEDULING" at the bottom of this file for automating
   this to fire every trading day).
"""

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ----------------------------------------------------------------------
# CONFIG - edit these
# ----------------------------------------------------------------------
TICKERS = ["TSLA"]              # tell me which tickers and I'll update this
POSITION_SIZE_USD = 1000        # fixed $ amount per trade (assumption - change freely)
LIQUIDITY_PCT = 0.25            # opening candle range must be >= 25% of ATR(14)
ATR_PERIOD = 14
BOX_VALID_MINUTES = 90          # box stays valid this long after it closes
POLL_SECONDS = 15               # how often to check for a newly closed 1-min bar
LOG_FILE = Path(__file__).parent / "orb_trade_log.csv"   # spreadsheet - open in Excel/Sheets

ET = ZoneInfo("America/New_York")

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# ----------------------------------------------------------------------
# Candle helpers
# ----------------------------------------------------------------------
@dataclass
class Candle:
    t: datetime
    o: float
    h: float
    l: float
    c: float


def is_bullish(c: Candle) -> bool:
    return c.c > c.o


def is_bearish(c: Candle) -> bool:
    return c.c < c.o


def body(c: Candle) -> float:
    return abs(c.c - c.o)


def upper_wick(c: Candle) -> float:
    return c.h - max(c.o, c.c)


def lower_wick(c: Candle) -> float:
    return min(c.o, c.c) - c.l


def is_hammer(c: Candle) -> bool:
    # small body near the top, long lower wick, little/no upper wick
    rng = c.h - c.l
    if rng <= 0:
        return False
    return lower_wick(c) >= 2 * body(c) and upper_wick(c) <= body(c) * 0.5


def is_inverse_hammer(c: Candle) -> bool:
    # small body near the bottom, long upper wick, little/no lower wick
    rng = c.h - c.l
    if rng <= 0:
        return False
    return upper_wick(c) >= 2 * body(c) and lower_wick(c) <= body(c) * 0.5


def is_bullish_engulfing(prev: Candle, cur: Candle) -> bool:
    return (
        is_bearish(prev)
        and is_bullish(cur)
        and cur.o <= prev.c
        and cur.c >= prev.o
    )


def is_bearish_engulfing(prev: Candle, cur: Candle) -> bool:
    return (
        is_bullish(prev)
        and is_bearish(cur)
        and cur.o >= prev.c
        and cur.c <= prev.o
    )


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------
def get_daily_atr(symbol: str, period: int = ATR_PERIOD) -> float:
    end = datetime.now(ET)
    start = end - timedelta(days=period * 3)  # buffer for weekends/holidays
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = data_client.get_stock_bars(req).df
    bars = bars.tail(period + 1)
    trs = []
    prev_close = None
    for _, row in bars.iterrows():
        h, l, c = row["high"], row["low"], row["close"]
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    trs = trs[-period:]
    return sum(trs) / len(trs) if trs else 0.0


def get_opening_15m_candle(symbol: str, session_date: datetime) -> Candle:
    start = datetime.combine(session_date.date(), datetime.min.time(), tzinfo=ET).replace(hour=9, minute=30)
    end = start + timedelta(minutes=15)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start,
        end=end + timedelta(minutes=1),
    )
    bars = data_client.get_stock_bars(req).df
    row = bars.iloc[0]
    return Candle(t=start, o=row["open"], h=row["high"], l=row["low"], c=row["close"])


def get_latest_1m_candle(symbol: str, after: datetime) -> Candle | None:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=after,
        end=datetime.now(ET),
    )
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        return None
    if bars.index.nlevels > 1:
        bars = bars.xs(symbol, level="symbol")
    row = bars.iloc[-1]
    idx_time = bars.index[-1]
    return Candle(t=idx_time.to_pydatetime().astimezone(ET), o=row["open"], h=row["high"], l=row["low"], c=row["close"])


# ----------------------------------------------------------------------
# Spreadsheet logging (CSV - opens directly in Excel / Google Sheets)
# ----------------------------------------------------------------------
LOG_HEADERS = [
    "date", "time_logged", "symbol", "event", "box_low", "box_high",
    "bias", "side", "entry_price", "stop_price", "target_price", "qty",
    "order_id", "notes",
]


def log_row(**fields):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if is_new:
            writer.writeheader()
        row = {h: fields.get(h, "") for h in LOG_HEADERS}
        row["date"] = fields.get("date", datetime.now(ET).strftime("%Y-%m-%d"))
        row["time_logged"] = datetime.now(ET).strftime("%H:%M:%S")
        writer.writerow(row)


# ----------------------------------------------------------------------
# Order placement
# ----------------------------------------------------------------------
def place_bracket(symbol: str, side: OrderSide, entry_ref_price: float, stop_price: float, target_price: float):
    qty = max(1, int(POSITION_SIZE_USD // entry_ref_price))
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
    )
    result = trading_client.submit_order(order_data=order)
    print(f"[{symbol}] Submitted {side} qty={qty} stop={stop_price:.2f} target={target_price:.2f} -> order id {result.id}")
    return result


# ----------------------------------------------------------------------
# Per-ticker state machine for one trading day
# ----------------------------------------------------------------------
class TickerSession:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.box_high = None
        self.box_low = None
        self.bias = None          # "bullish" or "bearish" opening candle
        self.box_expiry = None
        self.traded_today = False
        self.setup_done = False

    def setup(self, session_date: datetime):
        atr = get_daily_atr(self.symbol)
        opening = get_opening_15m_candle(self.symbol, session_date)
        candle_range = opening.h - opening.l

        if atr <= 0 or candle_range < LIQUIDITY_PCT * atr:
            print(f"[{self.symbol}] Opening candle range {candle_range:.2f} < {LIQUIDITY_PCT*100:.0f}% of ATR {atr:.2f}. Skipping today.")
            log_row(symbol=self.symbol, event="SKIPPED_LIQUIDITY",
                    notes=f"range={candle_range:.2f} atr={atr:.2f} needed={LIQUIDITY_PCT*atr:.2f}")
            self.traded_today = True  # treat as "done" so we don't watch it
            self.setup_done = True
            return

        self.box_high = opening.h
        self.box_low = opening.l
        self.bias = "bullish" if is_bullish(opening) else "bearish"
        self.box_expiry = opening.t + timedelta(minutes=15 + BOX_VALID_MINUTES)
        self.setup_done = True
        print(f"[{self.symbol}] Box: {self.box_low:.2f}-{self.box_high:.2f} bias={self.bias} valid until {self.box_expiry.strftime('%H:%M')}")
        log_row(symbol=self.symbol, event="BOX_SET", box_low=self.box_low,
                box_high=self.box_high, bias=self.bias)

    def check_for_trigger_and_trade(self):
        if self.traded_today or not self.setup_done:
            return
        if datetime.now(ET) > self.box_expiry:
            print(f"[{self.symbol}] Box window expired with no trade.")
            log_row(symbol=self.symbol, event="NO_TRADE_EXPIRED", box_low=self.box_low,
                    box_high=self.box_high, bias=self.bias)
            self.traded_today = True
            return

        candle = get_latest_1m_candle(self.symbol, self.box_expiry - timedelta(minutes=BOX_VALID_MINUTES))
        if candle is None:
            return

        if self.bias == "bullish":
            # looking for reversal DOWN: inverse hammer or bearish engulfing above box_high
            if candle.h > self.box_high and (is_inverse_hammer(candle) or self._engulf_bear(candle)):
                self._enter(side=OrderSide.SELL, trigger=candle)
        else:
            # looking for reversal UP: hammer or bullish engulfing below box_low
            if candle.l < self.box_low and (is_hammer(candle) or self._engulf_bull(candle)):
                self._enter(side=OrderSide.BUY, trigger=candle)

    def _engulf_bear(self, candle):
        prev = get_latest_1m_candle(self.symbol, candle.t - timedelta(minutes=2))
        return prev is not None and is_bearish_engulfing(prev, candle)

    def _engulf_bull(self, candle):
        prev = get_latest_1m_candle(self.symbol, candle.t - timedelta(minutes=2))
        return prev is not None and is_bullish_engulfing(prev, candle)

    def _enter(self, side: OrderSide, trigger: Candle):
        # Entry = open of the NEXT 1-min candle. We wait for it, then submit
        # a market order right away (approximates the open print).
        print(f"[{self.symbol}] Trigger candle found at {trigger.t}. Waiting for next candle open...")
        time.sleep(60)
        next_candle = get_latest_1m_candle(self.symbol, trigger.t)
        if next_candle is None:
            self.traded_today = True
            return
        entry_price = next_candle.o
        if side == OrderSide.SELL:
            stop = trigger.h + 0.01
            target = self.box_low
        else:
            stop = trigger.l - 0.01
            target = self.box_high
        result = place_bracket(self.symbol, side, entry_price, stop, target)
        qty = max(1, int(POSITION_SIZE_USD // entry_price))
        log_row(symbol=self.symbol, event="TRADE_ENTERED", box_low=self.box_low,
                box_high=self.box_high, bias=self.bias, side=side.value,
                entry_price=entry_price, stop_price=stop, target_price=target,
                qty=qty, order_id=str(result.id))
        self.traded_today = True


# ----------------------------------------------------------------------
# Main loop - run this once per trading day, starting ~9:25 ET.
# It will do nothing until 9:45 (when the opening 15m candle closes),
# then poll for the pattern until 11:15 ET, then exit.
# ----------------------------------------------------------------------
def main():
    if not API_KEY or not SECRET_KEY:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables first.")

    today = datetime.now(ET)
    sessions = {sym: TickerSession(sym) for sym in TICKERS}

    opening_close = today.replace(hour=9, minute=45, second=0, microsecond=0)
    hard_stop = today.replace(hour=11, minute=15, second=0, microsecond=0)

    # Wait for the opening candle to close
    while datetime.now(ET) < opening_close:
        time.sleep(POLL_SECONDS)

    for s in sessions.values():
        s.setup(today)

    # Poll for pattern until the window closes or all tickers are done
    while datetime.now(ET) < hard_stop and not all(s.traded_today for s in sessions.values()):
        for s in sessions.values():
            s.check_for_trigger_and_trade()
        time.sleep(POLL_SECONDS)

    print("Session complete.")


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# SCHEDULING (run this every weekday at 9:25 ET)
# ----------------------------------------------------------------------
# cron (Linux/Mac), in your local time zone equivalent of 9:25 ET:
#   25 9 * * 1-5 cd /path/to/script && /usr/bin/python3 orb_reversal.py >> orb.log 2>&1
#
# The script itself blocks until ~11:15 ET each run, so cron just needs to
# kick it off once per morning - it doesn't need to loop itself.
