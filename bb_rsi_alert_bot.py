"""
bb_rsi_alert_bot.py

Bollinger Band + RSI alert bot for BTCUSDT and SOLUSDT on the 5-minute timeframe.
Replaces the earlier EMA-touch alert bot entirely.

ENTRY LOGIC (per closed 5m candle):
    BUY  alert : (candle.low  <= BB_lower OR candle.close <= BB_lower)  AND RSI <= 30
    SELL alert : (candle.high >= BB_upper OR candle.close >= BB_upper)  AND RSI >= 70

RE-ALERT / DE-DUPE LOGIC (per symbol, per side):
    1. First time both conditions are true together -> fire a "fresh" alert.
       State moves to COOLDOWN.
    2. While in COOLDOWN, wait for RSI to "reset" back through the neutral
       side of the threshold (RSI > 30 for buy, RSI < 70 for sell).
       Once that happens, state moves to REARMED.
    3. While REARMED, if RSI touches the extreme again (<=30 / >=70):
         - If this happens within 30 minutes of the last alert time,
           fire a "re-alert" (RSI condition alone is enough, no BB touch
           required) and go back to COOLDOWN.
         - If more than 30 minutes have passed since the last alert,
           the BB touch condition is required again (same as a fresh
           entry). If BB touch is also true, fire a "fresh" alert and go
           back to COOLDOWN. If not, drop back to WATCHING and wait for
           a normal fresh setup.

State is persisted to state.json between runs (the bot runs fresh every
minute via an external cron trigger -> GitHub Actions workflow dispatch).
"""

import json
import os
import smtplib
import ssl
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "SOLUSDT"]
INTERVAL = "5m"
KLINES_LIMIT = 100  # enough history for BB(20) and RSI(14) to be stable

BB_LENGTH = 20
BB_MULT = 2.0

RSI_LENGTH = 14

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

REALERT_WINDOW_SECONDS = 30 * 60  # 30 minutes

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STATE_FILE = Path(__file__).resolve().parent / "state.json"

# Email config - pulled from environment variables (GitHub Actions secrets).
# Adjust these names if your existing repo secrets use different keys.
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", GMAIL_ADDRESS)

STATE_WATCHING = "watching"
STATE_COOLDOWN = "cooldown"
STATE_REARMED = "rearmed"

DEFAULT_SIDE_STATE = {"state": STATE_WATCHING, "last_alert_time": None}


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_klines(symbol: str, interval: str = INTERVAL, limit: int = KLINES_LIMIT):
    """Fetch closed candles from Binance spot data API."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    candles = []
    for row in raw:
        candles.append(
            {
                "open_time": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "close_time": row[6],
            }
        )

    # Binance includes the currently-forming candle as the last entry.
    # Drop it so we only ever evaluate fully closed candles.
    now_ms = int(time.time() * 1000)
    if candles and candles[-1]["close_time"] > now_ms:
        candles = candles[:-1]

    return candles


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_bollinger_bands(closes, length=BB_LENGTH, mult=BB_MULT):
    """Simple moving average + stddev bands, matching TradingView BB (SMA basis)."""
    bands = []
    for i in range(len(closes)):
        if i + 1 < length:
            bands.append((None, None, None))
            continue
        window = closes[i + 1 - length : i + 1]
        basis = sum(window) / length
        variance = sum((x - basis) ** 2 for x in window) / length
        stddev = variance ** 0.5
        upper = basis + mult * stddev
        lower = basis - mult * stddev
        bands.append((lower, basis, upper))
    return bands


def compute_rsi(closes, length=RSI_LENGTH):
    """Wilder's RSI (matches TradingView's default RSI calculation)."""
    rsi_values = [None] * len(closes)
    if len(closes) <= length:
        return rsi_values

    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    def rsi_from_avgs(avg_gain, avg_loss):
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi_values[length] = rsi_from_avgs(avg_gain, avg_loss)

    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rsi_values[i + 1] = rsi_from_avgs(avg_gain, avg_loss)

    return rsi_values


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_side_state(state, symbol, side):
    state.setdefault(symbol, {})
    state[symbol].setdefault(side, dict(DEFAULT_SIDE_STATE))
    return state[symbol][side]


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(subject: str, body: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("WARNING: GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set, skipping email send.")
        print("---- EMAIL WOULD HAVE BEEN SENT ----")
        print("Subject:", subject)
        print(body)
        print("-------------------------------------")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


def build_email_body(symbol, side, candle, rsi_value, bb_lower, bb_basis, bb_upper, alert_kind):
    close_time = datetime.fromtimestamp(candle["close_time"] / 1000, tz=timezone.utc)
    direction = "BUY" if side == "buy" else "SELL"
    band_label = "Lower Band" if side == "buy" else "Upper Band"
    band_value = bb_lower if side == "buy" else bb_upper

    lines = [
        f"Direction: {direction}",
        f"Symbol: {symbol}",
        f"Timeframe: {INTERVAL}",
        f"Candle Close Time (UTC): {close_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Close Price: {candle['close']:.2f}",
        f"High: {candle['high']:.2f}   Low: {candle['low']:.2f}",
        f"RSI (14): {rsi_value:.2f}",
        f"BB {band_label}: {band_value:.2f}",
        f"BB Basis (SMA20): {bb_basis:.2f}",
        f"Alert Type: {alert_kind}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def evaluate_symbol(symbol, state):
    candles = fetch_klines(symbol)
    if len(candles) < max(BB_LENGTH, RSI_LENGTH) + 2:
        print(f"[{symbol}] Not enough candles yet, skipping.")
        return

    closes = [c["close"] for c in candles]
    bands = compute_bollinger_bands(closes)
    rsi_values = compute_rsi(closes)

    idx = len(candles) - 1  # latest fully closed candle
    candle = candles[idx]
    bb_lower, bb_basis, bb_upper = bands[idx]
    rsi_value = rsi_values[idx]

    if bb_lower is None or rsi_value is None:
        print(f"[{symbol}] Indicators not ready yet, skipping.")
        return

    now = time.time()

    buy_bb_touch = candle["low"] <= bb_lower or candle["close"] <= bb_lower
    buy_rsi_extreme = rsi_value <= RSI_OVERSOLD
    buy_rsi_reset = rsi_value > RSI_OVERSOLD

    sell_bb_touch = candle["high"] >= bb_upper or candle["close"] >= bb_upper
    sell_rsi_extreme = rsi_value >= RSI_OVERBOUGHT
    sell_rsi_reset = rsi_value < RSI_OVERBOUGHT

    _process_side(
        symbol=symbol,
        side="buy",
        state=state,
        candle=candle,
        rsi_value=rsi_value,
        bb_lower=bb_lower,
        bb_basis=bb_basis,
        bb_upper=bb_upper,
        bb_touch=buy_bb_touch,
        rsi_extreme=buy_rsi_extreme,
        rsi_reset=buy_rsi_reset,
        now=now,
    )

    _process_side(
        symbol=symbol,
        side="sell",
        state=state,
        candle=candle,
        rsi_value=rsi_value,
        bb_lower=bb_lower,
        bb_basis=bb_basis,
        bb_upper=bb_upper,
        bb_touch=sell_bb_touch,
        rsi_extreme=sell_rsi_extreme,
        rsi_reset=sell_rsi_reset,
        now=now,
    )


def _process_side(symbol, side, state, candle, rsi_value, bb_lower, bb_basis, bb_upper,
                   bb_touch, rsi_extreme, rsi_reset, now):
    side_state = get_side_state(state, symbol, side)
    current = side_state["state"]
    last_alert_time = side_state["last_alert_time"]

    def fire(alert_kind):
        subject = "Trade Alert"
        body = build_email_body(symbol, side, candle, rsi_value, bb_lower, bb_basis, bb_upper, alert_kind)
        send_email(subject, body)
        side_state["last_alert_time"] = now
        side_state["state"] = STATE_COOLDOWN
        print(f"[{symbol}] {side.upper()} ALERT FIRED ({alert_kind}) RSI={rsi_value:.2f}")

    if current == STATE_WATCHING:
        if bb_touch and rsi_extreme:
            fire("fresh")

    elif current == STATE_COOLDOWN:
        if rsi_reset:
            side_state["state"] = STATE_REARMED

    elif current == STATE_REARMED:
        if rsi_extreme:
            elapsed = now - last_alert_time if last_alert_time is not None else None
            if elapsed is not None and elapsed <= REALERT_WINDOW_SECONDS:
                fire("re-alert (RSI only, within 30 min)")
            elif bb_touch:
                fire("fresh (BB + RSI, after 30 min cooldown)")
            else:
                # RSI extreme again but no BB touch and >30 min since last
                # alert -> full condition not met, fall back to watching.
                side_state["state"] = STATE_WATCHING

    else:
        # Unknown state, reset defensively.
        side_state["state"] = STATE_WATCHING


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    state = load_state()

    for symbol in SYMBOLS:
        try:
            evaluate_symbol(symbol, state)
        except Exception as exc:  # noqa: BLE001
            print(f"[{symbol}] ERROR: {exc}", file=sys.stderr)

    save_state(state)


if __name__ == "__main__":
    main()
