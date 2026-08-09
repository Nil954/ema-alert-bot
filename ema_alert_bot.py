"""
EMA Touch Alert Bot — GitHub Actions edition (single-run per invocation)
=========================================================================

Same strategy as before, but rewritten to run ONCE per execution instead of
looping forever — because GitHub Actions spins up a fresh machine on a
schedule, runs this script, then shuts the machine down. The workflow file
(.github/workflows/ema_alert.yml) triggers this every 5 minutes and commits
state.json back to the repo so alert "armed/disarmed" state survives between
runs.

Strategy recap
---------------
BUY watch active when:  EMA200 > EMA400  AND  EMA54 > EMA101
SELL watch active when: EMA200 < EMA400  AND  EMA54 < EMA101

While a watch is active, an alert fires once when a CLOSED 5-min candle's
high/low comes within TOUCH_BUFFER_PCT of the 54 or 101 EMA. It then stays
quiet until price moves RESET_DISTANCE_PCT away and could touch again.

Credentials (GMAIL_ADDRESS, GMAIL_APP_PASSWORD, ALERT_TO_EMAIL) are read from
environment variables, which GitHub Actions injects from your repo's
encrypted Secrets. Local testing fallback values are below (safe to leave
as placeholders — never commit real credentials into this file).
"""

import os
import json
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import timezone
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

PAIRS = ["BTCUSDT", "SOLUSDT"]
INTERVAL = "5m"
KLINE_LIMIT = 1000

EMA_FAST = 54
EMA_SLOW = 101
EMA_TREND_FAST = 200
EMA_TREND_SLOW = 400

RSI_LENGTH = 14

TOUCH_BUFFER_PCT = 0.0          # 0 = exact touch only
RESET_DISTANCE_PCT = 0.003      # 0.3%

STATE_FILE = Path(__file__).parent / "state.json"

# Credentials: pulled from environment variables (set as GitHub Secrets).
# These placeholder fallbacks only matter for local testing.
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "your_email@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "xxxx xxxx xxxx xxxx")
ALERT_TO_EMAIL = os.environ.get("ALERT_TO_EMAIL", "your_email@gmail.com")

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ema_bot")

# ============================================================
# Data fetching
# ============================================================

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    now = pd.Timestamp.now(tz=timezone.utc)
    if df.iloc[-1]["close_time"] > now:
        df = df.iloc[:-1]  # drop still-forming candle

    return df.reset_index(drop=True)


# ============================================================
# Indicators
# ============================================================

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df[f"ema{EMA_FAST}"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df[f"ema{EMA_SLOW}"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df[f"ema{EMA_TREND_FAST}"] = df["close"].ewm(span=EMA_TREND_FAST, adjust=False).mean()
    df[f"ema{EMA_TREND_SLOW}"] = df["close"].ewm(span=EMA_TREND_SLOW, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    df["rsi"] = rsi
    return df


# ============================================================
# State persistence
# ============================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            log.warning("Could not parse state.json, starting fresh.")
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def key_for(symbol: str, ema_level: int, direction: str) -> str:
    return f"{symbol}:{ema_level}:{direction}"


def default_slot() -> dict:
    return {"armed": True, "regime_active": False}


# ============================================================
# Email
# ============================================================

def send_email_alert(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ALERT_TO_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [ALERT_TO_EMAIL], msg.as_string())
        log.info(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Failed to send email: {e}")


def format_alert(symbol, direction, ema_level, ema_value, price, rsi_value, candle_time):
    return (
        f"{symbol} — {direction.upper()} zone touch on EMA{ema_level}\n\n"
        f"Pair:        {symbol}\n"
        f"Direction:   {direction.upper()}\n"
        f"EMA touched: EMA{ema_level} ({ema_value:.4f})\n"
        f"Close price: {price:.4f}\n"
        f"RSI(14):     {rsi_value:.2f}\n"
        f"Candle time (UTC): {candle_time}\n"
    )


# ============================================================
# Core check logic
# ============================================================

def touches(candle_high: float, candle_low: float, ema_value: float, buffer_pct: float) -> bool:
    upper = ema_value * (1 + buffer_pct)
    lower = ema_value * (1 - buffer_pct)
    return candle_low <= upper and candle_high >= lower


def process_symbol(symbol: str, state: dict) -> None:
    df = fetch_klines(symbol, INTERVAL, KLINE_LIMIT)
    if len(df) < EMA_TREND_SLOW + 10:
        log.warning(f"{symbol}: not enough candles yet for stable EMA{EMA_TREND_SLOW} ({len(df)} fetched).")
        return

    df = add_emas(df)
    df = add_rsi(df, RSI_LENGTH)

    last = df.iloc[-1]
    price = last["close"]
    high = last["high"]
    low = last["low"]
    rsi_value = last["rsi"]
    candle_time = last["close_time"]

    ema54 = last[f"ema{EMA_FAST}"]
    ema101 = last[f"ema{EMA_SLOW}"]
    ema200 = last[f"ema{EMA_TREND_FAST}"]
    ema400 = last[f"ema{EMA_TREND_SLOW}"]

    buy_regime = (ema200 > ema400) and (ema54 > ema101)
    sell_regime = (ema200 < ema400) and (ema54 < ema101)

    log.info(
        f"{symbol} close={price:.4f} ema54={ema54:.4f} ema101={ema101:.4f} "
        f"ema200={ema200:.4f} ema400={ema400:.4f} rsi={rsi_value:.2f} "
        f"buy_regime={buy_regime} sell_regime={sell_regime}"
    )

    for ema_level, ema_value in [(EMA_FAST, ema54), (EMA_SLOW, ema101)]:
        for direction, regime_active in [("buy", buy_regime), ("sell", sell_regime)]:
            k = key_for(symbol, ema_level, direction)
            slot = state.get(k, default_slot())

            if not regime_active:
                if slot["regime_active"]:
                    log.info(f"{symbol} {direction} regime on EMA{ema_level} turned OFF. Resetting.")
                state[k] = default_slot()
                continue

            if not slot["regime_active"]:
                log.info(f"{symbol} {direction} regime on EMA{ema_level} turned ON.")
            slot["regime_active"] = True

            is_touch = touches(high, low, ema_value, TOUCH_BUFFER_PCT)
            distance_pct = abs(price - ema_value) / ema_value

            if is_touch and slot["armed"]:
                subject = f"[EMA Alert] {symbol} {direction.upper()} — EMA{ema_level} touch"
                body = format_alert(symbol, direction, ema_level, ema_value, price, rsi_value, candle_time)
                send_email_alert(subject, body)
                slot["armed"] = False
                log.info(f"ALERT FIRED: {k} at price {price:.4f} (EMA {ema_value:.4f})")

            elif not slot["armed"] and distance_pct >= RESET_DISTANCE_PCT:
                slot["armed"] = True
                log.info(f"{k} re-armed (price moved {distance_pct*100:.2f}% away from EMA).")

            state[k] = slot


# ============================================================
# Entry point — runs once, then exits (GitHub Actions triggers each run)
# ============================================================

def main():
    log.info("EMA Alert Bot run starting.")
    state = load_state()

    for symbol in PAIRS:
        try:
            process_symbol(symbol, state)
        except Exception as e:
            log.error(f"Error processing {symbol}: {e}")

    save_state(state)
    log.info("EMA Alert Bot run finished.")


if __name__ == "__main__":
    main()
