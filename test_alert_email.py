"""
test_alert_email.py

ONE-TIME TEST SCRIPT - not part of the bot's normal operation.

This forces a fake BUY alert email to fire using dummy data, so you can
confirm the Gmail SMTP send actually works end-to-end (auth, app password,
recipient address) before trusting the real bot in production.

It does NOT touch state.json and does NOT fetch real market data.

Run it the same way as the real bot (same env vars required):
    GMAIL_ADDRESS=...
    GMAIL_APP_PASSWORD=...
    RECIPIENT_EMAIL=...

Delete this file from the repo once you've confirmed the test email arrives.
"""

import os
from datetime import datetime, timezone

import bb_rsi_alert_bot as bot


def main():
    fake_candle = {
        "low": 62950.0,
        "high": 63050.0,
        "close": 62980.0,
        "close_time": int(datetime.now(timezone.utc).timestamp() * 1000),
    }

    subject = "Trade Alert"
    body = bot.build_email_body(
        symbol="BTCUSDT",
        side="buy",
        candle=fake_candle,
        rsi_value=27.5,
        bb_lower=63000.0,
        bb_basis=63400.0,
        bb_upper=63800.0,
        alert_kind="TEST EMAIL - not a real signal",
    )

    print("Sending test email...")
    bot.send_email(subject, body)
    print("Done. Check your inbox (and spam folder) for a 'Trade Alert' email.")


if __name__ == "__main__":
    main()
