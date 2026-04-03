"""
PolyWeatherBot — Telegram Alerts & Command Polling
Sends trade signals and accepts /status, /pause, /resume, /scan commands.
"""

import logging
import threading
import time

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = settings.TELEGRAM_CHAT_ID

# Bot state — shared across threads
_bot_state = {
    "paused": False,
    "scan_requested": False,
    "last_update_id": 0,
}


def is_configured() -> bool:
    """Check if Telegram is properly configured."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_alert(message: str) -> bool:
    """Send a message via Telegram bot."""
    if not is_configured():
        logger.debug("Telegram not configured, skipping alert")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def send_signal_alert(signal) -> bool:
    """Send a formatted trading signal alert."""
    msg = (
        f"*PolyWeatherBot Signal*\n\n"
        f"City: {signal.city}\n"
        f"{signal.question}\n"
        f"Side: *{signal.side}*\n"
        f"Model prob: {signal.model_prob:.1%}\n"
        f"Market price: {signal.market_price:.1%}\n"
        f"Edge: {signal.edge:.1%}\n"
        f"Kelly bet: ${signal.kelly_bet:.2f}\n"
        f"EV: {signal.expected_value:.1%}"
    )
    return send_alert(msg)


def send_trade_alert(signal, result: dict) -> bool:
    """Send alert when a trade is executed."""
    paper = "PAPER" if result.get("paper") else "LIVE"
    msg = (
        f"{paper} *Trade Executed*\n\n"
        f"{signal.city} - {signal.side}\n"
        f"Price: {signal.market_price:.4f}\n"
        f"Size: ${signal.kelly_bet:.2f}\n"
        f"Edge: {signal.edge:.1%}"
    )
    return send_alert(msg)


def _poll_commands():
    """Poll Telegram for bot commands. Runs in a background thread."""
    if not is_configured():
        return

    logger.info("Telegram command polling started")
    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={
                    "offset": _bot_state["last_update_id"] + 1,
                    "timeout": 30,
                },
                timeout=35,
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])

            for update in updates:
                _bot_state["last_update_id"] = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                # Only respond to configured chat
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue

                if text == "/pause":
                    _bot_state["paused"] = True
                    send_alert("Bot paused. Send /resume to continue.")
                elif text == "/resume":
                    _bot_state["paused"] = False
                    send_alert("Bot resumed.")
                elif text == "/scan":
                    _bot_state["scan_requested"] = True
                    send_alert("Scan requested. Running now...")
                elif text == "/status":
                    status = "Paused" if _bot_state["paused"] else "Running"
                    send_alert(f"*Bot Status:* {status}")

        except Exception as e:
            logger.error("Telegram polling error: %s", e)
            time.sleep(5)


def start_polling():
    """Start background command polling thread."""
    if not is_configured():
        logger.info("Telegram not configured, skipping command polling")
        return
    thread = threading.Thread(target=_poll_commands, daemon=True)
    thread.start()


def is_paused() -> bool:
    return _bot_state["paused"]


def consume_scan_request() -> bool:
    """Check and consume a pending scan request."""
    if _bot_state["scan_requested"]:
        _bot_state["scan_requested"] = False
        return True
    return False
