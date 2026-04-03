"""
PolyWeatherBot — CLI Scanner
Main scan loop: fetch forecasts, find markets, compute edges, execute trades.
"""

import logging
import time
from typing import Optional

from backend.config import settings, CITIES
from backend.data.weather import get_forecast, prob_above, prob_below
from backend.data.polymarket import fetch_weather_markets, place_limit_order, WeatherMarket
from backend.core.edge import evaluate_market, Signal
import cli.alerts as alerts

logger = logging.getLogger(__name__)

# Track signals for dashboard
_last_signals: list = []
_trade_history: list = []
_scan_count = 0


def get_last_signals() -> list:
    return list(_last_signals)


def get_trade_history() -> list:
    return list(_trade_history)


def get_scan_count() -> int:
    return _scan_count


def _match_city_key(market_city: str) -> Optional[str]:
    """Match a market's city name to our CITIES dict key."""
    mc = market_city.lower().strip()
    for key, info in CITIES.items():
        name_lower = info["name"].lower()
        if mc in name_lower or name_lower in mc:
            return key
        if key == "nyc" and ("new york" in mc or "nyc" in mc):
            return key
        if key == "los_angeles" and ("los angeles" in mc or " la " in mc):
            return key
        if key == "sao_paulo" and ("sao paulo" in mc or "são paulo" in mc):
            return key
    return None


def scan_once() -> list:
    """Run a single scan: fetch markets, compute forecasts, find edges."""
    global _last_signals, _scan_count
    _scan_count += 1

    logger.info("=== Scan #%d starting ===", _scan_count)

    # Step 1: Discover weather markets
    markets = fetch_weather_markets()
    if not markets:
        logger.info("No active weather markets found")
        _last_signals = []
        return []

    logger.info("Found %d weather markets to evaluate", len(markets))

    all_signals = []

    # Step 2: For each market, get forecast and compute edge
    for market in markets:
        city_key = _match_city_key(market.city)
        if not city_key:
            logger.debug("City not tracked: %s", market.city)
            continue

        city_info = CITIES[city_key]
        forecast = get_forecast(city_info["lat"], city_info["lon"], city_info["name"])
        if not forecast:
            logger.warning("No forecast available for %s", city_info["name"])
            continue

        # Compute model probability based on market direction
        if market.threshold_c is None:
            continue

        if market.direction == "above":
            model_prob = prob_above(
                market.threshold_c, forecast.mean_high, forecast.std_high
            )
        else:
            model_prob = prob_below(
                market.threshold_c, forecast.mean_low, forecast.std_low
            )

        # Evaluate for trading signals
        signals = evaluate_market(
            city=city_info["name"],
            question=market.question,
            direction=market.direction,
            threshold_c=market.threshold_c,
            model_prob=model_prob,
            market_price_yes=market.best_ask_yes,
            market_price_no=market.best_ask_no,
            token_id_yes=market.token_id_yes,
            token_id_no=market.token_id_no,
            condition_id=market.condition_id,
            min_edge=settings.MIN_EDGE,
        )
        all_signals.extend(signals)

    _last_signals = all_signals
    logger.info("Scan #%d complete: %d signals found", _scan_count, len(all_signals))

    return all_signals


def execute_signals(signals: list) -> list:
    """Execute trades for all signals. Returns list of trade results."""
    results = []
    for sig in signals:
        alerts.send_signal_alert(sig)

        result = place_limit_order(
            token_id=sig.token_id,
            side="BUY",
            price=sig.market_price,
            size=sig.kelly_bet,
        )

        if result:
            alerts.send_trade_alert(sig, result)
            trade_record = {
                "timestamp": time.time(),
                "city": sig.city,
                "side": sig.side,
                "price": sig.market_price,
                "size": sig.kelly_bet,
                "edge": sig.edge,
                "model_prob": sig.model_prob,
                "result": result,
            }
            _trade_history.append(trade_record)
            results.append(trade_record)

    return results


def run_loop():
    """Main scanning loop. Runs until interrupted."""
    mode = "PAPER" if settings.PAPER_TRADING else "LIVE"
    logger.info("Starting scan loop in %s mode (interval: %ds)", mode, settings.SCAN_INTERVAL_SEC)

    alerts.start_polling()
    alerts.send_alert(f"PolyWeatherBot started in *{mode}* mode")

    while True:
        try:
            if alerts.is_paused():
                logger.info("Bot is paused, waiting...")
                time.sleep(5)
                continue

            if alerts.consume_scan_request():
                logger.info("Manual scan requested via Telegram")

            signals = scan_once()

            if signals:
                execute_signals(signals)
            else:
                logger.info("No signals this scan")

            logger.info("Next scan in %d seconds", settings.SCAN_INTERVAL_SEC)
            for _ in range(settings.SCAN_INTERVAL_SEC):
                if alerts.consume_scan_request():
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Scan loop stopped by user")
            alerts.send_alert("PolyWeatherBot stopped")
            break
        except Exception as e:
            logger.error("Scan loop error: %s", e)
            time.sleep(30)
