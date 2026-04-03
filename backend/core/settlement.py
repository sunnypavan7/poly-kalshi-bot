"""Trade settlement logic for BTC 5-min and weather markets."""
import httpx
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session

from backend.models.database import Trade, BotState, Signal

logger = logging.getLogger("trading_bot")


async def fetch_polymarket_resolution(market_id: str, event_slug: Optional[str] = None) -> Tuple[bool, Optional[float]]:
    """
    Fetch actual market resolution from Polymarket API.

    Returns: (is_resolved, settlement_value)
        - settlement_value: 1.0 if Up/Yes won, 0.0 if Down/No won
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if event_slug:
                response = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"slug": event_slug}
                )
                response.raise_for_status()
                events = response.json()

                if events:
                    event = events[0] if isinstance(events, list) else events
                    markets = event.get("markets", [])
                    if markets:
                        return _parse_market_resolution(markets[0])

            url = f"https://gamma-api.polymarket.com/markets/{market_id}"
            response = await client.get(url)

            if response.status_code == 404:
                return await _search_market_in_events(market_id)

            response.raise_for_status()
            market = response.json()
            return _parse_market_resolution(market)

    except Exception as e:
        logger.warning("Failed to fetch resolution for %s: %s", event_slug or market_id, e)
        return False, None


async def _search_market_in_events(market_id: str) -> Tuple[bool, Optional[float]]:
    """Search for market in events (both active and closed)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for closed in [True, False]:
                params = {
                    "closed": str(closed).lower(),
                    "limit": 200
                }
                response = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params=params
                )
                response.raise_for_status()
                events = response.json()

                for event in events:
                    for market in event.get("markets", []):
                        if str(market.get("id")) == str(market_id):
                            return _parse_market_resolution(market)

        return False, None

    except Exception as e:
        logger.warning("Failed to search for market %s: %s", market_id, e)
        return False, None


def _parse_market_resolution(market: dict) -> Tuple[bool, Optional[float]]:
    """
    Parse market data to determine if resolved and outcome.

    - outcomePrices[0] > 0.99 -> first outcome won (Yes or Up)
    - outcomePrices[0] < 0.01 -> second outcome won (No or Down)
    """
    is_closed = market.get("closed", False)

    if not is_closed:
        return False, None

    outcome_prices = market.get("outcomePrices", [])
    if not outcome_prices:
        return False, None

    try:
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)

        first_price = float(outcome_prices[0]) if outcome_prices else 0.5

        if first_price > 0.99:
            logger.info("Market %s resolved: UP/YES won", market.get("id"))
            return True, 1.0
        elif first_price < 0.01:
            logger.info("Market %s resolved: DOWN/NO won", market.get("id"))
            return True, 0.0
        else:
            return False, None

    except (ValueError, IndexError, TypeError) as e:
        logger.warning("Failed to parse outcome prices: %s", e)
        return False, None


def calculate_pnl(trade: Trade, settlement_value: float) -> float:
    """
    Calculate P&L for a trade given the settlement value.

    Maps up->yes, down->no internally:
    - UP/YES position wins when settlement = 1.0
    - DOWN/NO position wins when settlement = 0.0
    """
    direction = trade.direction
    if direction == "up":
        direction = "yes"
    elif direction == "down":
        direction = "no"

    if direction == "yes":
        if settlement_value == 1.0:
            pnl = trade.size * (1.0 - trade.entry_price)
        else:
            pnl = -trade.size * trade.entry_price
    else:
        if settlement_value == 0.0:
            pnl = trade.size * (1.0 - trade.entry_price)
        else:
            pnl = -trade.size * trade.entry_price

    return round(pnl, 2)


async def check_market_settlement(trade: Trade) -> Tuple[bool, Optional[float], Optional[float]]:
    """Check if a trade's market has settled."""
    is_resolved, settlement_value = await fetch_polymarket_resolution(
        trade.market_ticker,
        event_slug=trade.event_slug
    )

    if not is_resolved or settlement_value is None:
        return False, None, None

    pnl = calculate_pnl(trade, settlement_value)

    mapped_dir = "UP" if trade.direction in ("up", "yes") else "DOWN"
    outcome = "UP" if settlement_value == 1.0 else "DOWN"
    result = "WIN" if mapped_dir == outcome else "LOSS"

    logger.info("Trade %s settled: %s @ %.0f%% -> %s P&L: $%+.2f",
                trade.id, mapped_dir, trade.entry_price * 100, result, pnl)

    return True, settlement_value, pnl


async def check_weather_settlement(trade: Trade) -> Tuple[bool, Optional[float], Optional[float]]:
    """Check if a weather trade's market has settled."""
    platform = getattr(trade, 'platform', 'polymarket') or 'polymarket'

    if platform == "kalshi":
        is_resolved, settlement_value = await _fetch_kalshi_resolution(trade.market_ticker)
    else:
        is_resolved, settlement_value = await fetch_polymarket_resolution(
            trade.market_ticker,
            event_slug=trade.event_slug,
        )

    if is_resolved and settlement_value is not None:
        pnl = calculate_pnl(trade, settlement_value)
        return True, settlement_value, pnl

    return False, None, None


async def _fetch_kalshi_resolution(ticker: str) -> Tuple[bool, Optional[float]]:
    """Fetch resolution status for a Kalshi market."""
    try:
        from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present

        if not kalshi_credentials_present():
            return False, None

        client = KalshiClient()
        data = await client.get_market(ticker)
        market = data.get("market", data)

        status = market.get("status", "")
        result = market.get("result", "")

        if status in ("finalized", "determined") and result:
            if result == "yes":
                return True, 1.0
            elif result == "no":
                return True, 0.0

        return False, None

    except Exception as e:
        logger.warning("Failed to fetch Kalshi resolution for %s: %s", ticker, e)
        return False, None


async def settle_pending_trades(db: Session) -> List[Trade]:
    """Process all pending trades for settlement using REAL market outcomes."""
    try:
        pending = db.query(Trade).filter(Trade.settled == False).all()
    except Exception as e:
        logger.error("Failed to query pending trades: %s", e)
        return []

    if not pending:
        logger.info("No pending trades to settle")
        return []

    logger.info("Checking %d pending trades for settlement...", len(pending))
    settled_trades = []

    for trade in pending:
        try:
            market_type = getattr(trade, 'market_type', 'btc') or 'btc'
            if market_type == "weather":
                is_settled, settlement_value, pnl = await check_weather_settlement(trade)
            else:
                is_settled, settlement_value, pnl = await check_market_settlement(trade)

            if is_settled and settlement_value is not None:
                trade.settled = True
                trade.settlement_value = settlement_value
                trade.pnl = pnl
                trade.settlement_time = datetime.now(timezone.utc)

                if pnl is not None and pnl > 0:
                    trade.result = "win"
                elif pnl is not None and pnl < 0:
                    trade.result = "loss"
                else:
                    trade.result = "push"

                settled_trades.append(trade)

                # Update linked Signal for calibration
                if trade.signal_id:
                    linked_signal = db.query(Signal).filter(Signal.id == trade.signal_id).first()
                    if linked_signal:
                        actual_outcome = "up" if settlement_value == 1.0 else "down"
                        linked_signal.actual_outcome = actual_outcome
                        linked_signal.outcome_correct = (linked_signal.direction == actual_outcome)
                        linked_signal.settlement_value = settlement_value
                        linked_signal.settled_at = datetime.now(timezone.utc)
        except Exception as e:
            logger.error("Failed to settle trade %s: %s", trade.id, e)
            continue

    if settled_trades:
        try:
            db.commit()
            logger.info("Settled %d trades", len(settled_trades))
        except Exception as e:
            logger.error("Failed to commit settlements: %s", e)
            db.rollback()
            return []
    else:
        logger.info("No trades ready for settlement (markets still open)")

    return settled_trades


async def update_bot_state_with_settlements(db: Session, settled_trades: List[Trade]) -> None:
    """Update bot state with P&L from settled trades."""
    if not settled_trades:
        return

    try:
        state = db.query(BotState).first()
        if not state:
            logger.warning("Bot state not found")
            return

        for trade in settled_trades:
            if trade.pnl is not None:
                state.total_pnl += trade.pnl
                state.bankroll += trade.pnl
                if trade.result == "win":
                    state.winning_trades += 1

        db.commit()
        logger.info("Updated bot state: Bankroll $%.2f, P&L $%+.2f", state.bankroll, state.total_pnl)
    except Exception as e:
        logger.error("Failed to update bot state: %s", e)
        db.rollback()
