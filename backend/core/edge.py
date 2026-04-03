"""
Unified Edge Calculation & Kelly Sizing
Merged from P1 (CLI weather) and P2 (web BTC/weather).

Edge = model_probability - market_ask_price
Kelly fraction for position sizing.
Uses norm.cdf (NEVER norm.pdf) for weather probability calculations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal with edge, size, and expected value."""
    city: str
    question: str
    direction: str              # "above" or "below" for weather; "up" or "down" for BTC
    threshold_c: Optional[float]  # Celsius threshold (weather only)
    model_prob: float           # our estimated probability (0-1)
    market_price: float         # market ask price (0-1)
    edge: float                 # model_prob - market_price
    kelly_bet: float            # optimal bet in USD
    expected_value: float       # EV per dollar
    side: str                   # "YES" or "NO"
    token_id: str
    condition_id: str
    exchange: str = "polymarket"  # "polymarket" or "kalshi"
    strategy: str = "weather"     # "weather" or "btc"


def calculate_edge(model_prob: float, market_price: float) -> float:
    """
    Edge = model_probability - market_ask_price.
    Positive edge means the model thinks the event is more likely than the market.
    """
    return model_prob - market_price


def kelly_bet_size(
    edge: float,
    market_price: float,
    bankroll: float = None,
    kelly_frac: float = None,
    max_bet: float = None,
) -> float:
    """
    Kelly criterion bet sizing.
    bet_size = (edge / (1 - market_price)) * kelly_fraction * bankroll
    Capped at max_bet.
    """
    if bankroll is None:
        bankroll = settings.INITIAL_BANKROLL
    if kelly_frac is None:
        kelly_frac = settings.CLI_KELLY_FRACTION
    if max_bet is None:
        max_bet = settings.MAX_TRADE_SIZE

    if market_price >= 1.0 or market_price <= 0.0:
        return 0.0
    if edge <= 0:
        return 0.0

    raw_kelly = (edge / (1.0 - market_price)) * kelly_frac * bankroll
    return min(max(raw_kelly, 0.0), max_bet)


def expected_value(model_prob: float, market_price: float) -> float:
    """
    Expected value per dollar bet.
    EV = model_prob / market_price - 1
    """
    if market_price <= 0:
        return 0.0
    return model_prob / market_price - 1.0


def evaluate_market(
    city: str,
    question: str,
    direction: str,
    threshold_c: Optional[float],
    model_prob: float,
    market_price_yes: float,
    market_price_no: float,
    token_id_yes: str,
    token_id_no: str,
    condition_id: str,
    min_edge: float = None,
    exchange: str = "polymarket",
    strategy: str = "weather",
) -> list:
    """
    Evaluate a market for trading signals.
    Checks both YES and NO sides for edge.
    Returns list of signals that meet minimum edge threshold.
    """
    if min_edge is None:
        min_edge = settings.WEATHER_MIN_EDGE_THRESHOLD

    signals = []

    # Check YES side: model says event likely, market underprices YES
    edge_yes = calculate_edge(model_prob, market_price_yes)
    if edge_yes >= min_edge:
        bet = kelly_bet_size(edge_yes, market_price_yes)
        ev = expected_value(model_prob, market_price_yes)
        signals.append(Signal(
            city=city,
            question=question,
            direction=direction,
            threshold_c=threshold_c,
            model_prob=model_prob,
            market_price=market_price_yes,
            edge=edge_yes,
            kelly_bet=bet,
            expected_value=ev,
            side="YES",
            token_id=token_id_yes,
            condition_id=condition_id,
            exchange=exchange,
            strategy=strategy,
        ))
        logger.info(
            "Signal: %s YES | edge=%.3f prob=%.3f ask=%.3f bet=$%.2f",
            city, edge_yes, model_prob, market_price_yes, bet,
        )

    # Check NO side: model says event unlikely, market underprices NO
    model_prob_no = 1.0 - model_prob
    edge_no = calculate_edge(model_prob_no, market_price_no)
    if edge_no >= min_edge:
        bet = kelly_bet_size(edge_no, market_price_no)
        ev = expected_value(model_prob_no, market_price_no)
        signals.append(Signal(
            city=city,
            question=question,
            direction=direction,
            threshold_c=threshold_c,
            model_prob=model_prob_no,
            market_price=market_price_no,
            edge=edge_no,
            kelly_bet=bet,
            expected_value=ev,
            side="NO",
            token_id=token_id_no,
            condition_id=condition_id,
            exchange=exchange,
            strategy=strategy,
        ))
        logger.info(
            "Signal: %s NO | edge=%.3f prob=%.3f ask=%.3f bet=$%.2f",
            city, edge_no, model_prob_no, market_price_no, bet,
        )

    return signals
