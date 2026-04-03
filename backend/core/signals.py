"""Signal generator for BTC 5-minute Up/Down markets."""
import logging
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass, field
import asyncio

from backend.config import settings
from backend.data.btc_markets import BtcMarket, fetch_active_btc_markets
from backend.data.crypto import fetch_crypto_price, compute_btc_microstructure
from backend.models.database import SessionLocal, Signal

logger = logging.getLogger("trading_bot")


@dataclass
class TradingSignal:
    """A trading signal for a BTC 5-min market."""
    market: BtcMarket

    # Core signal data
    model_probability: float = 0.5
    market_probability: float = 0.5
    edge: float = 0.0
    direction: str = "up"  # "up" or "down"

    # Confidence and sizing
    confidence: float = 0.5
    kelly_fraction: float = 0.0
    suggested_size: float = 0.0

    # Metadata
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # BTC price context
    btc_price: float = 0.0
    btc_change_1h: float = 0.0
    btc_change_24h: float = 0.0

    @property
    def passes_threshold(self) -> bool:
        """Check if signal passes minimum edge threshold."""
        return abs(self.edge) >= settings.MIN_EDGE_THRESHOLD


def calculate_edge(
    model_prob: float,
    market_price: float
) -> tuple:
    """
    Calculate edge and determine direction.

    Returns:
        (edge, direction) where direction is "up" or "down"
    """
    up_edge = model_prob - market_price
    down_edge = (1 - model_prob) - (1 - market_price)

    if up_edge >= down_edge:
        return up_edge, "up"
    else:
        return down_edge, "down"


def calculate_kelly_size(
    edge: float,
    probability: float,
    market_price: float,
    direction: str,
    bankroll: float
) -> float:
    """
    Calculate position size using fractional Kelly criterion.

    Kelly formula: f = (p * b - q) / b
    """
    if direction == "up":
        win_prob = probability
        price = market_price
    else:
        win_prob = 1 - probability
        price = 1 - market_price

    if price <= 0 or price >= 1:
        return 0

    odds = (1 - price) / price
    lose_prob = 1 - win_prob
    kelly = (win_prob * odds - lose_prob) / odds

    # Apply fractional Kelly (15% for web mode)
    kelly *= settings.KELLY_FRACTION
    max_fraction = 0.05
    kelly = min(kelly, max_fraction)
    kelly = max(kelly, 0)

    size = kelly * bankroll
    size = min(size, settings.MAX_TRADE_SIZE)

    return size


async def generate_btc_signal(market: BtcMarket) -> Optional[TradingSignal]:
    """
    Generate a trading signal for a BTC 5-min Up/Down market.

    Uses real 1-minute candle data to compute:
    - RSI (mean reversion), Momentum (trend), VWAP deviation,
      SMA crossover, and market skew as a weighted composite.
    - Convergence filter: requires 2/4 indicators to agree.
    - Entry price filter: only enter when price <= MAX_ENTRY_PRICE.
    """
    try:
        micro = await compute_btc_microstructure()
    except Exception as e:
        logger.warning("Failed to compute microstructure: %s", e)
        return None

    if not micro:
        return None

    market_up_prob = market.up_price

    # Skip resolved markets
    if market_up_prob < 0.02 or market_up_prob > 0.98:
        return None

    # --- Individual indicator signals (each returns a bias from -1 to +1) ---

    # 1) RSI: mean reversion
    if micro.rsi < 30:
        rsi_signal = 0.5 + (30 - micro.rsi) / 30
    elif micro.rsi > 70:
        rsi_signal = -0.5 - (micro.rsi - 70) / 30
    elif micro.rsi < 45:
        rsi_signal = (45 - micro.rsi) / 30
    elif micro.rsi > 55:
        rsi_signal = -(micro.rsi - 55) / 30
    else:
        rsi_signal = 0.0
    rsi_signal = max(-1.0, min(1.0, rsi_signal))

    # 2) Momentum: weighted blend
    mom_blend = micro.momentum_1m * 0.5 + micro.momentum_5m * 0.35 + micro.momentum_15m * 0.15
    momentum_signal = max(-1.0, min(1.0, mom_blend / 0.10))

    # 3) VWAP deviation
    vwap_signal = max(-1.0, min(1.0, micro.vwap_deviation / 0.05))

    # 4) SMA crossover
    sma_signal = max(-1.0, min(1.0, micro.sma_crossover / 0.03))

    # 5) Market skew: contrarian
    market_skew = market_up_prob - 0.50
    skew_signal = max(-1.0, min(1.0, -market_skew * 4))

    # --- Convergence filter ---
    indicator_signs = [rsi_signal, momentum_signal, vwap_signal, sma_signal]
    up_votes = sum(1 for s in indicator_signs if s > 0.05)
    down_votes = sum(1 for s in indicator_signs if s < -0.05)
    has_convergence = up_votes >= 2 or down_votes >= 2

    # --- Weighted composite ---
    w = settings
    composite = (
        rsi_signal * w.WEIGHT_RSI
        + momentum_signal * w.WEIGHT_MOMENTUM
        + vwap_signal * w.WEIGHT_VWAP
        + sma_signal * w.WEIGHT_SMA
        + skew_signal * w.WEIGHT_MARKET_SKEW
    )

    # Convert composite to probability
    model_up_prob = 0.50 + composite * 0.15
    model_up_prob = max(0.35, min(0.65, model_up_prob))

    edge, direction = calculate_edge(model_up_prob, market_up_prob)

    # --- Entry price filter ---
    if direction == "up":
        entry_price = market_up_prob
    else:
        entry_price = market.down_price

    # Time-remaining filter
    now = datetime.now(timezone.utc)
    window_end = market.window_end
    if window_end.tzinfo is None:
        window_end = window_end.replace(tzinfo=timezone.utc)
    time_remaining = (window_end - now).total_seconds()
    time_ok = settings.MIN_TIME_REMAINING <= time_remaining <= settings.MAX_TIME_REMAINING

    passes_filters = has_convergence and entry_price <= settings.MAX_ENTRY_PRICE and time_ok

    if not passes_filters:
        edge = 0.0

    # Confidence
    vol_factor = min(1.0, micro.volatility / 0.05) if micro.volatility > 0 else 0.5
    convergence_strength = max(up_votes, down_votes) / 4.0
    confidence = min(0.8, 0.3 + convergence_strength * 0.3 + abs(composite) * 0.2) * vol_factor

    # Kelly sizing
    bankroll = settings.INITIAL_BANKROLL
    suggested_size = calculate_kelly_size(
        edge=abs(edge),
        probability=model_up_prob,
        market_price=market_up_prob,
        direction=direction,
        bankroll=bankroll,
    )

    # Build reasoning
    filter_status = "ACTIONABLE" if passes_filters else "FILTERED"
    filter_reasons = []
    if not has_convergence:
        filter_reasons.append(f"convergence {max(up_votes, down_votes)}/4 < 2")
    if not time_ok:
        filter_reasons.append(f"time {time_remaining:.0f}s not in [{settings.MIN_TIME_REMAINING},{settings.MAX_TIME_REMAINING}]")
    if entry_price > settings.MAX_ENTRY_PRICE:
        filter_reasons.append(f"entry {entry_price:.0%} > {settings.MAX_ENTRY_PRICE:.0%}")
    filter_note = f" [{', '.join(filter_reasons)}]" if filter_reasons else ""

    reasoning = (
        f"[{filter_status}]{filter_note} "
        f"BTC ${micro.price:,.0f} | RSI:{micro.rsi:.0f} Mom1m:{micro.momentum_1m:+.3f}% "
        f"Mom5m:{micro.momentum_5m:+.3f}% VWAP:{micro.vwap_deviation:+.3f}% "
        f"SMA:{micro.sma_crossover:+.4f}% Vol:{micro.volatility:.4f}% | "
        f"Composite:{composite:+.3f} -> Model UP:{model_up_prob:.0%} vs Mkt:{market_up_prob:.0%} | "
        f"Edge:{edge:+.1%} -> {direction.upper()} @ {entry_price:.0%} | "
        f"Convergence:{max(up_votes, down_votes)}/4 | "
        f"Window ends: {market.window_end.strftime('%H:%M UTC')}"
    )

    return TradingSignal(
        market=market,
        model_probability=model_up_prob,
        market_probability=market_up_prob,
        edge=edge,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=[f"binance_microstructure_{micro.source}"],
        reasoning=reasoning,
        btc_price=micro.price,
        btc_change_1h=micro.momentum_5m * 12,
        btc_change_24h=micro.momentum_15m * 96,
    )


async def scan_for_signals() -> List[TradingSignal]:
    """Scan BTC 5-min markets and generate signals."""
    signals = []

    logger.info("=" * 50)
    logger.info("BTC 5-MIN SCAN: Fetching markets from Polymarket...")

    try:
        markets = await fetch_active_btc_markets()
    except Exception as e:
        logger.error("Failed to fetch BTC markets: %s", e)
        markets = []

    logger.info("Found %d active BTC 5-min markets", len(markets))

    for market in markets:
        try:
            signal = await generate_btc_signal(market)
            if signal:
                signals.append(signal)
        except Exception as e:
            logger.debug("Signal generation failed for %s: %s", market.slug, e)

        await asyncio.sleep(0.1)

    signals.sort(key=lambda s: abs(s.edge), reverse=True)

    actionable = [s for s in signals if s.passes_threshold]
    logger.info("=" * 50)
    logger.info("SCAN COMPLETE: %d signals, %d actionable", len(signals), len(actionable))

    for signal in actionable[:5]:
        logger.info("  %s", signal.market.slug)
        logger.info("    Edge: %+.1f%% -> %s @ $%.2f", signal.edge * 100, signal.direction.upper(), signal.suggested_size)

    _persist_signals(signals)

    return signals


def _persist_signals(signals: list):
    """Save signals with non-zero edge to DB for calibration tracking."""
    to_save = [s for s in signals if abs(s.edge) > 0]
    if not to_save:
        return

    db = SessionLocal()
    try:
        for signal in to_save:
            existing = db.query(Signal).filter(
                Signal.market_ticker == signal.market.market_id,
                Signal.timestamp >= signal.timestamp.replace(second=0, microsecond=0),
            ).first()
            if existing:
                continue

            db_signal = Signal(
                market_ticker=signal.market.market_id,
                platform="polymarket",
                timestamp=signal.timestamp,
                direction=signal.direction,
                model_probability=signal.model_probability,
                market_price=signal.market_probability,
                edge=signal.edge,
                confidence=signal.confidence,
                kelly_fraction=signal.kelly_fraction,
                suggested_size=signal.suggested_size,
                sources=signal.sources,
                reasoning=signal.reasoning,
                executed=False,
            )
            db.add(db_signal)

        db.commit()
    except Exception as e:
        logger.warning("Failed to persist signals: %s", e)
        db.rollback()
    finally:
        db.close()


async def get_actionable_signals() -> List[TradingSignal]:
    """Get only signals that pass the edge threshold."""
    all_signals = await scan_for_signals()
    return [s for s in all_signals if s.passes_threshold]
