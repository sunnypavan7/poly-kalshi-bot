"""
PolyWeatherBot — Backtesting Engine
Uses REAL historical weather data from Open-Meteo archive (free, back to 1940).
NO random numbers for weather. Simulates market prices from model probabilities + noise.
Uses norm.cdf (NEVER norm.pdf) for probability calculations.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.config import settings, CITIES
from backend.data.weather import fetch_historical_temps, prob_above, prob_below

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""
    date: str
    city: str
    direction: str
    threshold_c: float
    model_prob: float
    market_price: float
    edge: float
    bet_size: float
    side: str           # "YES" or "NO"
    outcome: bool       # did the event actually happen?
    pnl: float          # profit or loss for this trade


@dataclass
class BacktestResult:
    """Full backtest results with statistics."""
    trades: list = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_edge: float = 0.0
    monthly_breakdown: dict = field(default_factory=dict)
    cities_tested: list = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""


def _generate_thresholds(highs: list, lows: list) -> list:
    """
    Generate realistic temperature thresholds for backtesting.
    Uses percentiles of actual data to create meaningful questions.
    """
    h = np.array([x for x in highs if x is not None], dtype=float)
    lo = np.array([x for x in lows if x is not None], dtype=float)

    if len(h) == 0 or len(lo) == 0:
        return []

    thresholds = []

    for pct in [25, 50, 75, 90]:
        val = float(np.percentile(h, pct))
        thresholds.append({
            "threshold_c": round(val, 1),
            "direction": "above",
            "description": f"High above {val:.1f}C (p{pct})",
        })

    for pct in [10, 25, 50, 75]:
        val = float(np.percentile(lo, pct))
        thresholds.append({
            "threshold_c": round(val, 1),
            "direction": "below",
            "description": f"Low below {val:.1f}C (p{pct})",
        })

    return thresholds


def _simulate_market_price(model_prob: float, rng: np.random.Generator) -> float:
    """
    Simulate a market price from the model probability.
    Adds noise to create realistic mispricings.
    Uses a seeded RNG for reproducibility.
    """
    noise = rng.normal(0, 0.12)
    market_price = model_prob + noise
    return float(np.clip(market_price, 0.05, 0.95))


def run_backtest(
    city_keys: list,
    start_date: str,
    end_date: str,
    bankroll: float = None,
    min_edge: float = None,
    kelly_fraction: float = None,
    max_bet: float = None,
    seed: int = 42,
) -> BacktestResult:
    """
    Run a full backtest using real historical weather data.

    Args:
        city_keys: List of city keys from CITIES dict
        start_date: Start date as "YYYY-MM-DD"
        end_date: End date as "YYYY-MM-DD"
        bankroll: Starting bankroll in USD
        min_edge: Minimum edge to trade
        kelly_fraction: Kelly fraction for bet sizing
        max_bet: Maximum bet size in USD
        seed: Random seed for reproducible market price simulation
    """
    if bankroll is None:
        bankroll = settings.BANKROLL
    if min_edge is None:
        min_edge = settings.MIN_EDGE
    if kelly_fraction is None:
        kelly_fraction = settings.CLI_KELLY_FRACTION
    if max_bet is None:
        max_bet = settings.MAX_BET_USD

    rng = np.random.default_rng(seed)
    result = BacktestResult(
        start_date=start_date,
        end_date=end_date,
    )

    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown = 0.0
    daily_returns = []

    logger.info("=" * 60)
    logger.info("BACKTEST: %s to %s", start_date, end_date)
    logger.info("Cities: %s", ", ".join(city_keys))
    logger.info("Bankroll: $%.2f | Min Edge: %.0f%% | Kelly: %.0f%%",
                bankroll, min_edge * 100, kelly_fraction * 100)
    logger.info("=" * 60)

    for city_key in city_keys:
        if city_key not in CITIES:
            logger.warning("Unknown city key: %s, skipping", city_key)
            continue

        city_info = CITIES[city_key]
        city_name = city_info["name"]
        result.cities_tested.append(city_name)

        logger.info("Fetching historical data for %s...", city_name)

        hist = fetch_historical_temps(
            lat=city_info["lat"],
            lon=city_info["lon"],
            start_date=start_date,
            end_date=end_date,
        )

        if not hist:
            logger.warning("No historical data for %s, skipping", city_name)
            continue

        dates = hist["dates"]
        highs = hist["highs"]
        lows = hist["lows"]

        logger.info("  Got %d days of data for %s", len(dates), city_name)

        thresholds = _generate_thresholds(highs, lows)
        if not thresholds:
            continue

        window = 30

        for i in range(window, len(dates)):
            day_date = dates[i]
            actual_high = highs[i]
            actual_low = lows[i]

            if actual_high is None or actual_low is None:
                continue

            recent_highs = [h for h in highs[i - window:i] if h is not None]
            recent_lows = [lo for lo in lows[i - window:i] if lo is not None]

            if len(recent_highs) < 10 or len(recent_lows) < 10:
                continue

            forecast_mean_high = float(np.mean(recent_highs))
            forecast_std_high = max(float(np.std(recent_highs)), 0.5)
            forecast_mean_low = float(np.mean(recent_lows))
            forecast_std_low = max(float(np.std(recent_lows)), 0.5)

            day_pnl = 0.0

            for thresh in thresholds:
                tc = thresh["threshold_c"]
                direction = thresh["direction"]

                # norm.cdf based probability (NEVER norm.pdf)
                if direction == "above":
                    model_prob = prob_above(tc, forecast_mean_high, forecast_std_high)
                    actual_outcome = actual_high > tc
                else:
                    model_prob = prob_below(tc, forecast_mean_low, forecast_std_low)
                    actual_outcome = actual_low < tc

                if model_prob < 0.05 or model_prob > 0.95:
                    continue

                market_price = _simulate_market_price(model_prob, rng)

                # Check YES side
                edge_yes = model_prob - market_price
                if edge_yes >= min_edge:
                    bet = min(
                        (edge_yes / (1.0 - market_price)) * kelly_fraction * current_bankroll,
                        max_bet,
                    )
                    bet = max(bet, 0.0)

                    if actual_outcome:
                        pnl = bet * (1.0 / market_price - 1.0)
                        result.wins += 1
                    else:
                        pnl = -bet
                        result.losses += 1

                    current_bankroll += pnl
                    day_pnl += pnl

                    result.trades.append(BacktestTrade(
                        date=day_date,
                        city=city_name,
                        direction=direction,
                        threshold_c=tc,
                        model_prob=model_prob,
                        market_price=market_price,
                        edge=edge_yes,
                        bet_size=bet,
                        side="YES",
                        outcome=actual_outcome,
                        pnl=pnl,
                    ))

                # Check NO side
                model_prob_no = 1.0 - model_prob
                market_price_no = 1.0 - market_price
                edge_no = model_prob_no - market_price_no
                if edge_no >= min_edge and market_price_no > 0.05:
                    bet = min(
                        (edge_no / (1.0 - market_price_no)) * kelly_fraction * current_bankroll,
                        max_bet,
                    )
                    bet = max(bet, 0.0)

                    actual_outcome_no = not actual_outcome
                    if actual_outcome_no:
                        pnl = bet * (1.0 / market_price_no - 1.0)
                        result.wins += 1
                    else:
                        pnl = -bet
                        result.losses += 1

                    current_bankroll += pnl
                    day_pnl += pnl

                    result.trades.append(BacktestTrade(
                        date=day_date,
                        city=city_name,
                        direction=direction,
                        threshold_c=tc,
                        model_prob=model_prob_no,
                        market_price=market_price_no,
                        edge=edge_no,
                        bet_size=bet,
                        side="NO",
                        outcome=actual_outcome_no,
                        pnl=pnl,
                    ))

            if day_pnl != 0:
                daily_returns.append(day_pnl)

            peak_bankroll = max(peak_bankroll, current_bankroll)
            dd = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            max_drawdown = max(max_drawdown, dd)

            if current_bankroll <= 0:
                logger.warning("Bankrupt on %s! Stopping backtest.", day_date)
                break

        if current_bankroll <= 0:
            break

    # Compute final statistics
    result.total_trades = len(result.trades)
    result.total_pnl = current_bankroll - bankroll
    result.roi = (result.total_pnl / bankroll) * 100 if bankroll > 0 else 0
    result.win_rate = (result.wins / result.total_trades * 100) if result.total_trades > 0 else 0
    result.max_drawdown = max_drawdown * 100
    result.avg_edge = (
        float(np.mean([t.edge for t in result.trades])) * 100
        if result.trades else 0
    )

    if daily_returns:
        dr = np.array(daily_returns)
        mean_ret = np.mean(dr)
        std_ret = np.std(dr)
        result.sharpe_ratio = (
            (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        )

    gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    for trade in result.trades:
        month_key = trade.date[:7]
        if month_key not in result.monthly_breakdown:
            result.monthly_breakdown[month_key] = {
                "trades": 0, "wins": 0, "pnl": 0.0,
            }
        result.monthly_breakdown[month_key]["trades"] += 1
        if trade.pnl > 0:
            result.monthly_breakdown[month_key]["wins"] += 1
        result.monthly_breakdown[month_key]["pnl"] += trade.pnl

    return result


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report to the console."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print()
    console.print(Panel(
        f"[bold cyan]PolyWeatherBot Backtest Report[/]\n"
        f"Period: {result.start_date} to {result.end_date}\n"
        f"Cities: {', '.join(result.cities_tested)}",
        border_style="cyan",
    ))

    stats_table = Table(title="Performance Summary", show_header=False, expand=True)
    stats_table.add_column("Metric", style="cyan", width=20)
    stats_table.add_column("Value", justify="right", width=15)

    pnl_style = "green" if result.total_pnl >= 0 else "red"
    roi_style = "green" if result.roi >= 0 else "red"

    stats_table.add_row("Total Trades", str(result.total_trades))
    stats_table.add_row("Wins / Losses", f"{result.wins} / {result.losses}")
    stats_table.add_row("Win Rate", f"{result.win_rate:.1f}%")
    stats_table.add_row("Total P&L", f"[{pnl_style}]${result.total_pnl:+,.2f}[/]")
    stats_table.add_row("ROI", f"[{roi_style}]{result.roi:+.1f}%[/]")
    stats_table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    stats_table.add_row("Max Drawdown", f"{result.max_drawdown:.1f}%")
    stats_table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    stats_table.add_row("Avg Edge", f"{result.avg_edge:.1f}%")

    console.print(stats_table)

    if result.monthly_breakdown:
        monthly_table = Table(title="Monthly Breakdown", expand=True)
        monthly_table.add_column("Month", style="cyan")
        monthly_table.add_column("Trades", justify="right")
        monthly_table.add_column("Win Rate", justify="right")
        monthly_table.add_column("P&L", justify="right")

        for month in sorted(result.monthly_breakdown.keys()):
            data = result.monthly_breakdown[month]
            wr = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            pnl_s = "green" if data["pnl"] >= 0 else "red"
            monthly_table.add_row(
                month,
                str(data["trades"]),
                f"{wr:.0f}%",
                f"[{pnl_s}]${data['pnl']:+,.2f}[/]",
            )

        console.print(monthly_table)

    if result.trades:
        city_stats = {}
        for t in result.trades:
            if t.city not in city_stats:
                city_stats[t.city] = {"trades": 0, "wins": 0, "pnl": 0.0}
            city_stats[t.city]["trades"] += 1
            if t.pnl > 0:
                city_stats[t.city]["wins"] += 1
            city_stats[t.city]["pnl"] += t.pnl

        city_table = Table(title="City Breakdown", expand=True)
        city_table.add_column("City", style="cyan")
        city_table.add_column("Trades", justify="right")
        city_table.add_column("Win Rate", justify="right")
        city_table.add_column("P&L", justify="right")

        for city in sorted(city_stats.keys()):
            data = city_stats[city]
            wr = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            pnl_s = "green" if data["pnl"] >= 0 else "red"
            city_table.add_row(
                city,
                str(data["trades"]),
                f"{wr:.0f}%",
                f"[{pnl_s}]${data['pnl']:+,.2f}[/]",
            )

        console.print(city_table)

    console.print()
    console.print(Panel(
        "[yellow]DISCLAIMER:[/] Backtest uses simulated market prices (real weather data + "
        "noise-based market simulation). Real-world results will differ. "
        "Divide ROI by 3-5x for realistic expectations.",
        border_style="yellow",
    ))
