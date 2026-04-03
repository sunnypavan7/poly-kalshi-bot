"""
PolyWeatherBot — Rich CLI Dashboard
Live terminal display: signals, positions, stats, countdown.
"""

import logging
import time
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.config import settings, CITIES
from cli.scanner import get_last_signals, get_trade_history, get_scan_count

logger = logging.getLogger(__name__)
console = Console()


def _build_header() -> Panel:
    mode = "[bold yellow]PAPER[/]" if settings.PAPER_TRADING else "[bold red]LIVE[/]"
    header = Text.from_markup(
        f"[bold cyan]PolyWeatherBot[/] | Mode: {mode} | "
        f"Bankroll: ${settings.BANKROLL:.0f} | Min Edge: {settings.MIN_EDGE:.0%} | "
        f"Scans: {get_scan_count()} | {datetime.now().strftime('%H:%M:%S')}"
    )
    return Panel(header, style="bold blue")


def _build_signals_table() -> Table:
    signals = get_last_signals()
    table = Table(title="Current Signals", expand=True)
    table.add_column("City", style="cyan", width=12)
    table.add_column("Side", style="bold", width=5)
    table.add_column("Model", justify="right", width=7)
    table.add_column("Market", justify="right", width=7)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("Bet $", justify="right", width=8)
    table.add_column("EV", justify="right", width=7)

    if not signals:
        table.add_row("-", "-", "-", "-", "No signals", "-", "-")
    else:
        for sig in signals[:15]:
            edge_style = "green" if sig.edge > 0.15 else "yellow"
            table.add_row(
                sig.city[:12],
                sig.side,
                f"{sig.model_prob:.1%}",
                f"{sig.market_price:.1%}",
                f"[{edge_style}]{sig.edge:.1%}[/]",
                f"${sig.kelly_bet:.2f}",
                f"{sig.expected_value:.1%}",
            )
    return table


def _build_trades_table() -> Table:
    trades = get_trade_history()
    table = Table(title="Recent Trades", expand=True)
    table.add_column("Time", width=8)
    table.add_column("City", style="cyan", width=12)
    table.add_column("Side", width=5)
    table.add_column("Price", justify="right", width=7)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Edge", justify="right", width=7)

    if not trades:
        table.add_row("-", "-", "-", "-", "No trades yet", "-")
    else:
        for trade in trades[-10:]:
            ts = datetime.fromtimestamp(trade["timestamp"]).strftime("%H:%M:%S")
            table.add_row(
                ts,
                trade["city"][:12],
                trade["side"],
                f"{trade['price']:.4f}",
                f"${trade['size']:.2f}",
                f"{trade['edge']:.1%}",
            )
    return table


def _build_stats_panel() -> Panel:
    trades = get_trade_history()
    total_invested = sum(t["size"] for t in trades)
    avg_edge = sum(t["edge"] for t in trades) / len(trades) if trades else 0

    stats = (
        f"Total Trades: {len(trades)}\n"
        f"Total Invested: ${total_invested:.2f}\n"
        f"Avg Edge: {avg_edge:.1%}\n"
        f"Scan Interval: {settings.SCAN_INTERVAL_SEC}s"
    )
    return Panel(stats, title="Statistics", border_style="green")


def build_dashboard() -> Layout:
    """Build the full dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(_build_header(), size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(_build_signals_table(), name="signals"),
        Layout(_build_trades_table(), name="trades"),
    )
    layout["right"].update(_build_stats_panel())
    return layout


def run_dashboard(scan_func, execute_func):
    """
    Run the dashboard with live refresh.
    scan_func: callable that returns list of signals
    execute_func: callable that takes signals and executes trades
    """
    import cli.alerts as alerts_mod

    mode = "PAPER" if settings.PAPER_TRADING else "LIVE"
    console.print(f"[bold cyan]PolyWeatherBot[/] starting in [bold]{mode}[/] mode...")
    console.print(f"Scan interval: {settings.SCAN_INTERVAL_SEC}s | Min edge: {settings.MIN_EDGE:.0%}")
    console.print("[dim]Press Ctrl+C to stop[/]\n")

    alerts_mod.start_polling()
    alerts_mod.send_alert(f"PolyWeatherBot started in {mode} mode")

    with Live(build_dashboard(), console=console, refresh_per_second=1) as live:
        while True:
            try:
                if alerts_mod.is_paused():
                    time.sleep(1)
                    live.update(build_dashboard())
                    continue

                # Run scan
                signals = scan_func()
                if signals:
                    execute_func(signals)

                # Countdown to next scan
                for remaining in range(settings.SCAN_INTERVAL_SEC, 0, -1):
                    if alerts_mod.consume_scan_request():
                        break
                    live.update(build_dashboard())
                    time.sleep(1)

            except KeyboardInterrupt:
                alerts_mod.send_alert("PolyWeatherBot stopped")
                console.print("\n[bold red]Bot stopped.[/]")
                break
            except Exception as e:
                logger.error("Dashboard error: %s", e)
                time.sleep(5)
