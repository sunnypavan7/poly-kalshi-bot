"""
PolyWeatherBot — Unified Entry Point
Usage:
    python main.py --paper                  # Paper trading CLI (safe)
    python main.py                          # Live trading CLI
    python main.py --scan-once              # Single scan then exit
    python main.py --backtest --cities nyc london tokyo
    python main.py --backtest --cities nyc london --start 2024-04-01 --end 2026-04-01
    python main.py --web                    # Launch FastAPI web dashboard
    python main.py --web --port 3000        # Web dashboard on custom port
"""

import argparse
import logging
import sys

from backend.config import settings, CITIES

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PolyWeatherBot — Weather + BTC arbitrage trading bot",
    )
    parser.add_argument(
        "--paper", action="store_true", default=False,
        help="Force paper trading mode (no real money)",
    )
    parser.add_argument(
        "--scan-once", action="store_true", default=False,
        help="Run a single scan then exit",
    )
    parser.add_argument(
        "--backtest", action="store_true", default=False,
        help="Run historical backtest instead of live trading",
    )
    parser.add_argument(
        "--cities", nargs="+", default=["nyc", "london", "tokyo"],
        help="City keys for backtest (e.g., nyc london tokyo). Default: nyc london tokyo",
    )
    parser.add_argument(
        "--start", default="2024-04-01",
        help="Backtest start date (YYYY-MM-DD). Default: 2024-04-01",
    )
    parser.add_argument(
        "--end", default="2026-04-01",
        help="Backtest end date (YYYY-MM-DD). Default: 2026-04-01",
    )
    parser.add_argument(
        "--web", action="store_true", default=False,
        help="Launch FastAPI web dashboard instead of CLI",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for web dashboard (default: 8000)",
    )

    args = parser.parse_args()

    # Override paper trading if flag is set
    if args.paper:
        settings.PAPER_TRADING = True
        settings.SIMULATION_MODE = True

    # -------------------------------------------------------------------
    # WEB MODE — Launch FastAPI dashboard
    # -------------------------------------------------------------------
    if args.web:
        import uvicorn
        from backend.models.database import init_db

        logger.info("Initializing database...")
        init_db()

        logger.info("Starting web dashboard on http://0.0.0.0:%d", args.port)
        logger.info("API docs at http://localhost:%d/docs", args.port)

        uvicorn.run(
            "backend.api.main:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
        )
        return

    # -------------------------------------------------------------------
    # BACKTEST MODE
    # -------------------------------------------------------------------
    if args.backtest:
        from cli.backtest import run_backtest, print_backtest_report

        valid_cities = []
        for c in args.cities:
            if c in CITIES:
                valid_cities.append(c)
            else:
                logger.warning("Unknown city '%s'. Available: %s", c, ", ".join(CITIES.keys()))

        if not valid_cities:
            logger.error("No valid cities specified. Available: %s", ", ".join(CITIES.keys()))
            sys.exit(1)

        logger.info("Starting backtest: %s to %s for %s",
                     args.start, args.end, ", ".join(valid_cities))

        result = run_backtest(
            city_keys=valid_cities,
            start_date=args.start,
            end_date=args.end,
        )
        print_backtest_report(result)
        return

    # -------------------------------------------------------------------
    # SCAN-ONCE MODE
    # -------------------------------------------------------------------
    if args.scan_once:
        from cli.scanner import scan_once, execute_signals
        signals = scan_once()
        if signals:
            execute_signals(signals)
            logger.info("Scan complete: %d signals found and executed", len(signals))
        else:
            logger.info("Scan complete: no signals found")
        return

    # -------------------------------------------------------------------
    # LIVE / PAPER TRADING MODE (CLI with Rich dashboard)
    # -------------------------------------------------------------------
    from cli.scanner import scan_once, execute_signals
    from cli.dashboard import run_dashboard

    run_dashboard(scan_func=scan_once, execute_func=execute_signals)


if __name__ == "__main__":
    main()
