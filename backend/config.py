"""
PolyWeatherBot — Unified Configuration
Merges CLI bot settings (P1) with FastAPI dashboard settings (P2).
All secrets from .env via pydantic-settings. NEVER hardcode secrets.
"""

import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # -----------------------------------------------------------------------
    # Database (SQLite for local, PostgreSQL for production)
    # -----------------------------------------------------------------------
    DATABASE_URL: str = "sqlite:///./tradingbot.db"

    # -----------------------------------------------------------------------
    # Telegram (CLI alerts + command polling)
    # -----------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # -----------------------------------------------------------------------
    # Weather APIs
    # -----------------------------------------------------------------------
    OPENWEATHER_API_KEY: str = ""

    # Open-Meteo — free, no key required
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_ENSEMBLE_URL: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    # -----------------------------------------------------------------------
    # Polymarket
    # -----------------------------------------------------------------------
    POLY_PRIVATE_KEY: str = ""
    POLY_API_KEY: str = ""
    POLY_API_SECRET: str = ""
    POLY_API_PASSPHRASE: str = ""
    POLYMARKET_PROXY: str = ""

    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    CLOB_API_URL: str = "https://clob.polymarket.com"

    # -----------------------------------------------------------------------
    # Kalshi API (RSA key auth for weather markets)
    # -----------------------------------------------------------------------
    KALSHI_API_KEY_ID: Optional[str] = None
    KALSHI_PRIVATE_KEY_PATH: Optional[str] = None
    KALSHI_ENABLED: bool = True

    # -----------------------------------------------------------------------
    # AI API Keys (optional, for contract parsing)
    # -----------------------------------------------------------------------
    ANTHROPIC_API_KEY: Optional[str] = None
    AI_MODEL: str = "claude-sonnet-4-20250514"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    AI_LOG_ALL_CALLS: bool = True
    AI_DAILY_BUDGET_USD: float = 1.0

    # -----------------------------------------------------------------------
    # General trading parameters
    # -----------------------------------------------------------------------
    SIMULATION_MODE: bool = True
    PAPER_TRADING: bool = True  # Alias for CLI mode
    INITIAL_BANKROLL: float = 10000.0
    BANKROLL: float = 1000.0  # CLI bankroll (separate from web)

    # -----------------------------------------------------------------------
    # BTC 5-min trading settings
    # -----------------------------------------------------------------------
    SCAN_INTERVAL_SECONDS: int = 60
    SETTLEMENT_INTERVAL_SECONDS: int = 120
    BTC_PRICE_SOURCE: str = "coinbase"
    MIN_EDGE_THRESHOLD: float = 0.02
    MAX_ENTRY_PRICE: float = 0.55
    MAX_TRADES_PER_WINDOW: int = 1
    MAX_TOTAL_PENDING_TRADES: int = 20
    KELLY_FRACTION: float = 0.15  # Web dashboard Kelly

    # Risk management
    DAILY_LOSS_LIMIT: float = 300.0
    MAX_TRADE_SIZE: float = 75.0
    MIN_TIME_REMAINING: int = 60
    MAX_TIME_REMAINING: int = 1800

    # Indicator weights for BTC composite signal
    WEIGHT_RSI: float = 0.20
    WEIGHT_MOMENTUM: float = 0.35
    WEIGHT_VWAP: float = 0.20
    WEIGHT_SMA: float = 0.15
    WEIGHT_MARKET_SKEW: float = 0.10
    MIN_MARKET_VOLUME: float = 100.0

    # -----------------------------------------------------------------------
    # Weather trading settings (shared by CLI + web)
    # -----------------------------------------------------------------------
    WEATHER_ENABLED: bool = True
    WEATHER_SCAN_INTERVAL_SECONDS: int = 300
    WEATHER_SETTLEMENT_INTERVAL_SECONDS: int = 1800
    WEATHER_MIN_EDGE_THRESHOLD: float = 0.08
    WEATHER_MAX_ENTRY_PRICE: float = 0.70
    WEATHER_MAX_TRADE_SIZE: float = 100.0
    WEATHER_CITIES: str = "nyc,chicago,miami,los_angeles,denver"

    # CLI-specific weather settings
    SCAN_INTERVAL_SEC: int = 120
    MIN_EDGE: float = 0.08
    MAX_BET_USD: float = 25.0
    CLI_KELLY_FRACTION: float = 0.25

    # -----------------------------------------------------------------------
    # Mapbox (optional for globe visualization)
    # -----------------------------------------------------------------------
    VITE_MAPBOX_TOKEN: str = ""

    # -----------------------------------------------------------------------
    # FRED / BLS (economics — future use)
    # -----------------------------------------------------------------------
    FRED_API_KEY: str = ""
    BLS_API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()

# -----------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------
# Cities tracked (21 — merged from both projects)
# -----------------------------------------------------------------------
CITIES = {
    "nyc": {"name": "New York", "lat": 40.7128, "lon": -74.0060,
            "nws_station": "KNYC", "nws_office": "OKX", "nws_gridpoint": "OKX/33,37"},
    "chicago": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298,
                "nws_station": "KORD", "nws_office": "LOT", "nws_gridpoint": "LOT/75,72"},
    "los_angeles": {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437,
                    "nws_station": "KLAX", "nws_office": "LOX", "nws_gridpoint": "LOX/154,44"},
    "miami": {"name": "Miami", "lat": 25.7617, "lon": -80.1918,
              "nws_station": "KMIA", "nws_office": "MFL", "nws_gridpoint": "MFL/75,53"},
    "denver": {"name": "Denver", "lat": 39.7392, "lon": -104.9903,
               "nws_station": "KDEN", "nws_office": "BOU", "nws_gridpoint": "BOU/62,60"},
    "toronto": {"name": "Toronto", "lat": 43.6532, "lon": -79.3832},
    "dallas": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "seattle": {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    "london": {"name": "London", "lat": 51.5074, "lon": -0.1278},
    "paris": {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    "berlin": {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    "madrid": {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    "amsterdam": {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    "tokyo": {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    "seoul": {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    "beijing": {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
    "singapore": {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    "sydney": {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    "hong_kong": {"name": "Hong Kong", "lat": 22.3193, "lon": 114.1694},
    "dubai": {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    "buenos_aires": {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    "sao_paulo": {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333},
}
