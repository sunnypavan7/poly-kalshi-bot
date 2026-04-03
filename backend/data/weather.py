"""
Unified weather data module.
- Open-Meteo ECMWF 51-member ensemble (primary, P1) + GFS 31-member (P2)
- OpenWeatherMap 5-day fallback (P1)
- NWS observed temperatures for settlement (P2)
- Historical archive for backtesting (P1)
- Probability via norm.cdf (P1) AND ensemble member counting (P2)
"""

import httpx
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import requests
from scipy.stats import norm

from backend.config import settings, CITIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Forecast:
    """Temperature forecast for a city (CLI mode)."""
    city: str
    date: str
    mean_high: float        # Celsius
    std_high: float
    mean_low: float
    std_low: float
    source: str


@dataclass
class EnsembleForecast:
    """Ensemble weather forecast with per-member data (web dashboard mode, Fahrenheit)."""
    city_key: str
    city_name: str
    target_date: date
    member_highs: List[float]  # Daily max temps (F) per ensemble member
    member_lows: List[float]   # Daily min temps (F) per ensemble member
    mean_high: float = 0.0
    std_high: float = 0.0
    mean_low: float = 0.0
    std_low: float = 0.0
    num_members: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.member_highs:
            self.mean_high = statistics.mean(self.member_highs)
            self.std_high = statistics.stdev(self.member_highs) if len(self.member_highs) > 1 else 0.0
            self.num_members = len(self.member_highs)
        if self.member_lows:
            self.mean_low = statistics.mean(self.member_lows)
            self.std_low = statistics.stdev(self.member_lows) if len(self.member_lows) > 1 else 0.0

    def probability_high_above(self, threshold_f: float) -> float:
        if not self.member_highs:
            return 0.5
        return sum(1 for h in self.member_highs if h > threshold_f) / len(self.member_highs)

    def probability_high_below(self, threshold_f: float) -> float:
        return 1.0 - self.probability_high_above(threshold_f)

    def probability_low_above(self, threshold_f: float) -> float:
        if not self.member_lows:
            return 0.5
        return sum(1 for l in self.member_lows if l > threshold_f) / len(self.member_lows)

    def probability_low_below(self, threshold_f: float) -> float:
        return 1.0 - self.probability_low_above(threshold_f)

    @property
    def ensemble_agreement(self) -> float:
        if not self.member_highs:
            return 0.5
        median = statistics.median(self.member_highs)
        above = sum(1 for h in self.member_highs if h > median)
        frac = above / len(self.member_highs)
        return max(frac, 1 - frac)


# ---------------------------------------------------------------------------
# Cache for ensemble forecasts
# ---------------------------------------------------------------------------
_forecast_cache: Dict[str, tuple] = {}
_CACHE_TTL = 900  # 15 minutes


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


# ---------------------------------------------------------------------------
# Open-Meteo ensemble forecast (CLI mode — Celsius, ECMWF 51 members)
# ---------------------------------------------------------------------------

def fetch_open_meteo_forecast(lat: float, lon: float, city_name: str) -> Optional[Forecast]:
    """Fetch today's forecast from Open-Meteo deterministic model."""
    try:
        today = date.today().isoformat()
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "models": "ecmwf_ifs04", "timezone": "auto",
            "start_date": today, "end_date": today,
        }
        resp = requests.get(settings.OPEN_METEO_FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        if not highs or not lows:
            return None
        return Forecast(
            city=city_name, date=today,
            mean_high=float(highs[0]), std_high=2.0,
            mean_low=float(lows[0]), std_low=2.0,
            source="open_meteo",
        )
    except Exception as e:
        logger.error("Open-Meteo forecast failed for %s: %s", city_name, e)
        return None


def fetch_open_meteo_ensemble_cli(lat: float, lon: float, city_name: str) -> Optional[Forecast]:
    """Fetch today's forecast from Open-Meteo with ECMWF 51-member ensemble (Celsius)."""
    try:
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m", "models": "ecmwf_ifs04",
            "timezone": "auto", "start_date": today, "end_date": tomorrow,
        }
        resp = requests.get(settings.OPEN_METEO_ENSEMBLE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        members = [v for k, v in hourly.items() if k.startswith("temperature_2m")]
        if not members:
            return fetch_open_meteo_forecast(lat, lon, city_name)
        arr = np.array(members, dtype=float)
        daily_maxes = np.nanmax(arr, axis=1)
        daily_mins = np.nanmin(arr, axis=1)
        return Forecast(
            city=city_name, date=today,
            mean_high=float(np.nanmean(daily_maxes)),
            std_high=max(float(np.nanstd(daily_maxes)), 0.5),
            mean_low=float(np.nanmean(daily_mins)),
            std_low=max(float(np.nanstd(daily_mins)), 0.5),
            source="open_meteo_ensemble",
        )
    except Exception:
        return fetch_open_meteo_forecast(lat, lon, city_name)


def fetch_owm_forecast(lat: float, lon: float, city_name: str) -> Optional[Forecast]:
    """Fallback: OpenWeatherMap 5-day/3-hour forecast for today only."""
    if not settings.OPENWEATHER_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": settings.OPENWEATHER_API_KEY, "units": "metric"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        today_str = date.today().isoformat()
        today_temps = [e["main"]["temp"] for e in data.get("list", []) if e.get("dt_txt", "").startswith(today_str)]
        if not today_temps:
            return None
        arr = np.array(today_temps)
        return Forecast(
            city=city_name, date=today_str,
            mean_high=float(np.max(arr)), std_high=max(float(np.std(arr)), 1.0),
            mean_low=float(np.min(arr)), std_low=max(float(np.std(arr)), 1.0),
            source="owm",
        )
    except Exception as e:
        logger.error("OWM forecast failed for %s: %s", city_name, e)
        return None


def get_forecast(lat: float, lon: float, city_name: str) -> Optional[Forecast]:
    """Get the best available forecast for a city (CLI mode, Celsius)."""
    forecast = fetch_open_meteo_ensemble_cli(lat, lon, city_name)
    if forecast:
        return forecast
    return fetch_owm_forecast(lat, lon, city_name)


# ---------------------------------------------------------------------------
# Probability calculations — ALWAYS use norm.cdf, NEVER norm.pdf
# ---------------------------------------------------------------------------

def prob_above(threshold: float, mean: float, std: float) -> float:
    """Probability that temperature exceeds threshold."""
    if std <= 0:
        return 1.0 if mean > threshold else 0.0
    return float(1.0 - norm.cdf(threshold, loc=mean, scale=std))


def prob_below(threshold: float, mean: float, std: float) -> float:
    """Probability that temperature is below threshold."""
    if std <= 0:
        return 1.0 if mean < threshold else 0.0
    return float(norm.cdf(threshold, loc=mean, scale=std))


# ---------------------------------------------------------------------------
# Ensemble forecast for web dashboard (Fahrenheit, GFS 31-member + ECMWF)
# ---------------------------------------------------------------------------

async def fetch_ensemble_forecast(city_key: str, target_date: Optional[date] = None) -> Optional[EnsembleForecast]:
    """Fetch ensemble forecast from Open-Meteo (free, GFS 31-member). Returns Fahrenheit."""
    if city_key not in CITIES:
        return None
    if target_date is None:
        target_date = date.today()

    cache_key = f"{city_key}_{target_date.isoformat()}"
    now = time.time()
    if cache_key in _forecast_cache:
        cached_time, cached_forecast = _forecast_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_forecast

    city = CITIES[city_key]
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            params = {
                "latitude": city["lat"], "longitude": city["lon"],
                "daily": "temperature_2m_max,temperature_2m_min",
                "temperature_unit": "fahrenheit",
                "start_date": target_date.isoformat(), "end_date": target_date.isoformat(),
                "models": "gfs_seamless",
            }
            response = await client.get(settings.OPEN_METEO_ENSEMBLE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            daily = data.get("daily", {})

            member_highs, member_lows = [], []
            for key, values in daily.items():
                if not isinstance(values, list) or not values or values[0] is None:
                    continue
                if "temperature_2m_max" in key:
                    member_highs.append(float(values[0]))
                elif "temperature_2m_min" in key:
                    member_lows.append(float(values[0]))

            if not member_highs:
                return None

            forecast = EnsembleForecast(
                city_key=city_key, city_name=city["name"],
                target_date=target_date,
                member_highs=member_highs, member_lows=member_lows,
            )
            _forecast_cache[cache_key] = (now, forecast)
            return forecast
    except Exception as e:
        logger.warning("Failed to fetch ensemble forecast for %s: %s", city_key, e)
        return None


# ---------------------------------------------------------------------------
# NWS observed temperatures (settlement source for Kalshi)
# ---------------------------------------------------------------------------

async def fetch_nws_observed_temperature(city_key: str, target_date: Optional[date] = None) -> Optional[Dict[str, float]]:
    """Fetch observed temperature from NWS API. Returns dict with 'high'/'low' in Fahrenheit."""
    city = CITIES.get(city_key)
    if not city or "nws_station" not in city:
        return None
    if target_date is None:
        target_date = date.today()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            station = city["nws_station"]
            start = datetime.combine(target_date, datetime.min.time()).isoformat() + "Z"
            end = datetime.combine(target_date + timedelta(days=1), datetime.min.time()).isoformat() + "Z"
            response = await client.get(
                f"https://api.weather.gov/stations/{station}/observations",
                params={"start": start, "end": end},
                headers={"User-Agent": "(trading-bot, contact@example.com)"},
            )
            response.raise_for_status()
            features = response.json().get("features", [])
            temps = []
            for obs in features:
                temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
                if temp_c is not None:
                    temps.append(_celsius_to_fahrenheit(temp_c))
            if not temps:
                return None
            return {"high": max(temps), "low": min(temps)}
    except Exception as e:
        logger.warning("Failed to fetch NWS observations for %s: %s", city_key, e)
        return None


# ---------------------------------------------------------------------------
# Historical data for backtesting (Open-Meteo archive — free, back to 1940)
# ---------------------------------------------------------------------------

def fetch_historical_temps(lat: float, lon: float, start_date: str, end_date: str) -> Optional[dict]:
    """Fetch daily historical temperature data from Open-Meteo archive."""
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto", "start_date": start_date, "end_date": end_date,
        }
        resp = requests.get(settings.OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return None
        return {
            "dates": dates,
            "highs": daily.get("temperature_2m_max", []),
            "lows": daily.get("temperature_2m_min", []),
        }
    except Exception as e:
        logger.error("Historical data fetch failed: %s", e)
        return None
