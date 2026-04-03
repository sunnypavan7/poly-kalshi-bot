"""Weather temperature market fetcher from Polymarket."""
import httpx
import json
import re
import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from backend.config import CITIES

logger = logging.getLogger("trading_bot")

# Map city names/variants found in market titles to our city keys
CITY_ALIASES = {
    "new york": "nyc",
    "nyc": "nyc",
    "new york city": "nyc",
    "chicago": "chicago",
    "miami": "miami",
    "los angeles": "los_angeles",
    "la": "los_angeles",
    "denver": "denver",
    "houston": "houston",
    "phoenix": "phoenix",
    "philadelphia": "philadelphia",
    "san antonio": "san_antonio",
    "san diego": "san_diego",
    "dallas": "dallas",
    "austin": "austin",
    "jacksonville": "jacksonville",
    "san francisco": "san_francisco",
    "seattle": "seattle",
    "nashville": "nashville",
    "boston": "boston",
    "atlanta": "atlanta",
    "london": "london",
    "tokyo": "tokyo",
    "sydney": "sydney",
    "mumbai": "mumbai",
    "sao paulo": "sao_paulo",
}

# Month name to number
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class WeatherMarket:
    """A weather temperature prediction market."""
    slug: str
    market_id: str
    platform: str
    title: str
    city_key: str
    city_name: str
    target_date: date
    threshold_f: float       # Temperature threshold in Fahrenheit
    metric: str              # "high" or "low"
    direction: str           # "above" or "below"
    yes_price: float         # Price of YES outcome (0-1)
    no_price: float          # Price of NO outcome (0-1)
    volume: float = 0.0
    closed: bool = False


def _parse_weather_market_title(title: str) -> Optional[dict]:
    """
    Parse a weather market title to extract city, threshold, metric, date.

    Handles patterns like:
    - "Will the high temperature in New York exceed 75 deg F on March 5?"
    - "NYC high temperature above 80 deg F on March 10, 2026"
    - "Chicago daily high over 60 deg F on March 3"
    - "Will Miami's low be above 65 deg F on March 7?"
    - "Temperature in Denver above 70 deg F on March 5, 2026"
    """
    title_lower = title.lower()

    # Must be temperature-related
    if not any(kw in title_lower for kw in ["temperature", "temp", "°f", "degrees", "high", "low"]):
        return None

    # Extract city
    city_key = None
    city_name = None
    for alias, key in sorted(CITY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in title_lower:
            city_key = key
            city_info = CITIES.get(key)
            if city_info:
                city_name = city_info["name"]
            else:
                city_name = alias.title()
            break

    if not city_key:
        return None

    # Extract threshold temperature
    temp_match = re.search(r'(\d+)\s*°?\s*f', title_lower)
    if not temp_match:
        temp_match = re.search(r'(\d+)\s*degrees', title_lower)
    if not temp_match:
        return None
    threshold_f = float(temp_match.group(1))

    # Determine metric (high vs low)
    metric = "high"  # default
    if "low" in title_lower:
        metric = "low"

    # Determine direction
    direction = "above"  # default
    if any(kw in title_lower for kw in ["below", "under", "less than", "drop below"]):
        direction = "below"

    # Extract date
    target_date = _extract_date(title_lower)
    if not target_date:
        return None

    return {
        "city_key": city_key,
        "city_name": city_name,
        "threshold_f": threshold_f,
        "metric": metric,
        "direction": direction,
        "target_date": target_date,
    }


def _extract_date(text: str) -> Optional[date]:
    """Extract a date from market title text."""
    today = date.today()

    month_names = "|".join(MONTH_MAP.keys())

    # Pattern: "March 5, 2026" or "March 5 2026" or "March 5"
    for match in re.finditer(rf'({month_names})\s+(\d{{1,2}})(?:\s*,?\s*(\d{{4}}))?', text):
        month_str = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year

        month = MONTH_MAP.get(month_str)
        if month and 1 <= day <= 31:
            try:
                return date(year, month, day)
            except ValueError:
                continue

    # Pattern: "3/5/2026" or "03/05"
    match = re.search(r'(\d{1,2})/(\d{1,2})(?:/(\d{4}))?', text)
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year
        try:
            return date(year, month, day)
        except ValueError:
            pass

    return None


async def fetch_polymarket_weather_markets(city_keys: Optional[List[str]] = None) -> List[WeatherMarket]:
    """
    Search Polymarket for weather temperature markets.
    Searches for temperature/weather events and parses their titles.
    """
    markets = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Search for weather/temperature events via tag
            for search_term in ["temperature", "weather high", "weather low"]:
                try:
                    response = await client.get(
                        "https://gamma-api.polymarket.com/events",
                        params={
                            "closed": "false",
                            "limit": 100,
                            "tag": "Weather",
                        }
                    )
                    response.raise_for_status()
                    events = response.json()

                    for event in events:
                        event_slug = event.get("slug", "")
                        for market_data in event.get("markets", []):
                            market = _parse_polymarket_weather(market_data, event_slug, city_keys)
                            if market:
                                markets.append(market)

                except Exception as e:
                    logger.debug("Weather market search for '%s' failed: %s", search_term, e)

            # Also try slug-based search for known patterns
            for slug_pattern in ["weather", "temperature", "temp-"]:
                try:
                    response = await client.get(
                        "https://gamma-api.polymarket.com/events",
                        params={
                            "closed": "false",
                            "limit": 100,
                            "slug_contains": slug_pattern,
                        }
                    )
                    response.raise_for_status()
                    events = response.json()

                    for event in events:
                        event_slug = event.get("slug", "")
                        for market_data in event.get("markets", []):
                            market = _parse_polymarket_weather(market_data, event_slug, city_keys)
                            if market and not any(m.market_id == market.market_id for m in markets):
                                markets.append(market)

                except Exception as e:
                    logger.debug("Weather slug search for '%s' failed: %s", slug_pattern, e)

    except Exception as e:
        logger.warning("Failed to fetch weather markets: %s", e)

    logger.info("Found %d weather temperature markets", len(markets))
    return markets


def _parse_polymarket_weather(
    market_data: dict,
    event_slug: str,
    city_keys: Optional[List[str]] = None,
) -> Optional[WeatherMarket]:
    """Parse a Polymarket market dict into a WeatherMarket if it's a temp market."""
    question = market_data.get("question", "") or market_data.get("groupItemTitle", "")
    if not question:
        return None

    parsed = _parse_weather_market_title(question)
    if not parsed:
        return None

    # Filter by requested cities
    if city_keys and parsed["city_key"] not in city_keys:
        return None

    # Only trade markets for dates in the future (or today)
    if parsed["target_date"] < date.today():
        return None

    # Parse prices
    outcome_prices = market_data.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except Exception:
            outcome_prices = []

    if not outcome_prices or len(outcome_prices) < 2:
        return None

    try:
        yes_price = float(outcome_prices[0])
        no_price = float(outcome_prices[1])
    except (ValueError, IndexError):
        return None

    # Skip resolved markets
    if market_data.get("closed", False):
        return None
    if yes_price > 0.98 or yes_price < 0.02:
        return None

    volume = float(market_data.get("volume", 0) or 0)

    return WeatherMarket(
        slug=event_slug,
        market_id=str(market_data.get("id", "")),
        platform="polymarket",
        title=question,
        city_key=parsed["city_key"],
        city_name=parsed["city_name"],
        target_date=parsed["target_date"],
        threshold_f=parsed["threshold_f"],
        metric=parsed["metric"],
        direction=parsed["direction"],
        yes_price=yes_price,
        no_price=no_price,
        volume=volume,
    )
