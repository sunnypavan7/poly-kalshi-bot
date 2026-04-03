"""
Polymarket Integration — unified from both projects.
Gamma API for market discovery (no auth).
CLOB API for pricing and trading (requires auth).
"""

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

WEATHER_KEYWORDS = [
    "temperature", "degrees", "°F", "°C", "hot", "cold", "warm", "cool",
    "heat", "freeze", "snow", "rain", "weather", "celsius", "fahrenheit",
    "high of", "low of", "above", "below", "exceed", "reach",
]

CITY_KEYWORDS = [
    "new york", "nyc", "chicago", "los angeles", "la", "miami",
    "toronto", "dallas", "seattle", "london", "paris", "berlin",
    "madrid", "amsterdam", "tokyo", "seoul", "beijing", "singapore",
    "sydney", "hong kong", "dubai", "buenos aires", "sao paulo", "são paulo",
    "denver", "atlanta", "philadelphia", "ankara",
]


@dataclass
class WeatherMarket:
    """A Polymarket weather market with parsed parameters."""
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    best_ask_yes: float
    best_ask_no: float
    city: str
    threshold_f: Optional[float]
    threshold_c: Optional[float]
    direction: str  # "above" or "below"
    end_date: str
    active: bool


def _make_session() -> requests.Session:
    """Create a requests session with optional SOCKS5 proxy."""
    session = requests.Session()
    proxy = settings.POLYMARKET_PROXY
    if proxy and proxy != "direct":
        session.proxies.update({"http": proxy, "https": proxy})
    return session


# ---------------------------------------------------------------------------
# Gamma API — market discovery (no auth)
# ---------------------------------------------------------------------------

def fetch_weather_markets() -> list[WeatherMarket]:
    """Discover active weather markets from Polymarket Gamma API."""
    session = _make_session()
    markets = []
    try:
        resp = session.get(
            f"{settings.GAMMA_API_URL}/markets",
            params={"active": "true", "closed": "false", "tag_slug": "weather", "limit": 200},
            timeout=20,
        )
        resp.raise_for_status()
        raw_markets = resp.json()

        for keyword in ["temperature", "degrees", "weather", "heat"]:
            try:
                resp2 = session.get(
                    f"{settings.GAMMA_API_URL}/markets",
                    params={"active": "true", "closed": "false", "limit": 100},
                    timeout=20,
                )
                resp2.raise_for_status()
                raw_markets.extend(resp2.json())
            except Exception:
                pass

        seen = set()
        for m in raw_markets:
            cid = m.get("condition_id") or m.get("conditionId", "")
            if cid in seen:
                continue
            seen.add(cid)
            question = m.get("question", "")
            if not any(kw in question.lower() for kw in WEATHER_KEYWORDS):
                continue
            parsed = _parse_weather_question(question)
            if not parsed:
                continue

            tokens = m.get("tokens", [])
            token_yes, token_no, ask_yes, ask_no = "", "", 0.5, 0.5
            for tok in tokens:
                outcome = tok.get("outcome", "").lower()
                if outcome == "yes":
                    token_yes = tok.get("token_id", "")
                    ask_yes = float(tok.get("price", 0.5))
                elif outcome == "no":
                    token_no = tok.get("token_id", "")
                    ask_no = float(tok.get("price", 0.5))

            markets.append(WeatherMarket(
                condition_id=cid, question=question,
                token_id_yes=token_yes, token_id_no=token_no,
                best_ask_yes=ask_yes, best_ask_no=ask_no,
                city=parsed["city"],
                threshold_f=parsed.get("threshold_f"), threshold_c=parsed.get("threshold_c"),
                direction=parsed["direction"],
                end_date=m.get("end_date_iso", ""), active=True,
            ))
        logger.info("Found %d active weather markets", len(markets))
    except Exception as e:
        logger.error("Gamma API market discovery failed: %s", e)
    return markets


def _parse_weather_question(question: str) -> Optional[dict]:
    """Parse city, threshold, and direction from a market question."""
    q = question.lower()
    city = None
    for kw in CITY_KEYWORDS:
        if kw in q:
            city = kw
            break
    if not city:
        return None

    direction = "above"
    if any(w in q for w in ["below", "under", "less than", "colder", "low of"]):
        direction = "below"

    threshold_f, threshold_c = None, None
    match_f = re.search(r"(\d+\.?\d*)\s*°?\s*[fF]", question)
    match_deg = re.search(r"(\d+\.?\d*)\s*degrees", question, re.IGNORECASE)
    match_c = re.search(r"(\d+\.?\d*)\s*°?\s*[cC]", question)

    if match_f:
        threshold_f = float(match_f.group(1))
        threshold_c = (threshold_f - 32) * 5 / 9
    elif match_c:
        threshold_c = float(match_c.group(1))
        threshold_f = threshold_c * 9 / 5 + 32
    elif match_deg:
        threshold_f = float(match_deg.group(1))
        threshold_c = (threshold_f - 32) * 5 / 9

    if threshold_c is None:
        return None
    return {"city": city, "direction": direction, "threshold_f": threshold_f, "threshold_c": threshold_c}


# ---------------------------------------------------------------------------
# CLOB API — order book pricing and trading (auth required)
# ---------------------------------------------------------------------------

def _build_hmac_signature(method: str, path: str, body: str, timestamp: str) -> str:
    message = timestamp + method.upper() + path + body
    return hmac.new(
        settings.POLY_API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _clob_headers(method: str, path: str, body: str = "") -> dict:
    ts = str(int(time.time()))
    sig = _build_hmac_signature(method, path, body, ts)
    return {
        "POLY_API_KEY": settings.POLY_API_KEY,
        "POLY_API_SIGNATURE": sig,
        "POLY_API_TIMESTAMP": ts,
        "POLY_API_PASSPHRASE": settings.POLY_API_PASSPHRASE,
        "Content-Type": "application/json",
    }


def get_order_book(token_id: str) -> Optional[dict]:
    if not settings.POLY_API_KEY:
        return None
    session = _make_session()
    path = f"/book?token_id={token_id}"
    try:
        resp = session.get(f"{settings.CLOB_API_URL}{path}", headers=_clob_headers("GET", path), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("CLOB order book fetch failed: %s", e)
        return None


def get_best_ask(token_id: str) -> Optional[float]:
    book = get_order_book(token_id)
    if not book:
        return None
    asks = book.get("asks", [])
    return float(min(asks, key=lambda x: float(x["price"]))["price"]) if asks else None


def place_limit_order(token_id: str, side: str, price: float, size: float) -> Optional[dict]:
    """Place a limit order on CLOB API. Returns order response or None."""
    if settings.PAPER_TRADING:
        logger.info("[PAPER] Would place %s order: token=%s price=%.4f size=%.2f", side, token_id, price, size)
        return {"paper": True, "side": side, "price": price, "size": size, "token_id": token_id}

    if not settings.POLY_API_KEY:
        logger.error("Cannot place order: CLOB API credentials not configured")
        return None

    session = _make_session()
    path = "/order"
    body = json.dumps({"tokenID": token_id, "side": side.upper(), "price": str(price), "size": str(size), "type": "GTC"})
    try:
        resp = session.post(f"{settings.CLOB_API_URL}{path}", headers=_clob_headers("POST", path, body), data=body, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        logger.info("Order placed: %s", result)
        return result
    except Exception as e:
        logger.error("Order placement failed: %s", e)
        return None
