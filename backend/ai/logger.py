"""AI interaction logger — persists AI calls to database for debugging."""
import logging
from datetime import datetime, timezone
from typing import Optional

from backend.models.database import SessionLocal, AILog

logger = logging.getLogger("trading_bot")


async def log_ai_interaction(
    provider: str,
    prompt: str,
    response: Optional[str],
    market_ticker: Optional[str] = None,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Persist an AI interaction to the database."""
    db = SessionLocal()
    try:
        log_entry = AILog(
            provider=provider,
            prompt=prompt[:2000],  # Truncate long prompts
            response=response[:2000] if response else None,
            market_ticker=market_ticker,
            latency_ms=latency_ms,
            error=error,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.warning("Failed to log AI interaction: %s", e)
        db.rollback()
    finally:
        db.close()
