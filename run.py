#!/usr/bin/env python3
"""Run the trading bot backend server."""
import logging
import os
import uvicorn
from backend.models.database import init_db

logger = logging.getLogger("trading_bot")

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()

    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on http://0.0.0.0:%d", port)
    logger.info("API docs available at http://localhost:%d/docs", port)

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RAILWAY_ENVIRONMENT") is None
    )
