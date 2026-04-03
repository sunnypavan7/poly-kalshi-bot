"""Abstract base class for AI providers used in market analysis."""
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("trading_bot")


class AIProvider(ABC):
    """Base class for AI-powered market analysis providers."""

    @abstractmethod
    async def analyze_market(self, prompt: str) -> Optional[str]:
        """Send a prompt to the AI provider and return the response text."""
        ...

    @abstractmethod
    async def classify_signal(self, market_data: dict) -> Optional[dict]:
        """
        Classify a trading signal using AI.

        Returns dict with keys:
            - direction: "up"/"down" or "yes"/"no"
            - confidence: float 0-1
            - reasoning: str
        """
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the AI provider."""
        ...
