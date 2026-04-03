"""Claude AI provider for market question parsing and analysis."""
import logging
from typing import Optional

from backend.ai.base import AIProvider
from backend.config import settings

logger = logging.getLogger("trading_bot")


class ClaudeProvider(AIProvider):
    """Claude claude-sonnet-4-20250514 for market question parsing and signal analysis."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            return self._client
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    async def analyze_market(self, prompt: str) -> Optional[str]:
        """Send a prompt to Claude and return the response."""
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=settings.AI_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Claude API error: %s", e)
            return None

    async def classify_signal(self, market_data: dict) -> Optional[dict]:
        """Use Claude to classify a trading signal."""
        prompt = (
            f"Analyze this prediction market and provide a trading recommendation.\n\n"
            f"Market: {market_data.get('title', 'Unknown')}\n"
            f"Current YES price: {market_data.get('yes_price', 0.5):.2%}\n"
            f"Category: {market_data.get('category', 'unknown')}\n\n"
            f"Respond with JSON only: {{\"direction\": \"yes\" or \"no\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}}"
        )

        response = await self.analyze_market(prompt)
        if not response:
            return None

        try:
            import json
            # Strip markdown code fences if present
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse Claude response as JSON")
            return None

    def provider_name(self) -> str:
        return "claude"
