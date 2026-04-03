"""Groq AI provider for fast signal classification."""
import logging
from typing import Optional

from backend.ai.base import AIProvider
from backend.config import settings

logger = logging.getLogger("trading_bot")


class GroqProvider(AIProvider):
    """Groq LLaMA for fast signal classification."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")

        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.groq.com/openai/v1",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            return self._client
        except Exception as e:
            raise RuntimeError(f"Failed to create Groq client: {e}")

    async def analyze_market(self, prompt: str) -> Optional[str]:
        """Send a prompt to Groq and return the response."""
        try:
            client = self._get_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Groq API error: %s", e)
            return None

    async def classify_signal(self, market_data: dict) -> Optional[dict]:
        """Use Groq for fast signal classification."""
        prompt = (
            f"Trading signal classification. Respond with JSON only.\n\n"
            f"Market: {market_data.get('title', 'Unknown')}\n"
            f"YES price: {market_data.get('yes_price', 0.5):.2%}\n"
            f"Category: {market_data.get('category', 'unknown')}\n\n"
            f"{{\"direction\": \"yes\" or \"no\", \"confidence\": 0.0-1.0, "
            f"\"reasoning\": \"brief\"}}"
        )

        response = await self.analyze_market(prompt)
        if not response:
            return None

        try:
            import json
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse Groq response as JSON")
            return None

    def provider_name(self) -> str:
        return "groq"
