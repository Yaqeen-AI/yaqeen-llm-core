from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from utils.rate_limits import AsyncSlidingWindowRateLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_GEMINI_RATE_LIMIT_ENABLED = os.getenv("GEMINI_RATE_LIMIT_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
_GEMINI_RATE_LIMIT_SAFETY = float(os.getenv("GEMINI_RATE_LIMIT_SAFETY", "0.9"))
_GEMINI_RPM_LIMIT = max(1, int(float(os.getenv("GEMINI_RPM_LIMIT", "15")) * _GEMINI_RATE_LIMIT_SAFETY))
_GEMINI_TPM_LIMIT = max(1, int(float(os.getenv("GEMINI_TPM_LIMIT", "250000")) * _GEMINI_RATE_LIMIT_SAFETY))
_GEMINI_RPD_LIMIT = max(1, int(float(os.getenv("GEMINI_RPD_LIMIT", "500")) * _GEMINI_RATE_LIMIT_SAFETY))
_GEMINI_TOKEN_CHARS_PER_TOKEN = max(1.0, float(os.getenv("GEMINI_TOKEN_CHARS_PER_TOKEN", "3.0")))
_GEMINI_RESPONSE_TOKEN_RESERVE = max(0, int(os.getenv("GEMINI_RESPONSE_TOKEN_RESERVE", "2048")))
_GEMINI_RETRY_BUFFER_SECONDS = float(os.getenv("GEMINI_RETRY_BUFFER_SECONDS", "2.0"))

_GEMINI_RPM_LIMITER = AsyncSlidingWindowRateLimiter(
    limit=_GEMINI_RPM_LIMIT,
    window_seconds=60,
    name="gemini-rpm",
    enabled=_GEMINI_RATE_LIMIT_ENABLED,
)
_GEMINI_TPM_LIMITER = AsyncSlidingWindowRateLimiter(
    limit=_GEMINI_TPM_LIMIT,
    window_seconds=60,
    name="gemini-tpm",
    enabled=_GEMINI_RATE_LIMIT_ENABLED,
)
_GEMINI_RPD_LIMITER = AsyncSlidingWindowRateLimiter(
    limit=_GEMINI_RPD_LIMIT,
    window_seconds=24 * 60 * 60,
    name="gemini-rpd",
    enabled=_GEMINI_RATE_LIMIT_ENABLED,
)


class LLMClient(Protocol):
    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        """Return a JSON object generated from the supplied messages."""


class GeminiLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.model = model or os.getenv("GEMINI_AGENT_MODEL", "gemini-3.1-flash-lite")
        self.temperature = temperature

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for agent reasoning.")
        await _acquire_gemini_budget(system, user)
        text = await asyncio.to_thread(self._generate_sync, system, user)
        return _parse_json_object(text)

    def _generate_sync(self, system: str, user: str, max_retries: int = 3) -> str:
        prompt = (
            f"{system}\n\n"
            "Return a single valid JSON object representing the answer. Do NOT include markdown blocks. Do NOT wrap in ```json.\n\n"
            f"{user}"
        )
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                from google import genai  # type: ignore

                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": self.temperature, "response_mime_type": "application/json"},
                )
                return response.text or "{}"
            except ImportError:
                import google.generativeai as genai_legacy  # type: ignore
                import time
                genai_legacy.configure(api_key=self.api_key)
                model = genai_legacy.GenerativeModel(self.model)
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "response_mime_type": "application/json",
                        },
                    )
                    return getattr(response, "text", "{}") or "{}"
                except Exception as e:
                    last_exception = e
                    retry_delay = _retry_delay_seconds(str(e))
                    time.sleep(retry_delay if retry_delay is not None else 2 ** attempt)
            except Exception as e:
                import time
                last_exception = e
                retry_delay = _retry_delay_seconds(str(e))
                time.sleep(retry_delay if retry_delay is not None else 2 ** attempt)
                
        logger.error("GeminiLLMClient failed after %d attempts. Last error: %s", max_retries, last_exception)
        return "{}"


async def _acquire_gemini_budget(system: str, user: str) -> None:
    estimated_tokens = _estimate_gemini_tokens(system, user)
    await _GEMINI_RPD_LIMITER.acquire(1)
    await _GEMINI_RPM_LIMITER.acquire(1)
    await _GEMINI_TPM_LIMITER.acquire(estimated_tokens)


def _estimate_gemini_tokens(system: str, user: str) -> int:
    prompt_chars = len(system or "") + len(user or "")
    prompt_tokens = ceil(prompt_chars / _GEMINI_TOKEN_CHARS_PER_TOKEN)
    return max(1, prompt_tokens + _GEMINI_RESPONSE_TOKEN_RESERVE)


def _retry_delay_seconds(message: str) -> float | None:
    retry_match = re.search(r"retryDelay['\"]?\s*:\s*['\"](?P<delay>\d+(?:\.\d+)?)s", message)
    if retry_match:
        return float(retry_match.group("delay")) + _GEMINI_RETRY_BUFFER_SECONDS

    please_retry_match = re.search(r"retry in (?P<delay>\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if please_retry_match:
        return float(please_retry_match.group("delay")) + _GEMINI_RETRY_BUFFER_SECONDS

    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return 60 + _GEMINI_RETRY_BUFFER_SECONDS
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
        cleaned = cleaned.removeprefix("JSON").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    
    try:
        parsed = json.loads(cleaned or "{}")
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except json.JSONDecodeError:
        logger.error("JSON decode error. Raw text: %r", text)
        return {}


class Agent(ABC):
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or GeminiLLMClient()

    async def _structured_json(self, system: str, user: str, model_type: type[T]) -> T:
        payload = await self.llm.generate_json(system, user)
        try:
            return model_type.model_validate(payload)
        except AttributeError:
            return model_type.parse_obj(payload)
        except ValidationError:
            logger.exception("Invalid structured output for %s: %s", model_type.__name__, payload)
            raise

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
