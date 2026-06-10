from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


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
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                import time
                last_exception = e
                time.sleep(2 ** attempt)
                
        logger.error("GeminiLLMClient failed after %d attempts. Last error: %s", max_retries, last_exception)
        return "{}"


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

