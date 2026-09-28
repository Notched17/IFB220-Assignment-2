"""
Thin wrapper around the IFB220 Developer API Portal endpoints.

Responsibilities:
  * one place that knows how to call chat completions (gpt-4.1-mini) and
    embeddings (text-embedding-3-small; the brief names Ada-002, but the
    portal returns 404 for that deployment)
  * retries with backoff for transient failures (timeouts, 429s, 5xxs)
  * translates SDK-level exceptions into the small set of exceptions the
    rest of the app needs to handle (ApiTimeoutError, ApiRateLimitError,
    ApiError), so callers never need to know which HTTP client is in use
  * records raw usage (tokens) for every call so src/usage_monitor.py can
    build accurate totals

This module does NOT know anything about guardrails or topics -- it is
pure "talk to the model" plumbing, which keeps it easy to unit test with
mocks (see tests/test_pipeline_mocked.py) and easy to swap providers later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AzureOpenAI,
    RateLimitError,
)

from src.config import Settings


class ApiError(Exception):
    """Base class for all API-facing errors raised by this client."""


class ApiTimeoutError(ApiError):
    pass


class ApiRateLimitError(ApiError):
    pass


class ApiContentFilterError(ApiError):
    """The portal's own (Azure) content filter blocked the request. This is
    not an outage, so callers treat it as a refusal, not an error."""


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class EmbeddingResult:
    vector: list[float]
    total_tokens: int


class ApiClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AzureOpenAI(
            # Real bug found via live testing: azure_endpoint=".../ifb220/openai/" made the SDK
            # request ".../ifb220/openai/openai/deployments/..." (404); base_url is used as-is.
            base_url=settings.base_url,
            api_key=settings.api_key.strip(),
            api_version=settings.api_version,
            timeout=settings.request_timeout_s,
            max_retries=0,  # retries are handled (and counted) by _with_retries below
        )

    # ------------------------------------------------------------------
    def chat_complete(
        self, messages: list[dict], *, temperature: float = 0.4, max_tokens: int | None = None
    ) -> ChatResult:
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        response = self._with_retries(
            lambda: self._client.chat.completions.create(
                model=self._settings.chat_deployment,
                messages=messages,
                temperature=temperature,
                **extra,
            )
        )
        choice = response.choices[0]
        usage = response.usage
        return ChatResult(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def embed(self, text: str) -> EmbeddingResult:
        response = self._with_retries(
            lambda: self._client.embeddings.create(
                model=self._settings.embedding_deployment,
                input=text,
            )
        )
        usage = response.usage
        return EmbeddingResult(
            vector=response.data[0].embedding,
            total_tokens=usage.total_tokens if usage else 0,
        )

    # ------------------------------------------------------------------
    def _with_retries(self, call):
        attempts = max(1, self._settings.max_retries)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return call()
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                # 5xx is worth retrying, 4xx (other than 429, handled above) is not.
                if exc.status_code == 400 and "content_filter" in str(exc.message):
                    raise ApiContentFilterError(f"blocked by provider content filter: {exc.message}") from exc
                if not 500 <= exc.status_code < 600:
                    raise ApiError(f"API returned {exc.status_code}: {exc.message}") from exc
                last_exc = exc
            if attempt < attempts:
                self._sleep_backoff(attempt)

        if isinstance(last_exc, RateLimitError):
            raise ApiRateLimitError(f"rate limited (429) after {attempts} attempts") from last_exc
        if isinstance(last_exc, APIStatusError):
            raise ApiError(f"API returned {last_exc.status_code} after {attempts} attempts") from last_exc
        raise ApiTimeoutError(f"{last_exc} (after {attempts} attempts)") from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min(2 ** attempt * 0.5, 8))
