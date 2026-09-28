"""Retry / error-translation behaviour of ApiClient, with the OpenAI SDK
call replaced by a fake (no network)."""

from unittest.mock import MagicMock, patch

import pytest
from openai import APIStatusError, RateLimitError

from src.api_client import ApiClient, ApiContentFilterError, ApiError, ApiRateLimitError
from src.config import Settings


def status_error(code: int):
    """Build an SDK status error without a real HTTP response object (the
    response type differs between openai SDK versions)."""
    cls = RateLimitError if code == 429 else APIStatusError
    err = cls.__new__(cls)
    Exception.__init__(err, f"HTTP {code}")
    err.status_code = code
    err.message = f"HTTP {code}"
    return err


def ok_response():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "hello"
    resp.usage.prompt_tokens, resp.usage.completion_tokens = 5, 2
    return resp


@pytest.fixture
def client():
    with patch.object(ApiClient, "_sleep_backoff") as sleep:
        c = ApiClient(Settings(api_key="  test-key \n", base_url="https://example.invalid/", max_retries=3))
        c._fake_create = MagicMock()
        c._client = MagicMock()
        c._client.chat.completions.create = c._fake_create
        c._sleep = sleep
        yield c


def test_api_key_is_stripped_when_client_is_built():
    c = ApiClient(Settings(api_key="  test-key \r\n", base_url="https://example.invalid/"))
    assert c._client.api_key == "test-key"


def test_429_then_success_is_retried(client):
    client._fake_create.side_effect = [status_error(429), status_error(429), ok_response()]
    result = client.chat_complete([{"role": "user", "content": "hi"}])
    assert result.content == "hello"
    assert client._fake_create.call_count == 3
    assert client._sleep.call_count == 2  # backoff between attempts only


def test_429_on_every_attempt_raises_rate_limit_error(client):
    client._fake_create.side_effect = [status_error(429)] * 3
    with pytest.raises(ApiRateLimitError):
        client.chat_complete([{"role": "user", "content": "hi"}])
    assert client._fake_create.call_count == 3
    assert client._sleep.call_count == 2  # no pointless sleep after the last attempt


def test_5xx_is_retried_then_raises_api_error(client):
    client._fake_create.side_effect = [status_error(503), status_error(500), status_error(502)]
    with pytest.raises(ApiError, match="502 after 3 attempts"):
        client.chat_complete([{"role": "user", "content": "hi"}])
    assert client._fake_create.call_count == 3


def test_5xx_then_success(client):
    client._fake_create.side_effect = [status_error(503), ok_response()]
    assert client.chat_complete([{"role": "user", "content": "hi"}]).content == "hello"


def test_4xx_is_not_retried(client):
    client._fake_create.side_effect = [status_error(401)]
    with pytest.raises(ApiError, match="401"):
        client.chat_complete([{"role": "user", "content": "hi"}])
    assert client._fake_create.call_count == 1


def test_content_filter_400_is_reported_as_its_own_error_and_not_retried(client):
    err = status_error(400)
    err.message = "Error code: 400 - {'error': {'code': 'content_filter'}}"
    client._fake_create.side_effect = [err]
    with pytest.raises(ApiContentFilterError):
        client.chat_complete([{"role": "user", "content": "hi"}])
    assert client._fake_create.call_count == 1
