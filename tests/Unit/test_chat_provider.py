"""Verify chat-provider request conversion, responses, and normalized errors."""

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from httpx import ConnectError, ReadTimeout

from app.exceptions.provider import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.models.message import ChatMessage
from app.models.role import Role
from app.providers import ollama_provider
from app.config.settings import get_settings


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat(self, **kwargs: object) -> SimpleNamespace | AsyncIterator[SimpleNamespace]:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def chunks() -> AsyncIterator[SimpleNamespace]:
                for content in ("First ", "", "second"):
                    yield SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )

            return chunks()
        return SimpleNamespace(
            message=SimpleNamespace(content="Chat response"),
        )


@pytest.mark.asyncio
async def test_chat_maps_messages_and_stream_filters_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeChatClient()
    client_options: dict[str, object] = {}

    def create_client(**kwargs: object) -> FakeChatClient:
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(
        ollama_provider,
        "AsyncClient",
        create_client,
    )
    provider = ollama_provider.OllamaProvider()
    messages = [ChatMessage(role=Role.USER, content="Question")]

    response = await provider.chat(messages)
    chunks = [chunk async for chunk in provider.stream_chat(messages)]

    expected_messages = [{"role": "user", "content": "Question"}]
    assert response == "Chat response"
    assert chunks == ["First ", "second"]
    assert client_options == {"host": get_settings().ollama_host}
    assert client.calls[0]["messages"] == expected_messages
    assert client.calls[1]["messages"] == expected_messages
    assert client.calls[1]["stream"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_error", "expected_error"),
    [
        (ConnectError("connection details"), ProviderUnavailableError),
        (ReadTimeout("timeout details"), ProviderTimeoutError),
        (RuntimeError("backend details"), ProviderError),
    ],
)
async def test_chat_translates_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    source_error: Exception,
    expected_error: type[ProviderError],
) -> None:
    class FailingChatClient:
        request: dict[str, object] | None = None

        async def chat(self, **_kwargs: object) -> None:
            self.request = _kwargs
            raise source_error

    client = FailingChatClient()
    client_options: dict[str, object] = {}

    def create_client(**kwargs: object) -> FailingChatClient:
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(
        ollama_provider,
        "AsyncClient",
        create_client,
    )
    provider = ollama_provider.OllamaProvider()

    with pytest.raises(expected_error) as raised:
        await provider.chat([ChatMessage(role=Role.USER, content="Question")])

    assert raised.value.__cause__ is source_error
    assert client.request is not None
    assert client.request["messages"] == [{"role": "user", "content": "Question"}]
    assert client_options == {"host": get_settings().ollama_host}
