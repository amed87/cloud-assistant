import pytest

from app.models.message import ChatMessage
from app.models.role import Role
from app.providers.ollama_provider import OllamaProvider


@pytest.mark.asyncio
async def test_provider_streams_response() -> None:
    provider = OllamaProvider()

    messages = [
        ChatMessage(
            role=Role.USER,
            content="Erkläre Kubernetes in zwei Sätzen.",
        ),
    ]

    tokens = [
        token
        async for token in provider.stream_chat(messages)
    ]

    assert tokens
    assert all(isinstance(token, str) for token in tokens)
    assert "".join(tokens).strip()

@pytest.mark.asyncio
async def test_provider_returns_chat_response() -> None:
    provider = OllamaProvider()

    messages = [
        ChatMessage(
            role=Role.USER,
            content="Erkläre Kubernetes in zwei Sätzen.",
        ),
    ]

    response = await provider.chat(messages)

    assert isinstance(response, str)
    assert response.strip()