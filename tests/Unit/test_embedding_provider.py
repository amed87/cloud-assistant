"""Verify embedding-provider request parameters and failure translation."""

from types import SimpleNamespace

import pytest
from httpx import ConnectError, TimeoutException

from app.embeddings import ollama_provider
from app.exceptions.provider import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class FailingEmbeddingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.request: dict[str, object] | None = None

    async def embed(self, **kwargs: object) -> None:
        self.request = kwargs
        raise self.error


@pytest.mark.asyncio
async def test_embed_returns_provider_embedding_and_sends_model_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEmbeddingClient:
        request: dict[str, object] | None = None

        async def embed(self, **kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            return SimpleNamespace(embeddings=[[0.1, 0.2]])

    client = FakeEmbeddingClient()
    monkeypatch.setattr(ollama_provider, "AsyncClient", lambda: client)
    provider = ollama_provider.OllamaEmbeddingProvider(model="test-model")

    embedding = await provider.embed("test text")

    assert embedding == [0.1, 0.2]
    assert client.request == {
        "model": "test-model",
        "input": "test text",
        "keep_alive": -1,
        "options": {
            "num_thread": provider.threads,
            "num_ctx": provider.ctx_size,
            "num_keep": 0,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_error", "expected_error"),
    [
        (ConnectError("connection details"), ProviderUnavailableError),
        (TimeoutException("timeout details"), ProviderTimeoutError),
        (RuntimeError("backend details"), ProviderError),
    ],
)
async def test_embed_translates_provider_errors(
    monkeypatch,
    source_error: Exception,
    expected_error: type[ProviderError],
) -> None:
    client = FailingEmbeddingClient(source_error)
    monkeypatch.setattr(ollama_provider, "AsyncClient", lambda: client)
    provider = ollama_provider.OllamaEmbeddingProvider()

    with pytest.raises(expected_error) as raised:
        await provider.embed("test text")

    assert raised.value.__cause__ is source_error
    assert str(source_error) not in raised.value.message
    assert client.request == {
        "model": provider.model,
        "input": "test text",
        "keep_alive": -1,
        "options": {
            "num_thread": provider.threads,
            "num_ctx": provider.ctx_size,
            "num_keep": 0,
        },
    }
