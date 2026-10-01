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

    async def embed(self, **kwargs):
        raise self.error


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