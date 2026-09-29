import pytest

from app.embeddings.ollama_provider import OllamaEmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_provider_returns_embedding() -> None:
    provider = OllamaEmbeddingProvider()

    embedding = await provider.embed("Was ist Kubernetes?")

    assert embedding
    assert all(isinstance(value, float) for value in embedding)