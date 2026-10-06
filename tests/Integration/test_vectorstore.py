import pytest

from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore
from app.vectorstores.models import VectorDocument


@pytest.mark.asyncio
async def test_vectorstore_adds_and_finds_document(tmp_path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", "vectorstore-test")

    embedding_provider = OllamaEmbeddingProvider()
    store = ChromaVectorStore()

    content = "Kubernetes orchestriert Container."
    embedding = await embedding_provider.embed(content)

    await store.add(
        [
            VectorDocument(
                id="test-vectorstore-1",
                content=content,
                embedding=embedding,
                metadata={"source": "test", "intent": "general"},
            )
        ]
    )

    results = await store.search(embedding)

    assert results
    assert any(result.content == content for result in results)
    store.collection.delete(ids=["test-vectorstore-1"])