"""Check persistent vector-store insertion and nearest-neighbor retrieval."""

import pytest

from app.config.settings import get_settings
from app.vectorstores.chroma import ChromaVectorStore
from app.vectorstores.models import VectorDocument


@pytest.mark.asyncio
async def test_vectorstore_adds_and_finds_document(tmp_path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", "vectorstore-test")

    store = ChromaVectorStore()

    content = "Kubernetes orchestriert Container."
    embedding = [0.1, 0.2, 0.3]

    await store.add(
        [
            VectorDocument(
                id="test-vectorstore-1",
                content=content,
                embedding=embedding,
                metadata={"source": "test"},
            )
        ]
    )

    results = await store.search(embedding)

    assert results
    assert any(result.content == content for result in results)
    store.collection.delete(ids=["test-vectorstore-1"])