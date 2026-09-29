from uuid import uuid4

import pytest

from app.documents.chunker import DocumentChunker
from app.documents.models import TextDocument
from app.documents.text_importer import TextImporter
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore


@pytest.mark.asyncio
async def test_imported_document_can_be_retrieved() -> None:
    document_id = f"kubernetes-test-{uuid4()}"

    document = TextDocument(
        id=document_id,
        source="kubernetes.txt",
        content=(
            "Kubernetes ist eine Plattform zur "
            "Orchestrierung von Containern. "
            "Kubernetes automatisiert Deployment, "
            "Skalierung und Verwaltung von "
            "Container-Anwendungen."
        ),
    )

    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()

    importer = TextImporter(
        chunker=DocumentChunker(
            chunk_size=1000,
            overlap=200,
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    try:
        await importer.import_data(document)

        query_embedding = await embedding_provider.embed(
            "Wie funktioniert Kubernetes?",
        )

        results = await vector_store.search(
            embedding=query_embedding,
            limit=5,
        )

        assert results
        assert any(
            "Kubernetes" in result.content
            for result in results
        )

    finally:
        await vector_store.delete(
            [f"{document_id}-0"],
        )