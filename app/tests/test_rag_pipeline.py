import asyncio

from app.documents.chunker import DocumentChunker
from app.documents.importer import DocumentImporter
from app.documents.models import Document
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore


async def main():

    document = Document(
        id="kubernetes-test",
        source="kubernetes.txt",
        content=(
            "Kubernetes ist eine Plattform zur "
            "Orchestrierung von Containern. "
            "Kubernetes automatisiert Deployment, "
            "Skalierung und Verwaltung von "
            "Container-Anwendungen."
        ),
    )

    chunker = DocumentChunker(
        chunk_size=1000,
        overlap=200,
    )

    embedding_provider = OllamaEmbeddingProvider()

    vector_store = ChromaVectorStore()

    importer = DocumentImporter(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    await importer.import_document(
        document,
    )

    print("Dokument erfolgreich importiert.")

    query = (
        "Wie wird ein Schokoladenkuchen gebacken?"
    )

    query_embedding = await embedding_provider.embed(
        query,
    )

    results = await vector_store.search(
        embedding=query_embedding,
        limit=5,
    )

    print()
    print(f"Treffer: {len(results)}")

    for result in results:

        print()
        print("ID:", result.id)
        print("Score:", result.score)
        print("Content:", result.content)
        print("Metadata:", result.metadata)


asyncio.run(main())