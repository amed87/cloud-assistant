import asyncio

from app.documents.importer import DocumentImporter
from app.documents.chunker import DocumentChunker
from app.documents.models import Document
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.models import VectorDocument


class MockVectorStore:

    def __init__(self):
        self.documents: list[VectorDocument] = []

    async def add(
        self,
        documents: list[VectorDocument],
    ) -> None:

        self.documents.extend(documents)


async def main():

    document = Document(
        id="test",
        source="test.txt",
        content=(
            "Kubernetes ist eine Plattform zur "
            "Orchestrierung von Containern."
        ),
    )

    chunker = DocumentChunker(
        chunk_size=1000,
        overlap=200,
    )

    embedding_provider = OllamaEmbeddingProvider()

    vector_store = MockVectorStore()

    importer = DocumentImporter(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    await importer.import_document(document)

    print(
        f"VectorDocuments: "
        f"{len(vector_store.documents)}"
    )

    for vector_document in vector_store.documents:

        print()
        print("ID:", vector_document.id)
        print("Content:", vector_document.content)
        print(
            "Embedding dimension:",
            len(vector_document.embedding),
        )
        print(
            "Metadata:",
            vector_document.metadata,
        )


asyncio.run(main())