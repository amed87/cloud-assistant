from app.documents.chunker import DocumentChunker
from app.documents.models import Document

from app.embeddings.base import EmbeddingProvider

from app.vectorstores.base import VectorStore
from app.vectorstores.models import VectorDocument


class DocumentImporter:

    def __init__(
        self,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def import_document(
        self,
        document: Document,
    ) -> None:

        chunks = self.chunker.chunk(document)

        vector_documents: list[VectorDocument] = []

        for chunk in chunks:

            embedding = await self.embedding_provider.embed(
                chunk.content,
            )

            vector_documents.append(
                VectorDocument(
                    id=chunk.id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata={
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        await self.vector_store.add(
            vector_documents,
        )