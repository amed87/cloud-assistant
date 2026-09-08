from app.documents.chunker import DocumentChunker
from app.documents.importer import DocumentImporter
from app.documents.models import TextDocument

from app.embeddings.base import EmbeddingProvider

from app.vectorstores.base import VectorStore
from app.vectorstores.models import VectorDocument


class TextImporter(DocumentImporter):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunker: DocumentChunker,
    ) -> None:
        super().__init__(
            embedding_provider,
            vector_store,
        )
        self.chunker = chunker

    async def import_data(
        self,
        document: TextDocument,
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