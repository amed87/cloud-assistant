from app.documents.importer import DocumentImporter
from app.documents.models import FAQDocument

from app.embeddings.base import EmbeddingProvider
from app.vectorstores.base import VectorStore
from app.vectorstores.models import VectorDocument


class FAQImporter(DocumentImporter):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        super().__init__(
            embedding_provider,
            vector_store,
        )

    async def import_data(
        self,
        faq: FAQDocument,
    ) -> None:
        vector_documents: list[VectorDocument] = []

        for entry in faq.entries:
            embedding = await self.embedding_provider.embed(
                entry.question,
            )

            vector_documents.append(
                VectorDocument(
                    id=entry.id,
                    content=entry.question,
                    embedding=embedding,
                    metadata={
                        "source": faq.source,
                        "chunk_index": 0,
                        "answer": entry.answer,
                        "keywords": entry.keywords,
                    },
                )
            )

        await self.vector_store.add(vector_documents)