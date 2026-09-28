from pathlib import PurePosixPath

from app.documents.importer import DocumentImporter
from app.documents.models import FAQDocument

from app.embeddings.base import EmbeddingProvider
from app.vectorstores.base import VectorStore
from app.vectorstores.models import VectorDocument


class FAQImporter(DocumentImporter):
    @staticmethod
    def _normalize_source(source: str) -> str:
        return PurePosixPath(source.replace("\\", "/")).as_posix().casefold()

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
        database = self.vector_store.get()
        existing_faq_ids = {
            entry_id
            for entry_id, metadata in zip(database["ids"], database["metadatas"])
            if metadata
            and self._normalize_source(metadata.get("source", ""))
            == self._normalize_source(faq.source)
        }
        current_ids = {entry.id for entry in faq.entries}
        to_delete = existing_faq_ids - current_ids

        for entry in faq.entries:
            if entry.id in existing_faq_ids:
                continue

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
                        "subjects": entry.subjects,
                        "question_type": entry.question_type,
                        "keywords": entry.keywords,
                    },
                )
            )

        if vector_documents:
            await self.vector_store.add(vector_documents)
        if to_delete:
            await self.vector_store.delete(list(to_delete))