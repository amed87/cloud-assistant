from abc import ABC, abstractmethod
from typing import Any

from app.documents.models import Document
from app.embeddings.base import EmbeddingProvider
from app.vectorstores.base import VectorStore


class DocumentImporter(ABC):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def get_current_database(self) -> dict[str, Any]:
        return self.vector_store.get()

    @abstractmethod
    async def import_data(self, data: Document) -> None:
        pass