import asyncio
import logging
import shutil
import uuid
from pathlib import Path

from chromadb import PersistentClient

from app.config.settings import get_settings
from app.exceptions.vectorstore import VectorStoreError
from app.vectorstores.base import VectorStore  # ✅ Erbt von VectorStore
from app.vectorstores.models import SearchResult, VectorDocument

logger = logging.getLogger(__name__)

class TempChromaStore(VectorStore):  # ✅ Implementiert VectorStore Interface
    """Temporäre ChromaDB-Instanz für QA-Checks. Implementiert VectorStore Interface."""

    def __init__(self, temp_path: str | None = None):
        settings = get_settings()

        if temp_path:
            self.temp_path = temp_path
        else:
            unique_id = str(uuid.uuid4())[:8]
            self.temp_path = f"{settings.CHROMA_TEMP_PATH}_{unique_id}"

        Path(self.temp_path).mkdir(parents=True, exist_ok=True)

        self.client = PersistentClient(path=self.temp_path)
        self.collection = self.client.get_or_create_collection(name="qa-temporary")

        logger.info("Initialized temporary ChromaDB at %s", self.temp_path)

    async def add(
        self,
        documents: list[VectorDocument],
    ) -> None:
        """Speichert Dokumente im Temp Vector Store."""
        try:
            await asyncio.to_thread(
                self.collection.upsert,
                ids=[doc.id for doc in documents],
                documents=[doc.content for doc in documents],
                embeddings=[doc.embedding for doc in documents],
                metadatas=[doc.metadata for doc in documents],
            )
            count = await asyncio.to_thread(self.collection.count)
            logger.info("Added %d documents to temp store (total: %d)", len(documents), count)
        except Exception as ex:
            logger.exception("Failed to add documents to temp store")
            raise VectorStoreError("Failed to add documents to temporary vector store.") from ex

    async def search(
        self,
        embedding: list[float],
        limit: int = 5,
        intent: str = "general",
    ) -> list[SearchResult]:
        """Sucht ähnliche Dokumente im Temp Vector Store."""
        try:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[embedding],
                n_results=limit,
            )

            return [
                SearchResult(
                    id=id_,
                    content=document,
                    score=1 - distance,
                    metadata=metadata or {},
                )
                for id_, document, metadata, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        except Exception as ex:
            logger.exception("Failed to search temp store")
            raise VectorStoreError("Failed to search temporary vector store.") from ex

    async def delete(
        self,
        ids: list[str],
    ) -> None:
        """Löscht Dokumente aus Temp Vector Store."""
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=ids,
            )
            logger.info("Deleted %d documents from temp store", len(ids))
        except Exception as ex:
            logger.exception("Failed to delete documents from temp store")
            raise VectorStoreError("Failed to delete documents from temporary vector store.") from ex

    def get(self) -> dict[str, object]:
        """Fetch the entire database."""
        try:
            db = self.collection.get(include=["metadatas"])
            logger.info("Fetched all documents from temp store.")
            return db
        except Exception as ex:
            logger.exception("Failed to fetch temp store database")
            raise VectorStoreError("Failed to fetch temporary vector store database.") from ex

    async def close(self) -> None:
        """Cleanup: temporäre DB löschen."""
        try:
            shutil.rmtree(self.temp_path)
            logger.info("Cleaned up temporary ChromaDB at %s", self.temp_path)
        except Exception as ex:
            logger.warning("Failed to cleanup temp store: %s", ex)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()