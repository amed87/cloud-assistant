import asyncio
import logging

from chromadb import PersistentClient

from app import documents
from app.config.settings import get_settings
from app.exceptions.vectorstore import VectorStoreError
from app.vectorstores.base import VectorStore
from app.vectorstores.models import (
    SearchResult,
    VectorDocument,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):

    def __init__(self):
        settings = get_settings()

        self.client = PersistentClient(
            path=settings.CHROMA_PATH,
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
        )

    async def add(
        self,
        documents: list[VectorDocument],
    ) -> None:
        try:
            await asyncio.to_thread(
                self.collection.upsert,
                ids=[
                    document.id
                    for document in documents
                ],
                documents=[
                    document.content
                    for document in documents
                ],
                embeddings=[
                    document.embedding
                    for document in documents
                ],
                metadatas=[
                    document.metadata
                    for document in documents
                ],
            )

            stored_count = await asyncio.to_thread(self.collection.count)
            expected = len(documents)
            logger.info("Stored %d/%d documents in ChromaDB (collection total: %d).",
            expected, expected, stored_count)

        except Exception as ex:

            logger.exception(
                "Failed to store documents in ChromaDB."
            )

            raise VectorStoreError(
                "Fehler beim Speichern in ChromaDB."
            ) from ex

    async def search(
        self,
        embedding: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:

        try:

            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[embedding],
                n_results=limit,
            )

            documents = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            search_results: list[SearchResult] = []

            for id_, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            ):

                search_results.append(
                    SearchResult(
                        id=id_,
                        content=document,
                        score=1 - distance,
                        metadata=metadata or {},
                    )
                )

            logger.info(
                "Found %d matching document(s).",
                len(search_results),
            )

            return search_results

        except Exception as ex:

            logger.exception(
                "Failed to search ChromaDB."
            )

            raise VectorStoreError(
                "Fehler bei der Suche in ChromaDB."
            ) from ex

    async def delete(
        self,
        ids: list[str],
    ) -> None:
        try:

            await asyncio.to_thread(
                self.collection.delete,
                ids=ids,
            )

            logger.info(
                "Deleted %d document(s) from ChromaDB.",
                len(ids),
            )

        except Exception as ex:

            logger.exception(
                "Failed to delete documents from ChromaDB."
            )

            raise VectorStoreError(
                "Fehler beim Löschen aus ChromaDB."
            ) from ex

    def get(self) -> dict[str, object]:
        try:
            db = self.collection.get(include=["metadatas"])
            logger.info("Gesamte Datenbank gefetcht.")
            return db
        except Exception as e:
            raise VectorStoreError(
                f"Fehler beim Fetchen der Datenbank: {e}"
            )