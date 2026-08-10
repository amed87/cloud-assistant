import logging

from pydantic import BaseModel

from app.embeddings.base import EmbeddingProvider
from app.exceptions.vectorstore import VectorStoreError
from app.vectorstores.base import VectorStore
from app.vectorstores.models import SearchResult

logger = logging.getLogger(__name__)


class RagContext(BaseModel):
    documents: list[SearchResult]

    @property
    def text(self) -> str:
        return "\n\n".join(
            document.content
            for document in self.documents
        )


class RagService:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def retrieve(
        self,
        question: str,
        limit: int = 5,
    ) -> RagContext:

        logger.info(
            "Retrieving RAG context for question: %s",
            question,
        )

        try:

            embedding = await self.embedding_provider.embed(
                question,
            )

            documents = await self.vector_store.search(
                embedding=embedding,
                limit=limit,
            )

            logger.info(
                "Retrieved %d document(s).",
                len(documents),
            )

            return RagContext(
                documents=documents,
            )

        except VectorStoreError:

            logger.exception(
                "Vector store retrieval failed."
            )

            raise

        except Exception:

            logger.exception(
                "Unexpected error while retrieving RAG context."
            )

            raise