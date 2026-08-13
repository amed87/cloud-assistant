import logging

from pydantic import BaseModel

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingProvider
from app.exceptions.vectorstore import VectorStoreError
from app.vectorstores.base import VectorStore
from app.vectorstores.models import SearchResult


logger = logging.getLogger(__name__)


class RagContext(BaseModel):
    documents: list[SearchResult]

    @property
    def text(self) -> str:
        chunks = []

        for document in self.documents:

            source = document.metadata.get(
                "source",
                "unknown",
            )

            chunk_index = document.metadata.get(
                "chunk_index",
                "unknown",
            )

            chunks.append(
                f"[Quelle: {source} | "
                f"Chunk: {chunk_index} | "
                f"Relevanz: {document.score:.3f}]\n"
                f"{document.content}"
            )

        return "\n\n".join(chunks)


class RagService:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        settings = get_settings()

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

        self.top_k = settings.RAG_TOP_K
        self.min_score = settings.RAG_MIN_SCORE

    async def retrieve(
        self,
        question: str,
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
                limit=self.top_k,
            )

            logger.info(
                "Retrieved %d document(s) before filtering.",
                len(documents),
            )

            relevant_documents = [
                document
                for document in documents
                if document.score >= self.min_score
            ]

            logger.info(
                "Retained %d document(s) with score >= %.2f.",
                len(relevant_documents),
                self.min_score,
            )

            return RagContext(
                documents=relevant_documents,
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