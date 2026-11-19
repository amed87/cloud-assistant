import logging

from pydantic import BaseModel

from app.config.settings import get_settings
from app.config.content_settings import load_content_config
from app.embeddings.base import EmbeddingProvider
from app.vectorstores.base import VectorStore
from app.vectorstores.models import SearchResult
from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.quality.ranking.reranker import Reranker
from app.quality.models import RetrievalResult


logger = logging.getLogger(__name__)


class RagContext(BaseModel):
    """Retrieved documents prepared as context for the language model."""

    documents: list[SearchResult]

    @property
    def text(self) -> str:
        """Format documents with source, chunk, and relevance metadata."""
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
    """Retrieve, rank, filter, and format documents for chat or quick answers."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
        answerability_gate: AnswerabilityGate | None,
    ):
        settings = get_settings()

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker
        self.answerability_gate = answerability_gate
        self.content_config = load_content_config(
            path=settings.CONTENT_CONFIG_FILE
        )

        self.top_k = settings.RAG_TOP_K
        self.min_score = settings.RAG_MIN_SCORE

    async def quick_retrieve(
            self,
            question: str,
            eval: bool = False,
    ) -> str | RetrievalResult:
        """Return the accepted FAQ answer or fallback; optionally expose diagnostics."""
        logger.info("Retrieving top document for question: %s", question)
        answer = self.content_config.standard_fallback

        documents, query_embedding = await self._retrieve_candidates(
            question,
            limit=5,
        )

        reranked, feature_scores = await self.reranker.rerank(
            documents,
            question,
            query_embedding=query_embedding,
        )
        decision = self.answerability_gate.evaluate_quick(
            question=question,
            documents=reranked,
        )

        if not decision.accepted:
            logger.info(f"No corresponding FAQ-Entry found for the question. \nReason: {decision.reason} \nConfidence score: {decision.confidence}")
        else:
            answer = decision.document.metadata.get("answer")


        logger.info(
            "Top document retrieved with score: %s",
            decision.document.score if decision.document else None,
        )

        if eval:
            return RetrievalResult(
                chroma=documents,
                reranker=reranked,
                feature_scores=feature_scores,
                gate=decision,
            )
        
        return answer
    
    async def retrieve(
        self,
        question: str,
    ) -> RagContext:
        """Return top-K candidates meeting the configured relevance threshold."""
        logger.info(
            "Retrieving RAG context for question: %s",
            question,
        )

        documents, _query_embedding = await self._retrieve_candidates(
            question,
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

    async def _retrieve_candidates(
            self,
            question: str,
            limit: int,
        ) -> tuple[list[SearchResult], list[float]]:
        """Embed the query and search the vector store for its nearest candidates.

        Returns the candidates together with the query embedding so reranker
        features (e.g. centroid classification) can reuse it without a second
        embedding call.
        """
        embedding = await self.embedding_provider.embed(question)

        documents = await self.vector_store.search(
            embedding=embedding,
            limit=limit,
        )
        return documents, embedding