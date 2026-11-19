"""Build the production RAG service used for the retrieval cache.

Uses the shared feature factory (app.quality.ranking.feature_factory), so
the cached reranker is structurally identical to the reranker in
evaluation/evaluate_rag.py and app/dependencies/quick_chat.py — any
divergence would break the calibration gate.
"""

from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore
from app.services.rag_service import RagService
from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.quality.ranking.reranker import Reranker
from app.quality.ranking.feature_factory import build_rerank_features


def create_rag_service() -> RagService:
    """Assemble the live RAG service with per-prompt decision features."""
    settings = get_settings()
    features, dm_feature = build_rerank_features()
    reranker = Reranker(features=features, dm_feature=dm_feature)

    return RagService(
        embedding_provider=OllamaEmbeddingProvider(),
        vector_store=ChromaVectorStore(),
        reranker=reranker,
        answerability_gate=AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=settings.RAG_MIN_SCORE_GAP,
        ),
    )
