from app.config.settings import get_settings

from app.services.quick_chat_service import (
    QuickChatService,
)

from app.services.rag_service import (
    RagService,
)

from app.vectorstores.chroma import (
    ChromaVectorStore,
)

from app.quality.answerability.answerability_gate import (
    AnswerabilityGate,
    )

from app.quality.ranking.reranker import (
    Reranker,
    )

from app.quality.ranking.feature_factory import build_rerank_features
from app.dependencies.providers import (
    embedding_provider,
)

settings = get_settings()
vector_store = ChromaVectorStore()
features, dm_feature = build_rerank_features()
reranker = Reranker(features, dm_feature=dm_feature)
answerability_gate = AnswerabilityGate(settings.RAG_MIN_SCORE, settings.RAG_MIN_SCORE_GAP)

rag_service = RagService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    reranker=reranker,
    answerability_gate=answerability_gate
)


def get_quick_chat_service() -> QuickChatService:
    return QuickChatService(
        rag_service=rag_service,
    )
