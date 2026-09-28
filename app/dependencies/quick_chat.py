from app.config.settings import get_settings

from app.embeddings.ollama_provider import (
    OllamaEmbeddingProvider,
)

from app.services.quick_chat_service import (
    QuickChatService,
)

from app.services.rag_service import (
    RagService,
)

from app.vectorstores.chroma import (
    ChromaVectorStore,
)

from app.answerability.answerability_gate import (
    AnswerabilityGate,
    )

settings = get_settings()
embedding_provider = OllamaEmbeddingProvider()
vector_store = ChromaVectorStore()
answerability_gate = AnswerabilityGate(settings.RAG_MIN_SCORE, settings.RAG_MIN_SCORE_GAP)

rag_service = RagService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    answerability_gate=answerability_gate
)


def get_quick_chat_service() -> QuickChatService:
    return QuickChatService(
        rag_service=rag_service,
    )