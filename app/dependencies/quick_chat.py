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

embedding_provider = OllamaEmbeddingProvider()

vector_store = ChromaVectorStore()

rag_service = RagService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
)


def get_quick_chat_service() -> QuickChatService:
    return QuickChatService(
        rag_service=rag_service,
    )