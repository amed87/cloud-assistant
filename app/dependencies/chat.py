from app.embeddings.ollama_provider import (
    OllamaEmbeddingProvider,
)

from app.providers.ollama_provider import OllamaProvider

from app.repositories.memory import (
    MemoryConversationRepository,
)

from app.prompts.default import (
    DefaultPromptProvider,
)

from app.prompts.pipeline import (
    PromptPipeline,
)

from app.prompts.steps.system_prompt import (
    SystemPromptStep,
)

from app.prompts.steps.conversation import (
    ConversationStep,
)

from app.services.chat_service import (
    ChatService,
)

from app.services.rag_service import (
    RagService,
)

from app.vectorstores.chroma import (
    ChromaVectorStore,
)


provider = OllamaProvider()

repository = MemoryConversationRepository()

prompt_provider = DefaultPromptProvider()

prompt_pipeline = PromptPipeline([
    SystemPromptStep(prompt_provider=prompt_provider),
    ConversationStep(),
])

embedding_provider = OllamaEmbeddingProvider()

vector_store = ChromaVectorStore()

rag_service = RagService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    answerability_gate=None,
)


def get_chat_service() -> ChatService:

    return ChatService(
        provider=provider,
        repository=repository,
        prompt_pipeline=prompt_pipeline,
        rag_service=rag_service,
    )