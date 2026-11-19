from app.config.settings import get_settings

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

from app.quality.answerability.answerability_gate import (
    AnswerabilityGate,
    )

from app.quality.ranking.reranker import (
    Reranker,
    )

from app.quality.ranking.feature_factory import build_rerank_features
from app.dependencies.providers import embedding_provider


settings = get_settings()
provider = OllamaProvider()
repository = MemoryConversationRepository()
prompt_provider = DefaultPromptProvider()
prompt_pipeline = PromptPipeline([
    SystemPromptStep(prompt_provider=prompt_provider),
    ConversationStep(),
])
vector_store = ChromaVectorStore()
# Der Chat-Pfad ruft keine Decision-Model-Prompts auf.
features, dm_feature = build_rerank_features(include_decision_model=False)
reranker = Reranker(features, dm_feature=dm_feature)
answerability_gate = AnswerabilityGate(settings.RAG_MIN_SCORE, settings.RAG_MIN_SCORE_GAP)
rag_service = RagService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    reranker=reranker,
    answerability_gate=None,
)


def get_chat_service() -> ChatService:

    return ChatService(
        provider=provider,
        repository=repository,
        prompt_pipeline=prompt_pipeline,
        rag_service=rag_service,
    )
