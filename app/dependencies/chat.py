from app.config.settings import get_settings
from app.config.content_settings import content_config
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

from app.quality.answerability.answerability_gate import (
    AnswerabilityGate,
    )

from app.quality.ranking.reranker import (
    Reranker,
    )

from app.quality.ranking.intent_feature import (
    IntentFeature,
)

from app.quality.ranking.keyword_feature import (
    KeywordFeature,
)


settings = get_settings()
provider = OllamaProvider()
repository = MemoryConversationRepository()
prompt_provider = DefaultPromptProvider()
prompt_pipeline = PromptPipeline([
    SystemPromptStep(prompt_provider=prompt_provider),
    ConversationStep(),
])
embedding_provider = OllamaEmbeddingProvider()
vector_store = ChromaVectorStore()
features = [
    IntentFeature(
        weight=settings.INTENT_WEIGHT, 
        bonus=settings.INTENT_MATCH_BONUS,
        penalty=settings.INTENT_MISMATCH_PENALTY,
        fallback=content_config.fragetyp_fallback,
        ),
    KeywordFeature(
        weight=settings.KEYWORD_WEIGHT,
        normalization=settings.KEYWORD_MATCH_NORMALIZATION,
    )
    ]
reranker = Reranker(features)
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