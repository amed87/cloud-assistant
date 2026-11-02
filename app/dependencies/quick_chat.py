from app.config.settings import get_settings
from app.config.content_settings import content_config

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

from app.quality.providers.decision.ollama_provider import (
    OllamaDecisionProvider,
)

from app.quality.ranking.decision_model_feature import (
    DecisionModelFeature,
)

from app.quality.ranking.dm_single_feature import (
    DMSingleFeature,
)

settings = get_settings()
embedding_provider = OllamaEmbeddingProvider()
decision_provider = OllamaDecisionProvider(settings.DECISION_MODEL)
vector_store = ChromaVectorStore()
dm_feature = DecisionModelFeature(
    weight=0.0,
    decision_provider=decision_provider,
)
decision_model_names = list(content_config.decision_model_prompts)
if len(settings.DECISION_MODEL_WEIGHTS) != len(decision_model_names):
    raise ValueError(
        "DECISION_MODEL_WEIGHTS must contain one weight for each "
        "decision_model_prompts entry."
    )
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
    ),
    *(
        DMSingleFeature(
            name=name,
            weight=settings.DECISION_MODEL_WEIGHTS[index],
            parent_feature=dm_feature,
        )
        for index, name in enumerate(decision_model_names)
    ),
]
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