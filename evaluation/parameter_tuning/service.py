"""Build the production RAG service used for the retrieval cache.

Must stay structurally identical to the reranker built in
evaluation/evaluate_rag.py: one parent DecisionModelFeature plus one
DMSingleFeature per prompt. Any divergence breaks the calibration gate.
"""

from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore
from app.services.rag_service import RagService
from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.quality.providers.decision.ollama_provider import OllamaDecisionProvider
from app.quality.ranking.reranker import Reranker
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.keyword_feature import KeywordFeature
from app.quality.ranking.decision_model_feature import DecisionModelFeature
from app.quality.ranking.dm_single_feature import DMSingleFeature


def create_rag_service() -> RagService:
    """Assemble the live RAG service with per-prompt decision features."""
    settings = get_settings()
    content_config = load_content_config(path=settings.CONTENT_CONFIG_FILE)
    prompt_names = list(content_config.decision_model_prompts)

    if len(settings.DECISION_MODEL_WEIGHTS) != len(prompt_names):
        raise ValueError(
            "DECISION_MODEL_WEIGHTS must contain one weight for each "
            "decision_model_prompts entry."
        )

    decision_provider = OllamaDecisionProvider(settings.DECISION_MODEL)
    dm_feature = DecisionModelFeature(
        weight=0.0,
        decision_provider=decision_provider,
    )
    reranker = Reranker(
        features=[
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
                    weight=weight,
                    parent_feature=dm_feature,
                )
                for name, weight in zip(prompt_names, settings.DECISION_MODEL_WEIGHTS)
            ),
        ],
        dm_feature=dm_feature,
    )

    return RagService(
        embedding_provider=OllamaEmbeddingProvider(),
        vector_store=ChromaVectorStore(),
        reranker=reranker,
        answerability_gate=AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=settings.RAG_MIN_SCORE_GAP,
        ),
    )