"""Smoke-test that live FAQ retrieval returns an answer or configured fallback."""

import shutil
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.documents.faq_importer import FAQImporter
from app.documents.faq_loader_csv import FAQLoaderCSV
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.quality.providers.decision.ollama_provider import OllamaDecisionProvider
from app.quality.ranking.decision_model_feature import DecisionModelFeature
from app.quality.ranking.dm_single_feature import DMSingleFeature
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.keyword_feature import KeywordFeature
from app.quality.ranking.reranker import Reranker
from app.services.rag_service import RagService
from app.vectorstores.chroma import ChromaVectorStore


FAQ_FIXTURE_CSV = Path(__file__).parents[1] / "fixtures" / "rag_relevance_faq.csv"
CONTENT_CONFIG_PATH = Path(__file__).parents[2] / "content_config.yaml"
content = load_content_config(str(CONTENT_CONFIG_PATH))


@pytest_asyncio.fixture
async def rag_service(tmp_path, monkeypatch) -> AsyncGenerator[RagService, None]:
    settings = get_settings()
    temp_path = tmp_path / "rag-relevance-chroma"
    temp_path.mkdir()
    monkeypatch.setattr(settings, "CHROMA_PATH", str(temp_path))
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", "rag-relevance-test")

    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    faq_document = await FAQLoaderCSV().load(str(FAQ_FIXTURE_CSV))
    await FAQImporter(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    ).import_data(faq_document)

    decision_model_feature = DecisionModelFeature(
        weight=0.0,
        decision_provider=OllamaDecisionProvider(settings.DECISION_MODEL),
    )
    decision_model_names = list(content.decision_model_prompts)
    reranker = Reranker(
        features=[
            IntentFeature(
                weight=settings.INTENT_WEIGHT,
                bonus=settings.INTENT_MATCH_BONUS,
                penalty=settings.INTENT_MISMATCH_PENALTY,
                fallback=content.fragetyp_fallback,
            ),
            KeywordFeature(
                weight=settings.KEYWORD_WEIGHT,
                normalization=settings.KEYWORD_MATCH_NORMALIZATION,
            ),
            *(
                DMSingleFeature(
                    name=name,
                        weight=0.0,
                    parent_feature=decision_model_feature,
                )
                    for name in decision_model_names
            ),
        ],
        dm_feature=decision_model_feature,
    )
    service = RagService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        answerability_gate=AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=settings.RAG_MIN_SCORE_GAP,
        ),
    )

    yield service

    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_quick_retrieve_returns_answer_or_configured_fallback(
    rag_service: RagService,
) -> None:
    answer = await rag_service.quick_retrieve("Was ist Kubernetes?")

    assert isinstance(answer, str)
    assert answer in {
        "Kubernetes ist eine Plattform zur Orchestrierung von Containern.",
        content.standard_fallback,
    }
