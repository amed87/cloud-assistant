import csv
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.answerability.answerability_gate import AnswerabilityGate
from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.documents.faq_importer import FAQImporter
from app.documents.faq_loader_csv import FAQLoaderCSV
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.services.rag_service import RagService
from app.vectorstores.chroma import ChromaVectorStore


GOLDEN_CSV = Path(__file__).parents[1] / "fixtures" / "test_queries.csv"
FAQ_FIXTURE_CSV = Path(__file__).parents[1] / "fixtures" / "rag_relevance_faq.csv"
content = load_content_config(str(Path(__file__).parents[2] / "content_config.yaml"))


def load_cases() -> list[dict]:
    with open(GOLDEN_CSV, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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

    service = RagService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        answerability_gate=AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=settings.RAG_MIN_SCORE_GAP,
        ),
    )

    yield service

    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("case", load_cases(), ids=lambda c: f"{c['kategorie']}:{c['frage'][:30]}")
async def test_quick_retrieve(rag_service, case) -> None:
    answer = await rag_service.quick_retrieve(case["frage"])

    kategorie = case["kategorie"]
    if kategorie == "negativ":
        assert answer == content.standard_fallback
    elif kategorie == "leeres_answer":
        assert answer == content.no_answer_fallback
    else:
        # positiv, paraphrase, fast_treffer: Treffer erwartet – Score nahe Threshold kann schwanken
        assert answer not in (content.standard_fallback, content.no_answer_fallback)

