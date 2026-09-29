import csv
from pathlib import Path

import pytest

from app.answerability.answerability_gate import AnswerabilityGate
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.services.rag_service import RagService
from app.vectorstores.chroma import ChromaVectorStore
from app.config.content_settings import load_content_config
from app.config.settings import get_settings


GOLDEN_CSV = Path(__file__).parents[2] / "data" / "test_queries.csv"
content = load_content_config(str(Path(__file__).parents[2] / "content_config.yaml"))


def load_cases() -> list[dict]:
    with open(GOLDEN_CSV, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        embedding_provider=OllamaEmbeddingProvider(),
        vector_store=ChromaVectorStore(),
        answerability_gate=AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=settings.RAG_MIN_SCORE_GAP,
        ),
    )

@pytest.mark.asyncio(loop_scope="session")
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

