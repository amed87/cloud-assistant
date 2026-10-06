import pytest
from pathlib import Path

from app.answerability.answerability_gate import AnswerabilityGate
from app.services.rag_service import RagService, RagContext
from app.vectorstores.models import SearchResult
from app.config.content_settings import load_content_config

CONTENT_CONFIG_PATH = Path(__file__).parents[2] / "content_config.yaml"


class FakeEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    def __init__(self, documents: list) -> None:
        self.documents = documents

    async def search(self, embedding, limit: int, intent: str) -> list:
        return self.documents[:limit]


def create_service(monkeypatch, documents: list) -> RagService:
    settings_type = type(
        "Settings",
        (),
        {
            "RAG_TOP_K": 3,
            "RAG_MIN_SCORE": 0.7,
            "CONTENT_CONFIG_FILE": str(CONTENT_CONFIG_PATH),
        },
    )
    settings = settings_type()

    monkeypatch.setattr(
        "app.services.rag_service.get_settings",
        lambda: settings,
    )

    return RagService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(documents),
        answerability_gate=AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=0.1,
        ),
    )


def result(
    score: float,
    answer: str | None = "Antwort",
) -> SearchResult:
    metadata = {
        "source": "test.txt",
        "chunk_index": 0,
    }

    if answer is not None:
        metadata["answer"] = answer

    return SearchResult(
        id=f"document-{score}",
        content="Dokumentinhalt",
        metadata=metadata,
        score=score,
    )


@pytest.mark.asyncio
async def test_retrieve_filters_irrelevant_documents(monkeypatch) -> None:
    service = create_service(
        monkeypatch,
        [
            result(0.9),
            result(0.6),
            result(0.7),
        ],
    )

    context = await service.retrieve("Frage")

    assert len(context.documents) == 2
    assert all(document.score >= 0.7 for document in context.documents)


@pytest.mark.asyncio
async def test_quick_retrieve_returns_top_answer(monkeypatch) -> None:
    service = create_service(
        monkeypatch,
        [result(0.9, answer="Die Antwort")],
    )

    answer = await service.quick_retrieve("Frage")

    assert answer == "Die Antwort"


@pytest.mark.asyncio
async def test_quick_retrieve_returns_fallback_for_irrelevant_result(
    monkeypatch,
) -> None:
    content_config = load_content_config(str(CONTENT_CONFIG_PATH))
    service = create_service(
        monkeypatch,
        [result(0.6)],
    )

    answer = await service.quick_retrieve("Frage")

    assert answer == content_config.standard_fallback


@pytest.mark.asyncio
async def test_quick_retrieve_returns_fallback_for_ambiguous_results(
    monkeypatch,
) -> None:
    content_config = load_content_config(str(CONTENT_CONFIG_PATH))
    service = create_service(
        monkeypatch,
        [result(0.9), result(0.85)],
    )

    answer = await service.quick_retrieve("Frage")

    assert answer == content_config.standard_fallback


def test_rag_context_formats_documents() -> None:
    context = RagContext(
        documents=[
            result(0.876),
        ],
    )

    assert context.text == (
        "[Quelle: test.txt | Chunk: 0 | Relevanz: 0.876]\n"
        "Dokumentinhalt"
    )


@pytest.mark.asyncio
async def test_quick_retrieve_uses_fallback_without_answer(
    monkeypatch,
) -> None:
    content_config = load_content_config(str(CONTENT_CONFIG_PATH))
    service = create_service(
        monkeypatch,
        [result(0.9, answer=None)],
    )

    answer = await service.quick_retrieve("Frage")

    assert answer == content_config.standard_fallback


@pytest.mark.asyncio
async def test_quick_retrieve_handles_empty_answer_string(monkeypatch) -> None:
    content_config = load_content_config(str(CONTENT_CONFIG_PATH))
    service = create_service(monkeypatch, [result(0.9, answer="")])
    answer = await service.quick_retrieve("Frage")
    assert answer == content_config.standard_fallback