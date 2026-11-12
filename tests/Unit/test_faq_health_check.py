"""Verify FAQ health-check reports for retrieval, margins, embeddings, and duplicates."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.documents.models import FAQDocument, FAQEntry
from app.models.faq_health import (
    DuplicatePair,
    DuplicateReport,
    EmbeddingHealthReport,
    LowMarginQuery,
    RetrievalMarginReport,
    SelfRetrievalFailure,
    SelfRetrievalReport,
)
from app.tools.faq_health_check import FAQHealthCheckService
from app.vectorstores.models import SearchResult


class FakeEmbeddingProvider:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    async def embed(self, text: str) -> list[float]:
        return self.embeddings[text]


class FakeStore:
    def __init__(
        self,
        results_by_question: dict[str, list[SearchResult]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        self.results_by_question = results_by_question or {}
        self.ids = ids or []

    async def search(self, embedding: list[float], limit: int = 5) -> list[SearchResult]:
        return self.results_by_question.get(str(embedding[0]), [])[:limit]

    def get(self) -> dict[str, list[str]]:
        return {"ids": self.ids}


class FakeReranker:
    async def rerank(
        self,
        documents: list[SearchResult],
        query: str,
    ) -> tuple[list[SearchResult], dict[str, dict[str, float]]]:
        return documents, {}


def entry(
    entry_id: str,
    question: str,
    question_type: str = "definition",
) -> FAQEntry:
    return FAQEntry(
        id=entry_id,
        question=question,
        answer=f"Answer to {question}",
        question_type=question_type,
    )


def result(
    result_id: str,
    score: float,
    answer: str = "Answer",
) -> SearchResult:
    return SearchResult(
        id=result_id,
        content=result_id,
        score=score,
        metadata={"answer": answer},
    )


def service(embeddings: dict[str, list[float]]) -> FAQHealthCheckService:
    return FAQHealthCheckService(
        embedding_provider=FakeEmbeddingProvider(embeddings),
        loader=SimpleNamespace(),
        create_temp_store=lambda: SimpleNamespace(),
        create_importer=lambda _store: SimpleNamespace(),
        reranker=FakeReranker(),
        live_progress=False,
    )


@pytest.mark.asyncio
async def test_self_retrieval_reports_missing_entries() -> None:
    entries = [entry("found", "question found"), entry("missing", "question missing")]
    checker = service({"question found": [1.0], "question missing": [2.0]})
    store = FakeStore(
        results_by_question={
            "1.0": [result("found", 0.9)],
            "2.0": [result("other", 0.8)],
        }
    )

    report = await checker._check_self_retrieval(store, entries)

    assert report.total_documents == 2
    assert report.passed_count == 1
    assert report.rate == 0.5
    assert [(failure.id, failure.question) for failure in report.failed_entries] == [
        ("missing", "question missing")
    ]


@pytest.mark.asyncio
async def test_retrieval_margin_report_contains_close_candidates() -> None:
    faq_entry = entry("query", "same question")
    checker = service({"same question": [1.0]})
    store = FakeStore(
        results_by_question={
            "1.0": [
                result("first", 0.9, answer="First answer"),
                result("second", 0.85, answer="Second answer"),
            ]
        }
    )

    report = await checker._analyze_retrieval_margins_from_queries(
        store,
        [faq_entry],
    )

    assert report.avg_margin == pytest.approx(0.05)
    assert report.min_margin == pytest.approx(0.05)
    assert report.below_threshold_count == 1
    assert report.low_margin_entries[0].rank_1_id == "first"
    assert report.low_margin_entries[0].rank_1_question == "First answer"


@pytest.mark.asyncio
async def test_embedding_health_detects_nan_zero_and_extreme_values() -> None:
    entries = [
        entry("nan", "nan"),
        entry("zero", "zero"),
        entry("extreme", "extreme"),
        entry("healthy", "healthy"),
    ]
    checker = service(
        {
            "nan": [float("nan"), 1.0],
            "zero": [0.0, 0.0],
            "extreme": [101.0, 1.0],
            "healthy": [1.0, 2.0],
        }
    )

    report = await checker._check_embedding_health_from_db(
        FakeStore(ids=[str(index) for index in range(8)]),
        entries,
    )

    assert report.total_vectors == 8
    assert report.nan_count == 2
    assert report.zero_count == 2
    assert report.extreme_values_count == 2
    assert report.health_rate == pytest.approx(0.25)
    assert report.unhealthy_ids == ["nan", "zero", "extreme"]


@pytest.mark.asyncio
async def test_duplicate_detection_marks_different_intents_as_separable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        entry("first", "first question", question_type="definition"),
        entry("second", "second question", question_type="process"),
        entry("third", "third question", question_type="time"),
    ]
    monkeypatch.setattr(
        np.random,
        "choice",
        lambda values, size, replace: entries,
    )
    checker = service(
        {
            "first question": [0.0, 0.0],
            "second question": [0.1, 0.0],
            "third question": [2.0, 2.0],
        }
    )

    report = await checker._detect_duplicates_in_store(FakeStore(), entries)

    assert report.affected_count == 2
    assert len(report.duplicate_pairs) == 1
    duplicate = report.duplicate_pairs[0]
    assert (duplicate.id_1, duplicate.id_2) == ("first", "second")
    assert duplicate.distance == pytest.approx(0.1)
    assert duplicate.semantics_issue
    assert duplicate.intent_separatable


@pytest.mark.asyncio
async def test_run_qa_checks_aggregates_failed_reports() -> None:
    checker = service({})
    checker._check_self_retrieval = lambda *_args: _return(
        SelfRetrievalReport(
            total_documents=1,
            passed_count=0,
            rate=0.0,
            failed_entries=[SelfRetrievalFailure(id="entry", question="Question")],
        )
    )
    checker._analyze_retrieval_margins_from_queries = lambda *_args: _return(
        RetrievalMarginReport(
            avg_margin=0.05,
            min_margin=0.05,
            below_threshold_count=1,
            low_margin_entries=[
                LowMarginQuery(
                    query_id="entry",
                    query_question="Question",
                    rank_1_id="first",
                    rank_1_question="First",
                    rank_1_score=0.8,
                    rank_2_id="second",
                    rank_2_question="Second",
                    rank_2_score=0.75,
                    margin=0.05,
                )
            ],
        )
    )
    checker._check_embedding_health_from_db = lambda *_args: _return(
        EmbeddingHealthReport(
            total_vectors=1,
            nan_count=1,
            zero_count=0,
            extreme_values_count=0,
            health_rate=0.0,
            unhealthy_ids=["entry"],
        )
    )
    checker._detect_duplicates_in_store = lambda *_args: _return(
        DuplicateReport(
            duplicate_pairs=[
                DuplicatePair(
                    id_1="first",
                    question_1="First question",
                    id_2="second",
                    question_2="Second question",
                    distance=0.1,
                )
            ],
            affected_count=2,
        )
    )

    report = await checker._run_qa_checks(FakeStore(), [entry("entry", "Question")])

    assert report.status == "FAIL"
    assert len(report.messages) == 4
    assert "1 Fragen fanden sich nicht selbst." in report.messages[0]


@pytest.mark.asyncio
async def test_run_qa_fails_empty_faq_and_always_closes_temp_store() -> None:
    class EmptyLoader:
        async def load(self, file_path: str) -> FAQDocument:
            assert file_path == "empty.csv"
            return FAQDocument(id="empty", source=file_path, entries=[])

    class TempStore:
        closed = False

        async def close(self) -> None:
            self.closed = True

    temp_store = TempStore()
    checker = FAQHealthCheckService(
        embedding_provider=FakeEmbeddingProvider({}),
        loader=EmptyLoader(),
        create_temp_store=lambda: temp_store,
        create_importer=lambda _store: pytest.fail("empty FAQ must not be imported"),
        reranker=FakeReranker(),
        live_progress=False,
    )

    report = await checker.run_qa("empty.csv")

    assert report.status == "FAIL"
    assert report.self_retrieval.total_documents == 0
    assert report.messages == ["Keine FAQ-Einträge vorhanden."]
    assert temp_store.closed


async def _return(value):
    return value
