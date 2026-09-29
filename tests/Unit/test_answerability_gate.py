import pytest

from app.answerability.answerability_gate import AnswerabilityGate
from app.vectorstores.models import SearchResult


def result(
    score: float,
    answer: str | None = "Antwort",
) -> SearchResult:
    metadata = {} if answer is None else {"answer": answer}
    return SearchResult(
        id=f"document-{score}",
        content="Dokumentinhalt",
        metadata=metadata,
        score=score,
    )


def test_rejects_empty_candidate_list() -> None:
    gate = AnswerabilityGate(min_score=0.7, min_score_gap=0.1)

    decision = gate.evaluate_quick("Frage", [])

    assert not decision.accepted
    assert decision.reason == "no_candidates"
    assert decision.confidence == 0.0
    assert decision.document is None


def test_rejects_score_below_threshold() -> None:
    document = result(0.69)
    gate = AnswerabilityGate(min_score=0.7, min_score_gap=0.1)

    decision = gate.evaluate_quick("Frage", [document])

    assert not decision.accepted
    assert decision.reason == "score_below_threshold"
    assert decision.confidence == 0.69
    assert decision.document == document


def test_accepts_score_equal_to_threshold_for_single_candidate() -> None:
    document = result(0.7)
    gate = AnswerabilityGate(min_score=0.7, min_score_gap=0.1)

    decision = gate.evaluate_quick("Frage", [document])

    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.confidence == 0.7
    assert decision.document == document


def test_rejects_candidates_with_gap_below_threshold() -> None:
    best = result(0.9)
    second = result(0.85)
    gate = AnswerabilityGate(min_score=0.7, min_score_gap=0.1)

    decision = gate.evaluate_quick("Frage", [best, second])

    assert not decision.accepted
    assert decision.reason == "ambiguous_candidates"
    assert decision.confidence == pytest.approx(0.05)
    assert decision.document == best


def test_accepts_candidates_with_gap_equal_to_threshold() -> None:
    best = result(0.75)
    second = result(0.5)
    gate = AnswerabilityGate(min_score=0.7, min_score_gap=0.25)

    decision = gate.evaluate_quick("Frage", [best, second])

    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.document == best


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_rejects_missing_or_blank_answer(answer: str | None) -> None:
    gate = AnswerabilityGate(min_score=0.7, min_score_gap=0.1)

    decision = gate.evaluate_quick("Frage", [result(0.9, answer=answer)])

    assert not decision.accepted
    assert decision.reason == "missing_answer"
    assert decision.confidence == 0.9
