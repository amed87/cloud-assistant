"""Verify decision-feature priming, prompt-specific scores, and candidate ordering."""

from types import SimpleNamespace

import pytest

from app.config.content_settings import content_config
from app.quality.ranking.decision_model_feature import DecisionModelFeature
from app.quality.ranking.dm_single_feature import DMSingleFeature
from app.quality.ranking.reranker import Reranker
from app.vectorstores.models import SearchResult


class FakeDecisionProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], list[str]]] = []

    async def decide(
        self,
        query: str,
        questions: dict[str, str],
        choices: list[str],
    ) -> dict[str, SimpleNamespace]:
        self.calls.append((query, questions, choices))
        feature_probabilities = [
            {"first candidate": 0.8, "second candidate": 0.2},
            {"first candidate": 0.1, "second candidate": 0.9},
        ]
        return {
            name: SimpleNamespace(
                probabilities=probabilities,
                confidence=1.0,
            )
            for name, probabilities in zip(questions, feature_probabilities)
        }


def candidate(candidate_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(
        id=candidate_id,
        content=content,
        score=score,
        metadata={},
    )


@pytest.mark.asyncio
async def test_reranker_primes_decision_scores_once_and_applies_named_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = {"intent_match": "prompt one", "answer_match": "prompt two"}
    monkeypatch.setattr(content_config, "decision_model_prompts", prompts)
    provider = FakeDecisionProvider()
    decision_feature = DecisionModelFeature(weight=0.0, decision_provider=provider)
    reranker = Reranker(
        features=[
            DMSingleFeature(
                name="intent_match",
                weight=1.0,
                parent_feature=decision_feature,
            ),
            DMSingleFeature(
                name="answer_match",
                weight=0.5,
                parent_feature=decision_feature,
            ),
        ],
        dm_feature=decision_feature,
    )
    candidates = [
        candidate("one", "first candidate", 0.4),
        candidate("two", "second candidate", 0.5),
    ]

    ranked, scores = await reranker.rerank(candidates, "query")

    assert len(provider.calls) == 1
    assert provider.calls[0] == ("query", prompts, ["first candidate", "second candidate"])
    assert [document.id for document in ranked] == ["one", "two"]
    assert ranked[0].score == pytest.approx(1.25)
    assert ranked[1].score == pytest.approx(1.15)
    assert scores["one"] == {
        "base": 0.4,
        "intent_match": 0.8,
        "answer_match": 0.1,
    }
    assert scores["two"] == {
        "base": 0.5,
        "intent_match": 0.2,
        "answer_match": 0.9,
    }


@pytest.mark.asyncio
async def test_reranker_handles_empty_candidate_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_config, "decision_model_prompts", {"intent_match": "prompt"})
    provider = FakeDecisionProvider()
    decision_feature = DecisionModelFeature(weight=0.0, decision_provider=provider)
    reranker = Reranker(
        features=[
            DMSingleFeature(
                name="intent_match",
                weight=1.0,
                parent_feature=decision_feature,
            )
        ],
        dm_feature=decision_feature,
    )

    ranked, scores = await reranker.rerank([], "query")

    assert ranked == []
    assert scores == {}
    assert provider.calls == []
