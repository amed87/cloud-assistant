"""Cover named decision-weight combinations and cached reranker score calculation."""

import pytest

from app.config.content_settings import content_config
from app.quality.ranking.dm_single_feature import DMSingleFeature
from app.vectorstores.models import SearchResult
from evaluation.hyperparameter_tuning import (
    CachedRerankScores,
    RerankConfig,
    SweepParameters,
    create_reranker,
    generate_unique_rerank_configs,
)


def test_sweep_generates_a_weight_for_each_decision_prompt() -> None:
    params = SweepParameters(
        decision_model_candidates=["test-model"],
        decision_model_weight_candidates=[0.0, 0.5],
        decision_model_feature_names=["first", "second"],
        match_bonus_candidates=[0.0],
        mismatch_penalty_candidates=[0.0],
        intent_weight_candidates=[1.0],
        keyword_weight_candidates=[1.0],
        keyword_normalization_candidates=[2],
        score_candidates=[0.0],
        gap_candidates=[0.0],
    )

    configs = list(generate_unique_rerank_configs(params))

    assert params.num_rerank_configs == 4
    assert len(configs) == 4
    assert {
        tuple(config.decision_model_weights.values())
        for config in configs
    } == {
        (0.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.0),
        (0.5, 0.5),
    }


def test_cached_scores_apply_weights_by_prompt_name() -> None:
    candidate = SearchResult(
        id="candidate",
        content="question",
        score=0.4,
        metadata={},
    )
    config = RerankConfig(
        decision_model="test-model",
        decision_model_weights={"first": 0.5, "second": 0.25},
        match_bonus=0.0,
        mismatch_penalty=0.0,
        intent_weight=0.0,
        keyword_weight=0.0,
        keyword_normalization=2,
    )
    cached_scores = CachedRerankScores(
        reranked_results=[],
        decision_scores={"candidate": {"first": 0.8, "second": 0.4}},
        query="query",
    )

    assert cached_scores.compute_score(candidate, config) == pytest.approx(0.9)


def test_create_reranker_uses_one_single_feature_per_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = {"first": "prompt one", "second": "prompt two"}
    monkeypatch.setattr(content_config, "decision_model_prompts", prompts)
    config = RerankConfig(
        decision_model="test-model",
        decision_model_weights={"first": 0.3, "second": 0.7},
        match_bonus=0.1,
        mismatch_penalty=0.2,
        intent_weight=1.0,
        keyword_weight=0.0,
        keyword_normalization=2,
    )

    reranker = create_reranker(config)
    decision_features = [
        feature
        for feature in reranker._features
        if isinstance(feature, DMSingleFeature)
    ]

    assert [feature.name for feature in decision_features] == ["first", "second"]
    assert [feature.weight for feature in decision_features] == [0.3, 0.7]
    assert all(feature.parent_feature is reranker._dm_feature for feature in decision_features)
