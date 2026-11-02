"""
Sweep intent-reranker and gate parameters.

Run:
python -m evaluation.hyperparameter_tuning
"""

import asyncio
import itertools
from dataclasses import dataclass
from typing import Iterator

from app.vectorstores.models import SearchResult
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
from app.config.content_settings import content_config

from evaluation.evaluate_rag import (
    load_cases,
    GOLDEN_CSV,
)


@dataclass(slots=True)
class SweepResult:
    match_bonus: float
    mismatch_penalty: float
    intent_weight: float

    keyword_weight: float
    keyword_normalization: float

    decision_model: str
    decision_model_weights: dict[str, float]

    min_score: float
    min_score_gap: float

    tp: int = 0
    fp: int = 0
    fn: int = 0
    wrong: int = 0

@dataclass(slots=True)
class CachedCase:
    question: str
    expected_entry: str
    chroma_results: list[SearchResult]
    feature_scores: CachedRerankScores


@dataclass(frozen=True)
class CachedRerankScores:
    """Vorab berechnete Feature-Scores für einen Case."""
    
    reranked_results: list[SearchResult]      # Sortierte Ergebnisse (ID + base_score)
    decision_scores: dict[str, dict[str, float]]
    query: str                              

    def compute_score(
        self,
        candidate: SearchResult,
        config: RerankConfig,
    ) -> float:
        """Berechnet den finalen Score mit neuen Gewichten."""
        base = candidate.score

        intent_feature = IntentFeature(
            weight=config.intent_weight,
            bonus=config.match_bonus,
            penalty=config.mismatch_penalty,
        )
        intent_score = intent_feature.score(query=self.query, candidate=candidate)

        keyword_feature = KeywordFeature(
            weight=config.keyword_weight,
            normalization=config.keyword_normalization,
        )
        keyword_score = keyword_feature.score(query=self.query, candidate=candidate)       
        
        return  (
            base +
            sum(
                weight * self.decision_scores.get(candidate.id, {}).get(name, 0.0)
                for name, weight in config.decision_model_weights.items()
            ) +
            config.intent_weight * intent_score +
            config.keyword_weight * keyword_score
        )


@dataclass(frozen=True)
class SweepParameters:
    """Alle Kandidatenlisten für den Hyperparameter-Sweep."""
    
    # Rerank-Parameter
    decision_model_candidates: list[str]
    decision_model_weight_candidates: list[float]
    decision_model_feature_names: list[str]
    match_bonus_candidates: list[float]
    mismatch_penalty_candidates: list[float]
    intent_weight_candidates: list[float]
    keyword_weight_candidates: list[float]
    keyword_normalization_candidates: list[int]
    
    # Gate-Parameter
    score_candidates: list[float]
    gap_candidates: list[float]
    
    @property
    def num_rerank_configs(self) -> int:
        """Anzahl semantisch einzigartiger Rerank-Konfigurationen (vor Filterung)."""
        return (
            len(self.decision_model_candidates)
            * len(self.decision_model_weight_candidates) ** len(self.decision_model_feature_names)
            * len(self.match_bonus_candidates)
            * len(self.mismatch_penalty_candidates)
            * len(self.intent_weight_candidates)
            * len(self.keyword_weight_candidates)
            * len(self.keyword_normalization_candidates)
        )

    @property
    def num_gate_configs(self) -> int:
        """Anzahl Gate-Konfigurationen."""
        return len(self.score_candidates) * len(self.gap_candidates)
    
    @property
    def total_combinations(self) -> int:
        """Gesamtanzahl Kombinationen vor Filterung."""
        return self.num_rerank_configs * self.num_gate_configs

@dataclass(frozen=True)
class RerankConfig:
    """Reranker-Konfiguration ohne Gate-Parameter."""
    decision_model: str
    decision_model_weights: dict[str, float]
    match_bonus: float
    mismatch_penalty: float
    intent_weight: float
    keyword_weight: float
    keyword_normalization: int


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def correctness_recall(
    tp: int,
    fn: int,
    wrong: int,
) -> float:
    return (
        tp / (tp + fn + wrong)
        if (tp + fn + wrong)
        else 0.0
    )


def f05(
    tp: int,
    fp: int,
    fn: int,
    wrong: int,
) -> float:

    p = precision(tp, fp + wrong)
    r = correctness_recall(tp, fn, wrong)

    if p == 0 and r == 0:
        return 0.0

    beta: float = 0.5

    denominator: float = (
        beta * beta * p + r
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * p
        * r
        / denominator
    )


def create_reranker(config: RerankConfig) -> Reranker:
    decision_provider = OllamaDecisionProvider(config.decision_model)
    dm_feature = DecisionModelFeature(
        weight=0.0,
        decision_provider=decision_provider,
    )

    return Reranker(
        features=[
            IntentFeature(
                weight=config.intent_weight,
                bonus=config.match_bonus,
                penalty=config.mismatch_penalty,
            ),
            KeywordFeature(
                weight=config.keyword_weight,
                normalization=config.keyword_normalization,
            ),
            *(
                DMSingleFeature(
                    name=name,
                    weight=config.decision_model_weights[name],
                    parent_feature=dm_feature,
                )
                for name in content_config.decision_model_prompts
            ),
        ],
        dm_feature=dm_feature,
    )


def generate_unique_rerank_configs(
    params: SweepParameters,
) -> Iterator[RerankConfig]:
    """
    Generator, der alle semantisch einzigartigen Rerank-Konfigurationen yieldet.
    
    Eliminiert Duplikate:
    - intent_weight == 0 ⇒ match_bonus und mismatch_penalty irrelevant
    - keyword_weight == 0 ⇒ keyword_normalization irrelevant  
    - alle decision-model Gewichte == 0 ⇒ decision_model irrelevant
    """
    for (
        decision_model,
        *decision_model_weights,
        match_bonus,
        mismatch_penalty,
        intent_weight,
        keyword_weight,
        keyword_normalization,
    ) in itertools.product(
        params.decision_model_candidates,
        *(
            [params.decision_model_weight_candidates]
            * len(params.decision_model_feature_names)
        ),
        params.match_bonus_candidates,
        params.mismatch_penalty_candidates,
        params.intent_weight_candidates,
        params.keyword_weight_candidates,
        params.keyword_normalization_candidates,
    ):
        # Filtern von Duplikaten
        if intent_weight == 0 and (match_bonus != 0 or mismatch_penalty != 0):
            continue
        if keyword_weight == 0 and keyword_normalization != params.keyword_normalization_candidates[0]:
            continue
        if all(weight == 0 for weight in decision_model_weights) and decision_model != params.decision_model_candidates[0]:
            continue
        
        yield RerankConfig(
            decision_model=decision_model,
            decision_model_weights=dict(zip(params.decision_model_feature_names, decision_model_weights)),
            match_bonus=match_bonus,
            mismatch_penalty=mismatch_penalty,
            intent_weight=intent_weight,
            keyword_weight=keyword_weight,
            keyword_normalization=keyword_normalization,
        )


async def run_sweep(
    cached_results: list[CachedCase],
    params: SweepParameters,
) -> list[SweepResult]:
    results: list[SweepResult] = []

    # Generator für eindeutige Rerank-Konfigurationen
    unique_rerank_configs = list(generate_unique_rerank_configs(params))
    total_combinations = len(unique_rerank_configs) * params.num_gate_configs 
    current_combination = 0

    print(f"\nStarting sweep ({total_combinations} combinations)")
    print(f"Unique Rerank configurations: {len(unique_rerank_configs):,}")
    print(f"Original Rerank candidates: {params.num_rerank_configs:,}")
    print(f"Filtered out: {params.num_rerank_configs - len(unique_rerank_configs):,}")

    for config in unique_rerank_configs:
        ranked_by_case: list[tuple[CachedCase, list[SearchResult]]] = []

        for case in cached_results:
            scored_candidates = [
                (cand, case.feature_scores.compute_score(cand, config))
                for cand in case.chroma_results
            ]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            ranked = [
                SearchResult(id=cand.id, content=cand.content, score=score, metadata=cand.metadata)
                for cand, score in scored_candidates
            ]
            ranked_by_case.append((case, ranked))

        for min_score in params.score_candidates:
            for min_gap in params.gap_candidates:
                current_combination += 1

                if current_combination % 10 == 0:
                    pct = current_combination / total_combinations * 100
                    print(f"\rProgress: {current_combination:,}/{total_combinations:,} ({pct:5.1f}%)",
                        end="", flush=True)

                sweep = SweepResult(
                    match_bonus=config.match_bonus,
                    mismatch_penalty=config.mismatch_penalty,
                    intent_weight=config.intent_weight,
                    keyword_weight=config.keyword_weight,
                    keyword_normalization=config.keyword_normalization,
                    min_score=min_score,
                    min_score_gap=min_gap,
                    decision_model=config.decision_model,
                    decision_model_weights=config.decision_model_weights,
                )

                gate = AnswerabilityGate(min_score=min_score, min_score_gap=min_gap)

                for case, ranked in ranked_by_case:     
                    decision = gate.evaluate_quick(case.question, ranked)

                    predicted = None
                    if decision.document:
                        predicted = decision.document.id.partition(":")[0]

                    if not case.expected_entry:
                        if decision.accepted:
                            sweep.fp += 1
                        continue

                    if not decision.accepted:
                        sweep.fn += 1
                    elif predicted == case.expected_entry:
                        sweep.tp += 1
                    else:
                        sweep.wrong += 1

                results.append(sweep)                   
    
    return results


async def retrieve_results(service: RagService, rows: list[dict]) -> list[CachedCase]:
    cached_results: list[CachedCase] = []

    print("Loading retrieval results...")

    for row in rows:
        question = row["frage"]
        expected = row["erwartete_entry_id"].strip()

        retrieval_result = await service.quick_retrieve(
            question,
            eval=True,
        )

        decision_scores = {
            cid: {
                name: scores.get(name, 0.0)
                for name in content_config.decision_model_prompts
            }
            for cid, scores in retrieval_result.feature_scores.items()
        }

        cached_results.append(
            CachedCase(
                question=question,
                expected_entry=expected,
                chroma_results=retrieval_result.chroma,
                feature_scores=CachedRerankScores(
                    reranked_results=retrieval_result.chroma,
                    decision_scores=decision_scores,
                    query=question,
                ),
            )
        )

    print(
        f"Loaded {len(cached_results)} cases."
    )
    return cached_results


def create_rag_service() -> RagService:
    settings = get_settings()
    
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    feature_names = list(content_config.decision_model_prompts)
    if len(settings.DECISION_MODEL_WEIGHTS) != len(feature_names):
        raise ValueError(
            "DECISION_MODEL_WEIGHTS must contain one weight for each "
            "decision_model_prompts entry."
        )
    default_config = RerankConfig(
        decision_model=settings.DECISION_MODEL,
        decision_model_weights=dict(zip(feature_names, settings.DECISION_MODEL_WEIGHTS)),
        match_bonus=settings.INTENT_MATCH_BONUS,
        mismatch_penalty=settings.INTENT_MISMATCH_PENALTY,
        intent_weight=settings.INTENT_WEIGHT,
        keyword_weight=settings.KEYWORD_WEIGHT,
        keyword_normalization=settings.KEYWORD_MATCH_NORMALIZATION,
    )
    reranker = create_reranker(default_config)

    answerability_gate = AnswerabilityGate(
        min_score=settings.RAG_MIN_SCORE,
        min_score_gap=settings.RAG_MIN_SCORE_GAP,
    )

    rag_service = RagService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        answerability_gate=answerability_gate,
    )

    return rag_service


def print_results(ranked: list[SweepResult]):    
    print("")
    print("\nTop 20 configurations")
    print("=" * 80)

    for r in ranked[:20]:
        metric = f05(
            r.tp,
            r.fp,
            r.fn,
            r.wrong,
        )

        print(
            f"intent weight={r.intent_weight:.2f} "
            f"intent bonus={r.match_bonus:.2f} "
            f"intent penalty={r.mismatch_penalty:.2f} "
            f"keyword weight={r.keyword_weight:.2f} "            
            f"keyword normalization={r.keyword_normalization:.0f} "
            f"decision model={r.decision_model} "
            f"decision model weights={r.decision_model_weights} "
            f"score={r.min_score:.2f} "
            f"gap={r.min_score_gap:.2f} "
            f"f05={metric:.3f} "
            f"tp={r.tp} "
            f"wrong={r.wrong} "
            f"fp={r.fp} "
            f"fn={r.fn}"
        )


def generate_candidates(
    start: float,
    stop: float,
    step: float,
) -> list[float]:

    count = int(round((stop - start) / step))

    return [
        round(start + i * step, 4)
        for i in range(count + 1)
    ]

async def main() -> None:
    rag_service = create_rag_service()
    rows = load_cases(GOLDEN_CSV)
    cached_results = await retrieve_results(rag_service, rows)

    params = SweepParameters(
        decision_model_candidates=[get_settings().DECISION_MODEL],
        decision_model_weight_candidates=generate_candidates(start=0.0, stop=1.0, step=0.2),
        decision_model_feature_names=list(content_config.decision_model_prompts),
        match_bonus_candidates=generate_candidates(start=0.0, stop=0.3, step=0.1) ,
        mismatch_penalty_candidates=generate_candidates(start=0.0, stop=0.3, step=0.1),
        intent_weight_candidates=generate_candidates(start=0.0, stop=2.0, step=0.5),
        keyword_weight_candidates=generate_candidates(start=0.0, stop=0.2, step=0.05),
        keyword_normalization_candidates=generate_candidates(start=2, stop=10, step=2),
        score_candidates=generate_candidates(start=0.0, stop=1.0, step=0.2),
        gap_candidates=generate_candidates(start=0.0, stop=0.5, step=0.1),
    )

    results = await run_sweep(cached_results=cached_results, params=params)

    ranked = sorted(
        results,
        key=lambda r: f05(r.tp, r.fp, r.fn, r.wrong,),
        reverse=True,
    )

    print_results(ranked)


if __name__ == "__main__":
    asyncio.run(main())