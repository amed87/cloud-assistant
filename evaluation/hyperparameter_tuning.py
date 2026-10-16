"""
Sweep intent-reranker and gate parameters.

Run:
python -m evaluation.sweep_reranker
"""

import asyncio
from dataclasses import dataclass
from typing import Sequence

from app.vectorstores.models import SearchResult
from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore
from app.services.rag_service import RagService
from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.quality.models import AnswerabilityDecision
from app.quality.ranking.reranker import Reranker
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.keyword_feature import KeywordFeature

from evaluation.evaluate_rag import (
    load_cases,
    GOLDEN_CSV,
)


@dataclass
class SweepResult:
    match_bonus: float
    mismatch_penalty: float
    intent_weight: float

    keyword_weight: float
    keyword_normalization: float

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


def create_reranker(
    intent_weight: float = 1.0,
    intent_match_bonus: float = 0.15,
    intent_mismatch_penalty: float = 0.1,
    keyword_weight: float = 0.0,
    keyword_normalization: float = 5,
) -> Reranker:

    return Reranker(
        features=[
            IntentFeature(
                weight=intent_weight,
                bonus=intent_match_bonus,
                penalty=intent_mismatch_penalty,
            ),
            KeywordFeature(
                weight=keyword_weight,
                normalization=keyword_normalization,
            ),
        ]
    )


def run_sweep(
    cached_results: list[CachedCase],
    match_bonus_candidates: list[float],
    mismatch_penalty_candidates: list[float],
    score_candidates: list[float],
    gap_candidates: list[float],
    keyword_weight_candidates: list[float],
    intent_weight_candidates: list[float],
    keyword_normalization_candidates: list[int],
) -> list[SweepResult]:
    results: list[SweepResult] = []
    total_combinations = (
        len(match_bonus_candidates)
        * len(mismatch_penalty_candidates)
        * len(score_candidates)
        * len(gap_candidates)
        * len(intent_weight_candidates)
        * len(keyword_weight_candidates)
        * len(keyword_normalization_candidates)
    )
    current_combination = 0

    for match_bonus in match_bonus_candidates:
        for mismatch_penalty in mismatch_penalty_candidates:
            for intent_weight in intent_weight_candidates:
                for keyword_weight in keyword_weight_candidates:                   
                    for keyword_normalization in keyword_normalization_candidates:
                        for min_score in score_candidates:
                            for min_gap in gap_candidates:                            
                                current_combination += 1

                                if current_combination % 10 == 0:
                                    pct = current_combination / total_combinations * 100

                                    print(
                                        f"\rProgress: "
                                        f"{current_combination:,}/{total_combinations:,} "
                                        f"({pct:5.1f}%)",
                                        end="",
                                        flush=True,
                                    )

                                sweep = SweepResult(
                                    match_bonus=match_bonus,
                                    mismatch_penalty=mismatch_penalty,
                                    intent_weight=intent_weight,
                                    keyword_weight=keyword_weight,
                                    keyword_normalization=keyword_normalization,
                                    min_score=min_score,
                                    min_score_gap=min_gap,
                                )

                                reranker = create_reranker(
                                    intent_weight=intent_weight,
                                    intent_match_bonus=match_bonus,
                                    intent_mismatch_penalty=mismatch_penalty,
                                    keyword_weight=keyword_weight,
                                    keyword_normalization=keyword_normalization,
                                )

                                gate = AnswerabilityGate(
                                    min_score=min_score,
                                    min_score_gap=min_gap,
                                )

                                for case in cached_results:
                                    reranked: list[SearchResult] = reranker.rerank(candidates=case.chroma_results, query=case.question)

                                    decision: AnswerabilityDecision = (
                                        gate.evaluate_quick(
                                            case.question,
                                            reranked,
                                        )
                                    )

                                    predicted = None

                                    if decision.document:
                                        predicted = (
                                            decision.document.id
                                            .partition(":")[0]
                                        )

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

        cached_results.append(
            CachedCase(
                question=question,
                expected_entry=expected,
                chroma_results=retrieval_result.chroma,
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
    reranker = create_reranker(
        intent_weight=settings.INTENT_WEIGHT,
        intent_match_bonus=settings.INTENT_MATCH_BONUS,
        intent_mismatch_penalty=settings.INTENT_MISMATCH_PENALTY,
        keyword_weight=settings.KEYWORD_WEIGHT,
        keyword_normalization=settings.KEYWORD_MATCH_NORMALIZATION,
    )

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

    match_bonus_candidates = generate_candidates(start=0.0, stop=0.3, step=0.1) 
    mismatch_penalty_candidates = generate_candidates(start=0.0, stop=0.3, step=0.1)
    intent_weight_candidates = generate_candidates(start=0.0, stop=2.0, step=0.5)
    keyword_weight_candidates = generate_candidates(start=0.0, stop=0.2, step=0.05)
    keyword_normalization_candidates = generate_candidates(start=2, stop=10, step=2)
    score_candidates = generate_candidates(start=0.0, stop=1.0, step=0.2)
    gap_candidates = generate_candidates(start=0.0, stop=0.5, step=0.1)

    results = run_sweep(
        cached_results=cached_results,
        match_bonus_candidates=match_bonus_candidates,
        mismatch_penalty_candidates=mismatch_penalty_candidates,
        score_candidates=score_candidates,
        gap_candidates=gap_candidates,
        keyword_weight_candidates=keyword_weight_candidates,
        intent_weight_candidates=intent_weight_candidates,
        keyword_normalization_candidates=keyword_normalization_candidates
    )

    ranked = sorted(
        results,
        key=lambda r: f05(
            r.tp,
            r.fp,
            r.fn,
            r.wrong,
        ),
        reverse=True,
    )

    print_results(ranked)


if __name__ == "__main__":
    asyncio.run(main())