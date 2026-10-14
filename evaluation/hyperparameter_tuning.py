"""
Sweep intent-reranker and gate parameters.

Run:
python -m evaluation.sweep_reranker
"""

import asyncio
from dataclasses import dataclass
from typing import Sequence

from app.vectorstores.models import SearchResult
from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore
from app.services.rag_service import RagService
from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.quality.models import AnswerabilityDecision
from app.quality.intent.intent_filter import filterIntent

from evaluation.evaluate_rag import (
    load_cases,
    GOLDEN_CSV,
)


@dataclass
class SweepResult:
    match_bonus: float
    mismatch_penalty: float

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



def detect_intent(text: str) -> str:
    return filterIntent(text)


def rerank_candidates(
    candidates: Sequence[SearchResult],
    query: str,
    match_bonus: float,
    mismatch_penalty: float,
) -> list[SearchResult]:
    fallback = load_content_config().fragetyp_fallback

    query_intent = detect_intent(query)

    reranked: list[SearchResult] = []

    for candidate in candidates:

        candidate_intent = candidate.metadata.get(
            "question_type",
            fallback,
        )

        score = candidate.score

        if candidate_intent == query_intent:
            score += match_bonus

        elif (
            query_intent != fallback
            and candidate_intent != fallback
        ):
            score -= mismatch_penalty

        reranked.append(
            SearchResult(
                id=candidate.id,
                content=candidate.content,
                score=score,
                metadata=candidate.metadata,
            )
        )

    reranked.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return reranked


async def main() -> None:

    settings = get_settings()

    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()

    answerability_gate = AnswerabilityGate(
        min_score=settings.RAG_MIN_SCORE,
        min_score_gap=settings.RAG_MIN_SCORE_GAP,
    )

    rag_service = RagService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        answerability_gate=answerability_gate,
    )

    rows = load_cases(GOLDEN_CSV)

    cached_results: list[CachedCase] = []

    print("Loading retrieval results...")

    for row in rows:

        question = row["frage"]
        expected = row["erwartete_entry_id"].strip()

        retrieval_result = await rag_service.quick_retrieve(
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

    results: list[SweepResult] = []

    match_bonus_candidates = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ]

    mismatch_penalty_candidates = [
        0.00,
        0.02,
        0.05,
        0.10,
    ]

    score_candidates = [
        round(x * 0.05, 2)
        for x in range(4, 19)
    ]

    gap_candidates = [
        round(x * 0.01, 2)
        for x in range(0, 31)
    ]

    total_combinations = (
        len(match_bonus_candidates)
        * len(mismatch_penalty_candidates)
        * len(score_candidates)
        * len(gap_candidates)
    )

    current_combination = 0

    for match_bonus in match_bonus_candidates:

        for mismatch_penalty in mismatch_penalty_candidates:

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
                        min_score=min_score,
                        min_score_gap=min_gap,
                    )

                    gate = AnswerabilityGate(
                        min_score=min_score,
                        min_score_gap=min_gap,
                    )

                    for case in cached_results:
                        reranked: list[SearchResult] = (
                            rerank_candidates(
                                candidates=case.chroma_results,
                                query=case.question,
                                match_bonus=match_bonus,
                                mismatch_penalty=mismatch_penalty,
                            )
                        )

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

    print("")
    print("\nTop 20 configurations")
    print("=" * 80)

    for r in ranked[-20:]:

        metric = f05(
            r.tp,
            r.fp,
            r.fn,
            r.wrong,
        )

        print(
            f"bonus={r.match_bonus:.2f} "
            f"penalty={r.mismatch_penalty:.2f} "
            f"score={r.min_score:.2f} "
            f"gap={r.min_score_gap:.2f} "
            f"f05={metric:.3f} "
            f"tp={r.tp} "
            f"wrong={r.wrong} "
            f"fp={r.fp} "
            f"fn={r.fn}"
        )


if __name__ == "__main__":
    asyncio.run(main())