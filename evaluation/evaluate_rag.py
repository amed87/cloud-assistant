"""Evaluate the RAG FAQ chatbot against the golden dataset.

Runs against ChromaDB and Ollama and reports:
- Score and answer for each golden case, including cases below the threshold
- Error classification for each case
- Recommended values for RAG_MIN_SCORE and RAG_MIN_SCORE_GAP

Run with: python -m evaluation.evaluate_rag
"""

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path

from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore

GOLDEN_CSV = Path(__file__).resolve().parents[2] / "eval" / "data" / "golden_dataset.csv"

# Known, accepted limitations of the current architecture.
# These cases are correctly classified as false positives because a threshold
# alone cannot determine whether a question is answerable.
KNOWN_FALSE_POSITIVES = {"fast_treffer"}

@dataclass
class CaseResult:
    question: str
    category: str
    score: float               # 0.0 when no candidate is found.
    second_score: float | None  # Score of the runner-up, or None if absent.
    entry_id: str | None       # Stable FAQ ID of the best match.
    answer: str                # Answer from the matched FAQ entry, or "".
    expected_entry: str
    error_type: str | None = None  # siehe Klassifikation in classify()

    @property
    def margin(self) -> float | None:
        """Return the margin to the runner-up; a small margin is ambiguous."""
        if self.second_score is None:
            return None
        return self.score - self.second_score

def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def classify(
    case_result: CaseResult,
    min_score: float,
    min_score_gap: float,
    fallback: str,
) -> str | None:
    """Classify a result; None means it is correct.

    Simulate both answerability thresholds used by quick_retrieve.
    """
    expected_entry = case_result.expected_entry.strip()
    ambiguous = (
        case_result.margin is not None
        and case_result.margin < min_score_gap
    )
    treffer = case_result.score >= min_score and not ambiguous

    if not expected_entry:                              # No match is expected.
        return None if not treffer else "false_positive"

    if not treffer:                                     # A match is expected but below threshold.
        return "fallback_wo_hit_expected"
    if case_result.entry_id != expected_entry:
        return "wrong_entry"
    return None                                         # inhaltliche Prüfung ggf. später

def sweep_threshold_and_gap(
    case_results: list[CaseResult],
    score_candidates: list[float],
    gap_candidates: list[float],
) -> list[tuple[float, float, int, int, int]]:
    """For each threshold pair: (min_score, gap, correct, wrong, false positives).

    Use measured scores to simulate both gate conditions.
    """
    results = []
    for min_score in score_candidates:
        for min_score_gap in gap_candidates:
            correct = 0
            wrong_entries = 0
            false_positives = 0

            for case_result in case_results:
                ambiguous = (
                    case_result.margin is not None
                    and case_result.margin < min_score_gap
                )
                accepted = case_result.score >= min_score and not ambiguous
                expected_entry = case_result.expected_entry.strip()

                if not accepted:
                    continue
                if not expected_entry:
                    false_positives += 1
                elif case_result.entry_id == expected_entry:
                    correct += 1
                else:
                    wrong_entries += 1

            results.append((
                min_score,
                min_score_gap,
                correct,
                wrong_entries,
                false_positives,
            ))
    return results

async def main() -> None:
    settings = get_settings()
    content_config = load_content_config()
    provider = OllamaEmbeddingProvider()
    store = ChromaVectorStore()

    rows = load_cases(GOLDEN_CSV)

    case_results: list[CaseResult] = []
    for row in rows:
        question = row["frage"]          
        embedding = await provider.embed(question)
        documents = await store.search(embedding=embedding, limit=2)

        best = documents[0] if documents else None
        second = documents[1] if len(documents) > 1 else None
        case_results.append(CaseResult(
            question=question,
            category=row["kategorie"],
            score=best.score if best else 0.0,
            second_score=second.score if second else None,
            entry_id=best.id.partition(":")[0] if best else None,
            answer=(best.metadata.get("answer") or "").strip() if best else "",
            expected_entry=row["erwartete_entry_id"].strip(),
        ))

    # Report the score, margin, matched entry, and simulated behavior.
    print(f"\n{'Question':<42} {'Category':<30} {'Score':>6} {'Margin':>7} {'Entry':>5}  Finding")
    for case_result in case_results:
        verdict = classify(
            case_result,
            settings.RAG_MIN_SCORE,
            settings.RAG_MIN_SCORE_GAP,
            content_config.standard_fallback,
        )
        # Mark known limitations instead of hiding them.
        if verdict == "false_positive" and case_result.category in KNOWN_FALSE_POSITIVES:
            verdict = "false_positive (bekannte Limitierung)"
        verdict = verdict or "ok"

        margin_str = f"{case_result.margin:.3f}" if case_result.margin is not None else "-"
        entry_str = case_result.entry_id if case_result.entry_id is not None else "-"
        print(f"{case_result.question[:40]:<40} "
              f"{case_result.category:<26} "
              f"{case_result.score:>6.3f} {margin_str:>7} {entry_str:>5}  {verdict}")

    # Search thresholds jointly; gaps extend beyond the largest measured value.
    score_candidates = [round(x * 0.05, 2) for x in range(4, 19)]  # 0.20 ... 0.90
    max_margin = max(
        (case_result.margin or 0.0 for case_result in case_results),
        default=0.0,
    )
    gap_candidates = [
        round(index * 0.01, 2)
        for index in range(int(max(0.0, max_margin) * 100) + 2)
    ]
    swept = sweep_threshold_and_gap(
        case_results,
        score_candidates,
        gap_candidates,
    )
    best = max(
        swept,
        key=lambda result: (
            result[2],
            -result[3],
            -result[4],
            -result[0],
            -result[1],
        ),
    )
    print(f"\nRecommended thresholds: RAG_MIN_SCORE={best[0]:.2f}, "
          f"RAG_MIN_SCORE_GAP={best[1]:.2f} "
          f"(correct hits: {best[2]}, wrong entries: {best[3]}, "
          f"false positives: {best[4]})")

if __name__ == "__main__":
    asyncio.run(main())