"""Bewertung des RAG-FAQ-Chatbots gegen das Golden Dataset.

Läuft gegen echte ChromaDB + Ollama und liefert:
- Score + Antwort je Golden-Case (auch unterhalb des Thresholds)
- Fehlertyp-Klassifikation pro Fall
- gemeinsame Empfehlung für RAG_MIN_SCORE und RAG_MIN_SCORE_GAP

Aufruf: python -m evaluation.evaluate_rag
"""

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path

from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore

GOLDEN_CSV = Path("data/golden_dataset.csv")

# Bekannte, akzeptierte Limitierungen der aktuellen Architektur.
# Diese Fälle sind als false_positive KORREKT klassifiziert – sie zeigen,
# was ein reiner Threshold-Ansatz nicht leisten kann (Beantwortbarkeit).
KNOWN_FALSE_POSITIVES = {"fast_treffer"}

@dataclass
class CaseResult:
    question: str
    category: str
    score: float               # 0.0, wenn kein Kandidat gefunden wurde
    second_score: float | None  # Score des Zweitplatzierten (None, wenn nur einer)
    entry_id: str | None       # stabile FAQ-ID des besten Treffers
    answer: str                # FAQ-Antwort des Treffers oder ""
    expected_entry: str
    error_type: str | None = None  # siehe Klassifikation in classify()

    @property
    def margin(self) -> float | None:
        """Abstand zum Zweitplatzierten – klein = ambigue Entscheidung."""
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
    """Ordnet einem Ergebnis einen Fehlertyp zu (None = korrekt).

    Simuliert beide Answerability-Grenzen von quick_retrieve.
    """
    expected_entry = case_result.expected_entry.strip()
    ambiguous = (
        case_result.margin is not None
        and case_result.margin < min_score_gap
    )
    treffer = case_result.score >= min_score and not ambiguous

    if not expected_entry:                              # kein Treffer erwartet
        return None if not treffer else "false_positive"

    if not treffer:                                     # Treffer erwartet, unter Threshold
        return "fallback_wo_hit_expected"
    if case_result.entry_id != expected_entry:
        return "wrong_entry"
    return None                                         # inhaltliche Prüfung ggf. später

def sweep_threshold_and_gap(
    case_results: list[CaseResult],
    score_candidates: list[float],
    gap_candidates: list[float],
) -> list[tuple[float, float, int, int, int]]:
    """Für jedes Grenzwert-Paar: (min_score, gap, korrekte, falsche, false_positives).

    Verwendet die gemessenen Scores und simuliert beide Gate-Bedingungen.
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

    # Report: Score, Margin, getroffener Entry und simuliertes Verhalten
    print(f"\n{'Question':<42} {'Category':<30} {'Score':>6} {'Margin':>7} {'Entry':>5}  Finding")
    for case_result in case_results:
        verdict = classify(
            case_result,
            settings.RAG_MIN_SCORE,
            settings.RAG_MIN_SCORE_GAP,
            content_config.standard_fallback,
        )
        # Erwartete Limitierung markieren statt verschweigen
        if verdict == "false_positive" and case_result.category in KNOWN_FALSE_POSITIVES:
            verdict = "false_positive (bekannte Limitierung)"
        verdict = verdict or "ok"

        margin_str = f"{case_result.margin:.3f}" if case_result.margin is not None else "-"
        entry_str = case_result.entry_id if case_result.entry_id is not None else "-"
        print(f"{case_result.question[:40]:<40} "
              f"{case_result.category:<26} "
              f"{case_result.score:>6.3f} {margin_str:>7} {entry_str:>5}  {verdict}")

    # Gemeinsame Grenzwert-Suche; Gaps reichen bis über den größten Messwert.
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