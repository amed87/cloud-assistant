"""Evaluate the RAG FAQ chatbot against the golden dataset.

Runs against the real RagService and reports:
- Score and answer for each golden case, including cases below the threshold
- Error classification for each case at multiple levels (retrieval, reranker, gate, overall)
- Recommended values for RAG_MIN_SCORE and RAG_MIN_SCORE_GAP

Run with: python -m evaluation.evaluate_rag
"""
import argparse
import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.vectorstores.chroma import ChromaVectorStore
from app.services.rag_service import RagService
from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.quality.models import RetrievalResult, AnswerabilityDecision
from app.quality.ranking.reranker import Reranker
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.keyword_feature import KeywordFeature

GOLDEN_CSV = Path(__file__).resolve().parents[2] / "eval" / "data" / "golden_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "eval" / "output" / f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

# Known, accepted limitations of the current architecture.
KNOWN_FALSE_POSITIVES = {"fast_treffer"}

@dataclass
class CaseResult:
    question: str
    category: str
    expected_entry: str
    
    # Retrieval metrics (ChromaDB) - skalare Werte, keine Listen
    chroma_top_score: float = 0.0
    chroma_entry_id: str | None = None
    chroma_margin: float | None = None
    retrieval_verdict: str | None = None
    retrieval_hit_top3: bool = False
    retrieval_hit_top5: bool = False
    retrieval_hit_top10: bool = False
    
    # Reranker metrics (Intent) - skalare Werte, keine Listen
    reranker_top_score: float = 0.0
    reranker_entry_id: str | None = None
    reranker_margin: float | None = None
    reranker_verdict: str | None = None
    
    # Gate metrics (Answerability) - ohne Reranker
    gate_chroma_accepted: bool = False
    gate_chroma_reason: str = ""
    gate_chroma_confidence: float = 0.0
    gate_chroma_entry_id: str | None = None
    gate_chroma_verdict: str | None = None

    # Gate metrics (Answerability) - mit Reranker
    gate_reranker_accepted: bool = False
    gate_reranker_reason: str = ""
    gate_reranker_confidence: float = 0.0
    gate_reranker_entry_id: str | None = None
    gate_reranker_verdict: str | None = None
    
    # Overall result
    overall_verdict: str | None = None
    
    # Legacy compatibility fields
    score: float = 0.0
    second_score: float | None = None
    margin: float | None = None
    entry_id: str | None = None
    answer: str = ""
    verdict: str | None = None

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write full results to CSV file"
    )
    return parser.parse_args()

def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def classify_retrieval(case_result: CaseResult, expected_entry: str) -> str | None:
    """Classify retrieval-level correctness.
    
    Returns None if correct.
    Tracks false positives when no entry is expected but something was retrieved.
    """
    expected = expected_entry.strip()
    
    if not expected:
        # No match expected - track false positive if anything was retrieved
        if case_result.chroma_entry_id:
            return "false_positive"
        return None
    
    # Match expected
    if not case_result.chroma_entry_id:
        return "retrieval_miss"
    
    if case_result.chroma_entry_id != expected:
        return "retrieval_wrong_entry"
    
    return None

def classify_reranker(case_result: CaseResult, expected_entry: str) -> str | None:
    """Classify reranker-level correctness.
    
    Returns None if correct.
    Tracks false positives when no entry is expected but something was reranked.
    """
    expected = expected_entry.strip()
    
    if not expected:
        # No match expected - track false positive if anything was reranked
        if case_result.reranker_entry_id:
            return "false_positive"
        return None
    
    # Match expected
    if not case_result.reranker_entry_id:
        return "reranker_miss"
    
    if case_result.reranker_entry_id != expected:
        return "reranker_wrong_entry"
    
    return None

def classify_gate(
    accepted: bool,
    entry_id: str | None,
    expected_entry: str,
) -> str | None:
    """Classify gate-level decision based on stored gate result.
    
    Uses the ACTUAL gate decision from RetrievalResult (computed with original thresholds).
    Does NOT simulate new thresholds - that's done in the sweep separately.
    """
    expected = expected_entry.strip()
    
    if not expected:
        # No match expected - gate should reject
        if accepted:
            return "false_positive"
        return None
    
    # Match expected - gate should accept
    if not accepted:
        return "fallback_wo_hit_expected"
    
    # Gate accepted - check if correct entry was returned
    if entry_id and entry_id != expected:
        return "wrong_entry"
    
    return None

def classify_overall(case_result: CaseResult, expected_entry: str) -> str | None:
    """Classify overall end-to-end correctness."""
    expected = expected_entry.strip()
    
    if not expected:
        # No match expected
        if case_result.gate_reranker_accepted:
            return "false_positive"
        return None
    
    # Match expected
    if not case_result.gate_reranker_accepted:
        return "overall_miss"
    
    if case_result.gate_reranker_entry_id and case_result.gate_reranker_entry_id != expected:
        return "wrong_entry"
    
    return None

def write_results_to_csv(results: list[CaseResult], path: Path) -> None:
    """Writes all results to CSV file with extended columns for multi-level evaluation."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "category", "score", "second_score",
            "margin", "entry_id", "answer", "expected_entry", "verdict",
            "retrieval_verdict", "reranker_verdict", "gate_chroma_verdict","gate_reranker_verdict", "overall_verdict",
            "chroma_top_score", "chroma_entry_id", "chroma_margin",
            "chroma_top_3", "chroma_top_5", "chroma_top_10",
            "reranker_top_score", "reranker_entry_id", "reranker_margin",
            "gate_chroma_confidence", "gate_chroma_accepted", "gate_chroma_reason", "gate_chroma_entry_id",
            "gate_reranker_confidence", "gate_reranker_accepted", "gate_reranker_reason"
        ])
        
        for r in results:
            margin_str = f"{r.margin:.3f}" if r.margin is not None else ""
            chroma_margin_str = f"{r.chroma_margin:.3f}" if r.chroma_margin is not None else ""
            reranker_margin_str = f"{r.reranker_margin:.3f}" if r.reranker_margin is not None else ""
            answer_str = (r.answer or "").replace("\n", " ").replace(",", ";")[:200]
            
            writer.writerow([
                r.question,
                r.category,
                f"{r.score:.3f}",
                f"{r.second_score:.3f}" if r.second_score is not None else "",
                margin_str,
                r.entry_id or "",
                answer_str,
                r.expected_entry,

                r.verdict or "",
                r.retrieval_verdict or "",
                r.reranker_verdict or "",
                r.gate_chroma_verdict or "",
                r.gate_reranker_verdict or "",
                r.overall_verdict or "",

                f"{r.chroma_top_score:.3f}",
                r.chroma_entry_id or "",
                chroma_margin_str,
                r.retrieval_hit_top3,
                r.retrieval_hit_top5,
                r.retrieval_hit_top10,

                f"{r.reranker_top_score:.3f}",
                r.reranker_entry_id or "",
                reranker_margin_str,

                f"{r.gate_chroma_confidence:.3f}",
                r.gate_chroma_accepted,
                r.gate_chroma_reason,
                r.gate_chroma_entry_id or "",

                f"{r.gate_reranker_confidence:.3f}",
                r.gate_reranker_accepted,
                r.gate_reranker_reason,
            ])

def get_gate_document(result: RetrievalResult):
    return result.gate.document 

def get_gate_entry_id(result: RetrievalResult):
    return result.gate.document.id.partition(":")[0] if result.gate.document else None


async def main() -> None:
    settings = get_settings()
    content_config = load_content_config()
    
    # Initialize components for RagService
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    reranker_features = [        
        IntentFeature(
            weight=settings.INTENT_WEIGHT, 
            bonus=settings.INTENT_MATCH_BONUS,
            penalty=settings.INTENT_MISMATCH_PENALTY,
            fallback=content_config.fragetyp_fallback,
            ),
        KeywordFeature(
            weight=settings.KEYWORD_WEIGHT,
            normalization=settings.KEYWORD_MATCH_NORMALIZATION,
        )
    ] 
    reranker = Reranker(features=reranker_features)
    
    
    # Initialize AnswerabilityGate with current settings
    answerability_gate = AnswerabilityGate(
        min_score=settings.RAG_MIN_SCORE,
        min_score_gap=settings.RAG_MIN_SCORE_GAP,
    )
    
    # Create RagService instance
    rag_service = RagService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        answerability_gate=answerability_gate,
    )

    args = parse_args()
    output_path = args.output or OUTPUT_PATH
    rows = load_cases(GOLDEN_CSV)

    # First pass: Run all retrievals once and collect results
    case_results: list[CaseResult] = []
    for row in rows:
        question = row["frage"]
        expected_entry = row["erwartete_entry_id"].strip()
        
        # Use RagService.quick_retrieve with eval=True to get full RetrievalResult
        retrieval_result: RetrievalResult = await rag_service.quick_retrieve(
            question=question,
            eval=True,
        )
        chroma_entry_ids = [
            candidate.id.partition(":")[0]
            for candidate in retrieval_result.chroma
        ]

        # Use the gate directly on the chroma retrieval results without reranker
        chroma_gate = AnswerabilityGate(
            min_score=settings.RAG_MIN_SCORE,
            min_score_gap=settings.RAG_MIN_SCORE_GAP,
        ).evaluate_quick(
            question,
            retrieval_result.chroma,
        )

        # Extract scalars from RetrievalResult (no lists stored)
        best_chroma = retrieval_result.chroma[0] if retrieval_result.chroma else None
        best_reranker = retrieval_result.reranker[0] if retrieval_result.reranker else None
        second_reranker = retrieval_result.reranker[1] if len(retrieval_result.reranker) > 1 else None
        gate_reranker_doc = get_gate_document(retrieval_result)
        gate_reranker_entry_id = get_gate_entry_id(retrieval_result)
        gate_chroma_entry_id = (chroma_gate.document.id.partition(":")[0] if chroma_gate.document else None)
        
        case_result = CaseResult(
            question=question,
            category=row["kategorie"],
            expected_entry=expected_entry,
            
            # Chroma retrieval - scalars only
            chroma_top_score=best_chroma.score if best_chroma else 0.0,
            chroma_entry_id=best_chroma.id.partition(":")[0] if best_chroma else None,
            chroma_margin=(
                (best_chroma.score - retrieval_result.chroma[1].score)
                if best_chroma and len(retrieval_result.chroma) > 1 else None
            ),
            retrieval_hit_top3=(
                expected_entry in chroma_entry_ids[:3]
                if expected_entry else False
            ),

            retrieval_hit_top5=(
                expected_entry in chroma_entry_ids[:5]
                if expected_entry else False
            ),

            retrieval_hit_top10=(
                expected_entry in chroma_entry_ids[:10]
                if expected_entry else False
            ),
            
            # Reranker - scalars only
            reranker_top_score=best_reranker.score if best_reranker else 0.0,
            reranker_entry_id=best_reranker.id.partition(":")[0] if best_reranker else None,
            reranker_margin=(
                (best_reranker.score - second_reranker.score)
                if best_reranker and second_reranker else None
            ),
            
            # Gate without reranker - scalars only
            gate_chroma_accepted=chroma_gate.accepted,
            gate_chroma_reason=chroma_gate.reason,
            gate_chroma_confidence=chroma_gate.confidence,
            gate_chroma_entry_id=gate_chroma_entry_id,

            # Gate with reranker - scalars only
            gate_reranker_accepted=retrieval_result.gate.accepted,
            gate_reranker_reason=retrieval_result.gate.reason,
            gate_reranker_confidence=retrieval_result.gate.confidence,
            gate_reranker_entry_id=gate_reranker_entry_id,
            
            # Overall
            score=retrieval_result.gate.confidence,
            second_score=second_reranker.score if second_reranker else None,
            entry_id=gate_reranker_entry_id,
            answer=(gate_reranker_doc.metadata.get("answer") if gate_reranker_doc else "") or "",
        )
        
        # Store margin for overall field
        case_result.margin = case_result.score - case_result.second_score if case_result.second_score else None
        
        # Classify each level (ORIGINAL thresholds)
        case_result.retrieval_verdict = classify_retrieval(case_result, expected_entry) or "ok"
        case_result.reranker_verdict = classify_reranker(case_result, expected_entry) or "ok"
        case_result.gate_reranker_verdict = classify_gate(
            case_result.gate_reranker_accepted, case_result.gate_reranker_entry_id, expected_entry) or "ok"
        case_result.gate_chroma_verdict = classify_gate(
                    case_result.gate_chroma_accepted, case_result.gate_chroma_entry_id, expected_entry) or "ok"
        
        # Handle known limitations
        if case_result.gate_reranker_verdict == "false_positive" and row.get("kategorie") in KNOWN_FALSE_POSITIVES:
            case_result.gate_reranker_verdict = "false_positive (known limitation)"
        
        case_result.overall_verdict = classify_overall(case_result, expected_entry) or "ok"
        case_result.verdict = case_result.overall_verdict
        
        case_results.append(case_result)

    # Print detailed results table
    print(f"\n{'Question':<35} {'Category':<20} {'Retrieval':>10} {'Reranker':>10} {'Gate':>10}  Overall")
    print("-" * 110)
    for cr in case_results:
        print(f"{cr.question[:33]:<35} "
              f"{cr.category[:18]:<20} "
              f"{cr.retrieval_verdict[:8]:>10} {cr.reranker_verdict[:8]:>10} "
              f"{cr.gate_reranker_verdict[:8]:>10}  {cr.overall_verdict[:10]}")

    # Write full results to CSV
    write_results_to_csv(case_results, output_path)
    print(f"\nDetailed results written to: {output_path}")

    # Summary statistics
    total_cases = len(case_results)
    expected_with_match = sum(1 for c in case_results if c.expected_entry.strip())
    expected_no_match = total_cases - expected_with_match
    
    retrieval_hits = sum(1 for c in case_results if c.retrieval_verdict == "ok")
    reranker_hits = sum(1 for c in case_results if c.reranker_verdict == "ok")
    gate_chroma_hits = sum(1 for c in case_results if c.gate_chroma_verdict == "ok")
    gate_reranker_hits = sum(1 for c in case_results if c.gate_reranker_verdict == "ok")
    overall_hits = sum(1 for c in case_results if c.overall_verdict == "ok")
    answerable_cases = [
        c
        for c in case_results
        if c.expected_entry.strip()
    ]
    answerable_count = len(answerable_cases)
    top3_hits = sum(
        1
        for c in answerable_cases
        if c.retrieval_hit_top3
    )
    top5_hits = sum(
        1
        for c in answerable_cases
        if c.retrieval_hit_top5
    )
    top10_hits = sum(
        1
        for c in answerable_cases
        if c.retrieval_hit_top10
    )

    print(f"\n{'='*70}")
    print("SUMMARY BY EVALUATION LEVEL")
    print(f"{'='*70}")
    print(f"Total test cases: {total_cases}")
    print(f"Cases expecting match: {expected_with_match}")
    print(f"Cases expecting no match: {expected_no_match}\n")
    
    print(f"{'Level':<20} {'Hits':<10} {'Rate':<10}")
    print("-" * 40)
    print(f"{'Retrieval (Chroma)':<20} {retrieval_hits:<10} {retrieval_hits/total_cases:.2%}")
    print(f"{'Reranker (Intent)':<20} {reranker_hits:<10} {reranker_hits/total_cases:.2%}")
    print(f"{'Gate (Chroma)':<20} {gate_chroma_hits:<10} {gate_chroma_hits/total_cases:.2%}")
    print(f"{'Gate (Reranker)':<20} {gate_reranker_hits:<10} {gate_reranker_hits/total_cases:.2%}")
    print(f"{'Overall (End-to-End)':<20} {overall_hits:<10} {overall_hits/total_cases:.2%}")


    print(f"\nRETRIEVAL COVERAGE")
    print("-" * 40)
    print(
        f"Top-3:  "
        f"{top3_hits:>3}"
        f" ({top3_hits/answerable_count:.2%})"
    )
    print(
        f"Top-5:  "
        f"{top5_hits:>3}"
        f" ({top5_hits/answerable_count:.2%})"
    )
    print(
        f"Top-10: "
        f"{top10_hits:>3}"
        f" ({top10_hits/answerable_count:.2%})"
    )
    print(f"\n{'='*80}")
    print("RERANKER ANALYSIS")
    print("=" * 80)

    for cr in case_results:

        if cr.reranker_verdict == "ok":
            continue

        print(
            f"\nQ: {cr.question}"
        )

        print(
            f"Expected: {cr.expected_entry}"
        )

        print(
            f"Chroma:   {cr.chroma_entry_id}"
        )

        print(
            f"Reranker: {cr.reranker_entry_id}"
        )

        print(
            f"Verdict:  {cr.reranker_verdict}"
        )

if __name__ == "__main__":
    asyncio.run(main())