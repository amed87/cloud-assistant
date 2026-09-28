from app.vectorstores.models import SearchResult
from app.answerability.models import AnswerabilityDecision

class AnswerabilityGate:
    def __init__(
        self,
        min_score: float,
        min_score_gap: float,
    ) -> None:
        self.min_score = min_score
        self.min_score_gap = min_score_gap

    def evaluate_quick(
        self,
        question: str,
        documents: list[SearchResult],
    ) -> AnswerabilityDecision:
        if not documents:
            return AnswerabilityDecision(
                accepted=False,
                reason="no_candidates",
                confidence=0.0,
            )

        best = documents[0]
        second_score = documents[1].score if len(documents) > 1 else 0.0
        margin = best.score - second_score

        if best.score < self.min_score:
            return AnswerabilityDecision(
                accepted=False,
                reason="score_below_threshold",
                confidence=best.score,
                document=best,
            )

        if len(documents) > 1 and margin < self.min_score_gap:
            return AnswerabilityDecision(
                accepted=False,
                reason="ambiguous_candidates",
                confidence=margin,
                document=best,
            )

        if not (best.metadata.get("answer") or "").strip():
            return AnswerabilityDecision(
                accepted=False,
                reason="missing_answer",
                confidence=best.score,
                document=best,
            )

        return AnswerabilityDecision(
            accepted=True,
            reason="accepted",
            confidence=best.score,
            document=best,
        )