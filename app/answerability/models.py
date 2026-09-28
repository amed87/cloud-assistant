from dataclasses import dataclass
from app.vectorstores.models import SearchResult


@dataclass(frozen=True)
class AnswerabilityDecision:
    """Result of the quick-chat answerability check.

    ``confidence`` is a diagnostic value, not a calibrated probability.
    It contains the best candidate score for most reasons, but contains the
    score gap between the best two candidates for ``ambiguous_candidates``.
    It does not affect the decision itself.
    """

    accepted: bool
    reason: str
    confidence: float
    document: SearchResult | None = None