from dataclasses import dataclass, field

@dataclass(frozen=True)
class DecisionResult:
    probabilities: dict[str, float]
    confidence: float