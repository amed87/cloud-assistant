from dataclasses import dataclass 
from app.vectorstores.models import SearchResult
from app.quality.ranking.ranking_feature import RankingFeature
from app.quality.ranking.decision_model_feature import DecisionModelFeature

@dataclass(slots=True)
class DMSingleFeature(RankingFeature):
    name: str
    weight: float
    parent_feature: DecisionModelFeature

    def score(self, query: str, candidate: SearchResult) -> float:
        """Return the normalized probability (0.0 to 1.0) for this candidate."""
        # Delegate the calculation to the parent DecisionModelFeature.
        parent_scores = self.parent_feature.score(query, candidate)
        return parent_scores.get(self.name, 0.0)
