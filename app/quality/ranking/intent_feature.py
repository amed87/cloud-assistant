from app.quality.filters.intent_filter import filterIntent
from app.quality.ranking.ranking_feature import RankingFeature
from app.vectorstores.models import SearchResult
from dataclasses import dataclass


@dataclass(slots=True)
class IntentFeature(RankingFeature):
    bonus: float = 0.15
    penalty: float = 0.1
    fallback: str = 'Generell'

    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:
        query_intent = filterIntent(query)
        candidate_intent = candidate.metadata.get(
            "question_type",
            self.fallback,
        )

        if candidate_intent == query_intent:
            return self.bonus

        if (
            query_intent != self.fallback
            and candidate_intent != self.fallback
        ):
            return -self.penalty

        return 0.0