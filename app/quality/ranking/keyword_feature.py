from app.quality.ranking.ranking_feature import RankingFeature
from app.vectorstores.models import SearchResult
from app.config.settings import get_settings
from dataclasses import dataclass

@dataclass(slots=True)
class KeywordFeature(RankingFeature):
    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:

        keywords = candidate.metadata.get(
            "keywords",
            [],
        )

        if not keywords:
            return 0.0

        query_lower = query.lower()

        matches = 0

        for keyword in keywords:

            if keyword.lower() in query_lower:
                matches += 1

        return min(
            matches / 5,
            1.0,
        )