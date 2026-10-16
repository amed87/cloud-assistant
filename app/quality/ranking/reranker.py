from app.quality.filters.intent_filter import filterIntent
from app.vectorstores.models import SearchResult
from app.config.content_settings import content_config
from app.config.settings import get_settings
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.ranking_feature import RankingFeature
from app.quality.ranking.keyword_feature import KeywordFeature

class Reranker:

    def __init__(
        self,
        features: list[RankingFeature],
    ) -> None:
        self._features = features

    def rerank(
        self,
        candidates: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        """Rerank the candidates based on the intent of the query."""
        boosted_results = []

        for candidate in candidates:
            final_score = candidate.score

            for feature in self._features:
                final_score += feature.weight * feature.score(query=query, candidate=candidate)

            boosted_results.append(
                SearchResult(
                    id=candidate.id,
                    content=candidate.content,
                    score=final_score,
                    metadata=candidate.metadata,
                )
            )

        # Sort the candidates by the boosted score in descending order
        boosted_results.sort(key=lambda x: x.score, reverse=True)

        return boosted_results