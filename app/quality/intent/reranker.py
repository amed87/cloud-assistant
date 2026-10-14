from app.quality.intent.intent_filter import filterIntent
from app.vectorstores.models import SearchResult
from app.config.content_settings import content_config
from app.config.settings import get_settings
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.ranking_feature import RankingFeature

def rerank_candidates(candidates: list[SearchResult], query: str) -> list[SearchResult]:
    """Rerank the candidates based on the intent of the query."""
    settings = get_settings()
    boosted_results = []
    features: list[RankingFeature] = [IntentFeature()]

    for candidate in candidates:
        final_score = candidate.score

        for feature in features:
            final_score += (
                settings.INTENT_WEIGHT * feature.score(query=query, candidate=candidate)
                )

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