from app.quality.intent.intent_filter import filterIntent
from app.vectorstores.models import SearchResult
from app.config.content_settings import content_config
from app.config.settings import get_settings
from app.quality.intent.intent_feature import IntentFeature

intent_feature = IntentFeature()

def rerank_candidates_by_intent(candidates: list[SearchResult], query: str) -> list[SearchResult]:
    """Rerank the candidates based on the intent of the query."""
    boosted_results = []
    for candidate in candidates:
        intent_score = intent_feature.score(
            query=query,
            candidate=candidate,
        )

        boosted_results.append(
            SearchResult(
                id=candidate.id,
                content=candidate.content,
                score=candidate.score + intent_score,
                metadata=candidate.metadata,
            )
        )

    # Sort the candidates by the boosted score in descending order
    boosted_results.sort(key=lambda x: x.score, reverse=True)

    return boosted_results