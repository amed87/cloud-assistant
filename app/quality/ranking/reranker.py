from app.vectorstores.models import SearchResult
from app.quality.ranking.ranking_feature import RankingFeature
from app.quality.ranking.decision_model_feature import DecisionModelFeature
from app.quality.ranking.dm_single_feature import DMSingleFeature

class Reranker:

    def __init__(
        self,
        features: list[RankingFeature],
        dm_feature: DecisionModelFeature | None = None
    ) -> None:
        self._features = features
        self._dm_feature = dm_feature

    async def rerank(
        self,
        candidates: list[SearchResult],
        query: str,
    ) -> tuple[list[SearchResult], dict[str, dict[str, float]]]:
        """Rerank the candidates based on the intent of the query."""
        boosted_results = []
        candidate_feature_scores: dict[str, dict[str, float]] = {}

        for feature in self._features:
            if isinstance(feature, DMSingleFeature) and self._dm_feature is not None:
                # Hier lösen wir den async Aufruf an Ollama auf, BEVOR wir in die synchrone Schleife gehen
                await self._dm_feature.prime_cache(query, candidates)
                break

        for candidate in candidates:
            final_score = candidate.score
            feature_breakdown: dict[str, float] = {"base": candidate.score}

            for feature in self._features:
                raw_score = feature.score(query=query, candidate=candidate)
                weighted_score = feature.weight * raw_score
                final_score += weighted_score
                feature_name = (
                    feature.name
                    if isinstance(feature, DMSingleFeature)
                    else feature.__class__.__name__
                )
                feature_breakdown[feature_name] = raw_score

            boosted_results.append(
                SearchResult(
                    id=candidate.id,
                    content=candidate.content,
                    score=final_score,
                    metadata=candidate.metadata,
                )
            )
            candidate_feature_scores[candidate.id] = feature_breakdown

        # Sort the candidates by the boosted score in descending order
        boosted_results.sort(key=lambda x: x.score, reverse=True)

        return boosted_results, candidate_feature_scores