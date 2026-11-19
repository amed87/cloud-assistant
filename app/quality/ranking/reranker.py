from app.vectorstores.models import SearchResult
from app.quality.ranking.ranking_feature import RankingFeature
from app.quality.ranking.decision_model_feature import DecisionModelFeature
from app.quality.ranking.dm_single_feature import DMSingleFeature

class Reranker:
    """Apply ranking features to retrieval results and retain their raw scores."""

    def __init__(
        self,
        features: list[RankingFeature],
        dm_feature: DecisionModelFeature | None = None
    ) -> None:
        # Ein Feature ist ausschließlich über sein Gewicht an/aus: weight == 0
        # trägt nichts zum Score bei, wird aber nicht mehr aufgerufen (kein
        # DM-Call, kein Priming). So bleibt die Aktivierung pro Feature eine
        # reine Konfigurationsfrage.
        self._features = [feature for feature in features if feature.weight != 0]
        self._dm_feature = dm_feature

    async def rerank(
        self,
        candidates: list[SearchResult],
        query: str,
        query_embedding: list[float] | None = None,
    ) -> tuple[list[SearchResult], dict[str, dict[str, float]]]:
        """Score and sort candidates, returning per-feature scores by candidate ID."""
        boosted_results = []
        candidate_feature_scores: dict[str, dict[str, float]] = {}

        # Resolve asynchronous dependencies before the synchronous scoring loop.
        # DMSingleFeatures share one parent call; facet features share one
        # classification pass — both are idempotent per query.
        for feature in self._features:
            await feature.prime(query, query_embedding)
        if (
            self._dm_feature is not None
            and any(isinstance(feature, DMSingleFeature) for feature in self._features)
        ):
            await self._dm_feature.prime_cache(query, candidates)

        for candidate in candidates:
            final_score = candidate.score
            feature_breakdown: dict[str, float] = {"base": candidate.score}

            for feature in self._features:
                raw_score = feature.score(query=query, candidate=candidate)
                weighted_score = feature.weight * raw_score
                final_score += weighted_score
                feature_name = (
                    getattr(feature, "name", None)
                    or feature.__class__.__name__
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