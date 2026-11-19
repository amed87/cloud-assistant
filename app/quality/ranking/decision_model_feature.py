from dataclasses import dataclass, field

from app.config.content_settings import content_config
from app.quality.providers.decision.models import DecisionResult
from app.quality.ranking.ranking_feature import RankingFeature
from app.vectorstores.models import SearchResult
from app.quality.providers.decision.base import DecisionProvider


@dataclass(slots=True)
class DecisionModelFeature(RankingFeature):
    """
    Semantic Reranking Feature using a local System One decision model.
    """
    decision_provider: DecisionProvider
    
    # Cache results so the model is not called repeatedly for the same query
    # while scoring each candidate.
    _last_query: str | None = field(default=None, init=False)
    _cached_scores: dict[str, DecisionResult] | None = field(default=None, init=False)

    async def prime_cache(self, query: str, all_candidates: list[SearchResult]) -> None:
        """
        Populate the cache asynchronously for the complete candidate list.
        The async orchestrator must call this before the synchronous scoring loop.
        Idempotent per query — repeated primes for the same query are no-ops.
        """
        if self._last_query == query and self._cached_scores is not None:
            return
        self._last_query = query
        
        # Extract the candidate text used as a decision choice.
        candidate_texts = [cand.content for cand in all_candidates]
        
        if not candidate_texts:
            self._cached_scores = {}
            return
            
        # Make one asynchronous inference call for all candidates.
        self._cached_scores = await self.decision_provider.decide(
            query=query,
            questions=content_config.decision_model_prompts,
            choices=candidate_texts
        )

    def score(self, query: str, candidate: SearchResult) -> dict[str, float]:
        """
        Return the normalized probability (0.0 to 1.0) for this candidate.
        """
        if self._last_query != query or self._cached_scores is None:
            return 0.0


        return { name: self._cached_scores[name].probabilities[candidate.content] for name in self._cached_scores.keys() } # output anpassen
