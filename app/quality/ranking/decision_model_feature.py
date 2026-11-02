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
    
    # Ein interner Cache verhindert, dass bei einer Schleife über 
    # alle Kandidaten das Modell mehrfach für dieselbe Query aufgerufen wird.
    _last_query: str | None = field(default=None, init=False)
    _cached_scores: dict[str, DecisionResult] | None = field(default=None, init=False)

    async def prime_cache(self, query: str, all_candidates: list[SearchResult]) -> None:
        """
        Befüllt den Cache asynchron für die gesamte Liste an Kandidaten.
        MUSS im asynchronen Orchestrator vor der synchronen Scoring-Schleife aufgerufen werden.
        """
        self._last_query = query
        
        # Extrahieren Sie hier das Textfeld, das Ihre Frage enthält (z.B. cand.question oder cand.text)
        candidate_texts = [cand.content for cand in all_candidates]
        
        if not candidate_texts:
            self._cached_scores = {}
            return
            
        # Ein einziger, asynchroner CPU-Inference-Aufruf für das gesamte Feld
        self._cached_scores = await self.decision_provider.decide(
            query=query,
            questions=content_config.decision_model_prompts,
            choices=candidate_texts
        )

    def score(self, query: str, candidate: SearchResult) -> dict[str, float]:
        """
        Gibt den normalisierten Wahrscheinlichkeitswert (0.0 bis 1.0) 
        für den spezifischen Kandidaten zurück.
        """
        if self._last_query != query or self._cached_scores is None:
            return 0.0


        return { name: self._cached_scores[name].probabilities[candidate.content] for name in self._cached_scores.keys() } # output anpassen
