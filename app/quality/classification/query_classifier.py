"""Cascade-Classifier für deklarierte Facetten.

Ablauf pro Query (einmal pro Anfrage, ``prime``):
    1. Cosine der Query-Embedding gegen alle Label-Zentroide je Facette.
    2. Confident, wenn Top1 >= floor und (Top1 - Top2) >= margin.
    3. Unsichere Facetten: optional EIN Decision-Model-Call je Facette,
       der die Query in die Labels der Facette klassifiziert (choices sind
       die Labels, nicht Kandidatenfragen — deutlich billiger als der
       klassische Kandidaten-DM-Call).
    4. Ohne DM-Fallback oder bei Fehlern: das Facetten-Fallback-Label.

Klassifizierte Labels konsumiert ``CentroidMatchFeature`` synchron.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.config.content_settings import CentroidFacet
from app.quality.classification.centroid_store import CentroidStore, LabelPrediction
from app.quality.providers.decision.base import DecisionProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FacetClassifierConfig:
    """Cascade schedule parameters (from settings at assembly time)."""

    floor: float          # necessary cosine similarity to the top label
    margin: float         # necessary distance Top1 - Top2
    dm_fallback: bool     # uncertain facets: classification via decision model


class QueryFacetClassifier:
    """Classifies the query into facet labels; centroids first, DM on demand."""

    def __init__(
        self,
        store: CentroidStore,
        facets: dict[str, CentroidFacet],
        config: FacetClassifierConfig,
        decision_provider: DecisionProvider | None = None,
    ) -> None:
        self._store = store
        self._facets = facets
        self._config = config
        self._provider = decision_provider
        self._last_query: str | None = None
        self._predictions: dict[str, LabelPrediction] = {}

    # ------------------------------------------------------------ Zugriff

    @property
    def facets(self) -> dict[str, CentroidFacet]:
        """Declared facets (diagnostics/factory default)."""
        return self._facets

    def prediction(self, facet: str) -> LabelPrediction | None:
        """Prediction of the facet from the last primed query (None if unused)."""
        return self._predictions.get(facet)

    @property
    def predictions(self) -> dict[str, LabelPrediction]:
        """All predictions of the last primed query (diagnostics/tests)."""
        return dict(self._predictions)

    # ------------------------------------------------------------- Cascade

    async def prime(self, query: str, query_embedding: list[float] | None) -> None:
        """Classify the query for all declared facets; idempotent per query+embedding."""
        if self._last_query == query and self._predictions:
            return
        self._last_query = query
        self._predictions = {}

        if not query_embedding:
            logger.info("No query embedding provided; facet classification skipped.")
            return

        uncertain: list[tuple[str, CentroidFacet, list[str]]] = []
        for name, facet in self._facets.items():
            labels = self._store.labels(name)
            if not labels:
                self._predictions[name] = LabelPrediction(
                    label=facet.fallback or None, confidence=0.0, source="fallback",
                )
                continue

            sims = [
                (label, _cosine(query_embedding, vector))
                for label, vector in labels.items()
            ]
            sims.sort(key=lambda pair: pair[1], reverse=True)
            top_label, top_sim = sims[0]
            second_sim = sims[1][1] if len(sims) > 1 else 0.0

            if top_sim >= self._config.floor and (top_sim - second_sim) >= self._config.margin:
                self._predictions[name] = LabelPrediction(
                    label=top_label, confidence=top_sim, source="centroid",
                )
            else:
                uncertain.append((name, facet, list(labels)))

        if uncertain:
            await self._classify_uncertain(query, uncertain)

    async def _classify_uncertain(
        self,
        query: str,
        uncertain: list[tuple[str, CentroidFacet, list[str]]],
    ) -> None:
        """DM fallback for uncertain facets — one call per facet (different choices)."""
        config = self._config
        for name, facet, labels in uncertain:
            if not (config.dm_fallback and self._provider):
                self._predictions[name] = LabelPrediction(
                    label=facet.fallback or None, confidence=0.0, source="fallback",
                )
                continue

            try:
                results = await self._provider.decide(
                    query=query,
                    questions={name: facet.dm_prompt or f"Which {name} matches the user query?"},
                    choices=labels,
                )
            except Exception as ex:
                logger.warning("Facet DM fallback failed for %r (%s); using fallback label.", name, ex)
                self._predictions[name] = LabelPrediction(
                    label=facet.fallback or None, confidence=0.0, source="fallback",
                )
                continue

            result = results.get(name)
            if result is None or not result.probabilities:
                self._predictions[name] = LabelPrediction(
                    label=facet.fallback or None, confidence=0.0, source="fallback",
                )
                continue

            label, confidence = max(result.probabilities.items(), key=lambda kv: kv[1])
            self._predictions[name] = LabelPrediction(
                label=label, confidence=confidence, source="decision_model",
            )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)
