"""Generisches Reranker-Feature: Facetten-Label-Abgleich Query ↔ Kandidat.

Die query-seitige Facetten-Vorhersage kommt aus der Zentroid-Cascade
(``QueryFacetClassifier``), die Kandidaten-Seite aus den FAQ-Metadaten
(``metadata_key`` der Facette). Scoring-Semantik entspricht dem
Intent-Feature — ein Label auf beiden Seiten ist Bonus, abweichende
Labels Penalty, solange keine der beiden Seiten das Fallback-Label
trägt (Fallback heißt „unbekannt", nicht „falsch").

Multi-Label-Facetten (z. B. ``subjects``): Bonus, wenn das Query-Label
eines der Kandidaten-Labels ist; sonst Penalty, sofern der Kandidat
nicht das Fallback-Label trägt.

Das Feature wird im Reranker über sein ``weight``-Feld ein- bzw.
ausgeschaltet: weight == 0 → Feature wird erst gar nicht Instanziert
bzw. im Reranker verworfen.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.content_settings import CentroidFacet
from app.quality.classification.query_classifier import QueryFacetClassifier
from app.quality.ranking.ranking_feature import RankingFeature
from app.vectorstores.models import SearchResult


@dataclass(slots=True)
class CentroidMatchFeature(RankingFeature):
    """Bonus/Penalty für Facetten-Übereinstimmung zwischen Query und Kandidat."""

    classifier: QueryFacetClassifier
    facet: CentroidFacet
    facet_name: str
    bonus: float = 0.15
    penalty: float = 0.10

    @property
    def name(self) -> str:
        """Stable feature identifier (also used for the per-feature score breakdown)."""
        return f"Centroid:{self.facet_name}"

    async def prime(self, query: str, query_embedding: list[float] | None) -> None:
        """Cascade classification for the query (shared across all facet features)."""
        await self.classifier.prime(query, query_embedding)

    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:
        prediction = self.classifier.prediction(self.facet_name)
        if prediction is None or prediction.label is None:
            # Unavailable group (empty store, no embedding) — do not vote.
            return 0.0

        raw = candidate.metadata.get(self.facet.metadata_key)
        labels = self._candidate_labels(raw, self.facet)
        if not labels:
            # The candidate does not carry the metadata field — no judgment.
            return 0.0

        if prediction.label in labels:
            return self.bonus

        # Mirror IntentFeature: penalty only when neither side carries the
        # fallback label — fallback labels mean "unknown", not "wrong".
        if prediction.label != self.facet.fallback and self.facet.fallback not in labels:
            return -self.penalty

        return 0.0

    @staticmethod
    def _candidate_labels(raw, facet: CentroidFacet) -> list[str]:
        """Candidate's labels as a clean list, without empty items."""
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(value).strip() for value in raw if str(value).strip()]
        label = str(raw).strip()
        return [label] if label else []


def build_centroid_features(
    classifier: QueryFacetClassifier,
    weights: list[float],
    bonus: float,
    penalty: float,
    facets: dict[str, CentroidFacet] | None = None,
) -> list[CentroidMatchFeature]:
    """One CentroidMatchFeature per declared facet, ordered as in content_config.

    Mirrors the DECISION_MODEL_WEIGHTS pattern: the weights list is positional
    along the facet declaration order; a length mismatch is an error so a
    config change cannot silently disable a facet. As with all features, a
    weight of 0 means the Reranker drops the feature entirely.
    """
    facet_map = facets if facets is not None else classifier.facets
    facet_names = list(facet_map.keys())
    if len(weights) != len(facet_names):
        raise ValueError(
            "CENTROID_FACET_WEIGHTS must provide one weight per declared facet "
            f"({len(facet_names)}: {facet_names}); got {len(weights)}."
        )

    features: list[CentroidMatchFeature] = []
    for facet_name, weight in zip(facet_names, weights):
        features.append(
            CentroidMatchFeature(
                weight=weight,
                classifier=classifier,
                facet=facet_map[facet_name],
                facet_name=facet_name,
                bonus=bonus,
                penalty=penalty,
            )
        )
    return features
