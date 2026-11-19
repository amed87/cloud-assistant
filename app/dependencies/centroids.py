"""Geteilte Zentroid-Komponenten für Chat, QuickChat und DB-Update.

Eine einzige Store-/Classifier-Instanz pro Prozess: sie teilt sich den
In-Memory-Embedding-Cache, lädt die Zentroide einmal (Neuladen nach
Updates via mtime-Check) und wird vom Update-Service direkt befüllt —
der DB-Update-Lauf installiert die neuen Zentroide damit in derselben
Instanz, die die Services vorher geladen haben.
"""

from pathlib import Path

from app.config.content_settings import content_config
from app.config.settings import get_settings
from app.quality.classification.centroid_store import CentroidStore
from app.quality.classification.query_classifier import (
    FacetClassifierConfig,
    QueryFacetClassifier,
)
from app.dependencies.providers import embedding_provider, decision_provider

settings = get_settings()

# Embedding- und Decision-Model-Provider stammen aus app.dependencies.providers —
# derselbe DecisionProvider bedient auch das normale DM-Feature (quick_chat).
centroid_store = CentroidStore(
    store_path=Path(settings.DB_PATH) / settings.CENTROID_FILE_NAME,
    embedding_provider=embedding_provider,
    facets=content_config.centroid_facets,
)

# Der Klassifikations-DM-Fallback feuert nur, wenn CENTROID_DM_FALLBACK
# gesetzt ist — der Provider selbst kostet sonst nichts.
facet_classifier = QueryFacetClassifier(
    store=centroid_store,
    facets=content_config.centroid_facets,
    config=FacetClassifierConfig(
        floor=settings.CENTROID_CONFIDENCE_FLOOR,
        margin=settings.CENTROID_CONFIDENCE_MARGIN,
        dm_fallback=settings.CENTROID_DM_FALLBACK,
    ),
    decision_provider=decision_provider,
)

centroid_features = None  # faul via get_centroid_features(), siehe unten


def get_centroid_features():
    """CentroidMatchFeatures gemäß content_config + settings (einmalig erzeugt)."""
    global centroid_features
    if centroid_features is None:
        from app.quality.ranking.centroid_feature import build_centroid_features

        centroid_features = build_centroid_features(
            classifier=facet_classifier,
            weights=settings.CENTROID_FACET_WEIGHTS,
            bonus=settings.CENTROID_MATCH_BONUS,
            penalty=settings.CENTROID_MATCH_PENALTY,
        )
    return centroid_features
