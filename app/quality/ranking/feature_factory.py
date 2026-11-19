"""Eine Quelle für den Reranker-Feature-Satz (Chat, QuickChat, Evaluation).

Alle Einstiegspunkte bauen ihren Reranker aus derselben Deklaration: Intent,
Keyword und die in content_config.yaml deklarierten Zentroid-Facetten — plus,
auf Wunsch, die parallelen Decision-Model-Prompts. So misst die Evaluation
exakt dasselbe Feature-Set, das der Chatbot zur Laufzeit benutzt, und eine
neue Reranker-Funktion wird nur noch hier ergänzt.
"""

from app.config.content_settings import content_config
from app.config.settings import get_settings
from app.quality.ranking.decision_model_feature import DecisionModelFeature
from app.quality.ranking.dm_single_feature import DMSingleFeature
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.keyword_feature import KeywordFeature
from app.dependencies.centroids import get_centroid_features
from app.dependencies.providers import decision_provider


def build_rerank_features(
    *,
    include_decision_model: bool = True,
) -> tuple[list, DecisionModelFeature | None]:
    """Build the configured reranker features.

    ``include_decision_model=False`` erzeugt ein Feature-Set ohne DM-Prompts
    (für Chat-Pfade, die keine Decision-Model-Aufrufe auslösen sollen).
    Feature-Gewichte von 0 sind erlaubt: der Reranker droppt solche Features
    und sie lösen dann auch keine Klassifikations-/DM-Aufrufe aus.
    """
    settings = get_settings()

    features: list = [
        IntentFeature(
            weight=settings.INTENT_WEIGHT,
            bonus=settings.INTENT_MATCH_BONUS,
            penalty=settings.INTENT_MISMATCH_PENALTY,
            fallback=content_config.fragetyp_fallback,
        ),
        KeywordFeature(
            weight=settings.KEYWORD_WEIGHT,
            normalization=settings.KEYWORD_MATCH_NORMALIZATION,
        ),
        *get_centroid_features(),
    ]

    if not include_decision_model:
        return features, None

    prompt_names = list(content_config.decision_model_prompts)
    if len(settings.DECISION_MODEL_WEIGHTS) != len(prompt_names):
        raise ValueError(
            "DECISION_MODEL_WEIGHTS must contain one weight for each "
            "decision_model_prompts entry."
        )

    dm_feature = DecisionModelFeature(
        weight=0.0,
        decision_provider=decision_provider,
    )
    features.extend(
        DMSingleFeature(
            name=name,
            weight=weight,
            parent_feature=dm_feature,
        )
        for name, weight in zip(prompt_names, settings.DECISION_MODEL_WEIGHTS)
    )
    return features, dm_feature
