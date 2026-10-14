from app.quality.intent.intent_filter import filterIntent
from app.vectorstores.models import SearchResult
from app.config.content_settings import content_config
from app.config.settings import get_settings


class IntentFeature:

    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:

        settings = get_settings()

        fallback = content_config.fragetyp_fallback

        intent = filterIntent(query)

        metadata_intent = candidate.metadata.get(
            "question_type",
            fallback,
        )

        if metadata_intent == intent:
            return settings.INTENT_MATCH_BONUS

        if (
            intent != fallback
            and metadata_intent != fallback
        ):
            return -settings.INTENT_MISMATCH_PENALTY

        return 0.0