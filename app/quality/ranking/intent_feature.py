from app.config.content_settings import content_config
from app.config.settings import get_settings
from app.quality.intent.intent_filter import filterIntent
from app.quality.ranking.ranking_feature import RankingFeature
from app.vectorstores.models import SearchResult
from dataclasses import dataclass


@dataclass(slots=True)
class IntentFeature(RankingFeature):
    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:

        settings = get_settings()

        fallback = content_config.fragetyp_fallback

        query_intent = filterIntent(query)

        candidate_intent = candidate.metadata.get(
            "question_type",
            fallback,
        )

        if candidate_intent == query_intent:
            return settings.INTENT_MATCH_BONUS

        if (
            query_intent != fallback
            and candidate_intent != fallback
        ):
            return -settings.INTENT_MISMATCH_PENALTY

        return 0.0