from functools import lru_cache

from app.config.settings import get_settings
from app.config.content_settings import content_config
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.documents.faq_loader_csv import FAQLoaderCSV
from app.vectorstores.chroma import ChromaVectorStore
from app.tools.faq_health_check import FAQHealthCheckService
from app.quality.ranking.reranker import Reranker
from app.quality.ranking.intent_feature import IntentFeature
from app.quality.ranking.keyword_feature import KeywordFeature

# Shared Resources (Singletons wie in production)
settings = get_settings()
_embedding_provider = OllamaEmbeddingProvider()
_loader = FAQLoaderCSV()
features =  [
    IntentFeature(
        weight=settings.INTENT_WEIGHT, 
        bonus=settings.INTENT_MATCH_BONUS,
        penalty=settings.INTENT_MISMATCH_PENALTY,
        fallback=content_config.fragetyp_fallback,
        ),
    KeywordFeature(
        weight=settings.KEYWORD_WEIGHT,
        normalization=settings.KEYWORD_MATCH_NORMALIZATION,
    )
    ]
_reranker = Reranker(features)

# Factory für TempChromaStore (pro Request)
def create_temp_store():
    """Erstellt neue temporäre ChromaDB-Instanz pro Request."""
    from app.vectorstores.temp_chroma import TempChromaStore
    return TempChromaStore()

# Factory für FAQImporter (mit injiziertem Store)
def create_importer(vector_store):
    """Erstellt FAQImporter mit injiziertem VectorStore."""
    from app.documents.faq_importer import FAQImporter
    return FAQImporter(
        embedding_provider=_embedding_provider,
        vector_store=vector_store,
    )

# Singleton Service mit ALLEN Dependencies injiziert (wie get_update_service())
@lru_cache()
def get_faq_health_check_service() -> FAQHealthCheckService:
    """Singleton FAQHealthCheckService mit allen Injected Dependencies."""
    return FAQHealthCheckService(
        embedding_provider=_embedding_provider,
        loader=_loader,
        create_temp_store=create_temp_store,
        create_importer=create_importer,
        reranker=_reranker,
    )