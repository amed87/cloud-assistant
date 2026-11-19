"""Geteilte Provider-Instanzen pro Prozess.

Frage-Auflösung (LLM-Provider werden von den Services selbst gehalten) der
zwei alles-umspannenden Brücken zu Ollama: Embeddings und Decision-Model.
Eine je Prozess, damit DM-Feature und Zentroid-Klassifikator dieselbe
Instanz — und damit denselben Cache — nutzen.
"""

from app.config.settings import get_settings
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.quality.providers.decision.ollama_provider import OllamaDecisionProvider

settings = get_settings()

embedding_provider = OllamaEmbeddingProvider()
decision_provider = OllamaDecisionProvider(settings.DECISION_MODEL)
