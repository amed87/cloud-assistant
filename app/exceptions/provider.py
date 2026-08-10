from app.exceptions.base import CloudAssistantError


class ProviderError(CloudAssistantError):
    """Allgemeiner Providerfehler."""


class ProviderUnavailableError(ProviderError):
    """LLM nicht erreichbar."""


class ProviderTimeoutError(ProviderError):
    """LLM hat nicht rechtzeitig geantwortet."""