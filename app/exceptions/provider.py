from app.exceptions.base import FaqBotError


class ProviderError(FaqBotError):
    """Allgemeiner Providerfehler."""


class ProviderUnavailableError(ProviderError):
    """LLM nicht erreichbar."""


class ProviderTimeoutError(ProviderError):
    """LLM hat nicht rechtzeitig geantwortet."""