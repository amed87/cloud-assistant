from app.exceptions.base import FaqBotError


class ProviderError(FaqBotError):
    """Base class for provider errors."""


class ProviderUnavailableError(ProviderError):
    """The language model is unavailable."""


class ProviderTimeoutError(ProviderError):
    """The language model did not respond in time."""