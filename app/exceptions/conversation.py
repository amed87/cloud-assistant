from app.exceptions.base import FaqBotError


class ConversationError(FaqBotError):
    """Base class for conversation repository errors."""


class ConversationNotFoundError(ConversationError):
    """The conversation does not exist."""