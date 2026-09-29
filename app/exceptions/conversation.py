from app.exceptions.base import FaqBotError


class ConversationError(FaqBotError):
    """Fehler im Conversation Repository."""


class ConversationNotFoundError(ConversationError):
    """Conversation existiert nicht."""