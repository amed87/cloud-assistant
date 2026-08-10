from app.exceptions.base import CloudAssistantError


class ConversationError(CloudAssistantError):
    """Fehler im Conversation Repository."""


class ConversationNotFoundError(ConversationError):
    """Conversation existiert nicht."""