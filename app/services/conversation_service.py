from app.models.conversation import Conversation
from app.repositories.base import ConversationRepository


class ConversationService:
    """Provide conversation lookup, initialization, and persistence operations."""

    def __init__(
        self,
        repository: ConversationRepository,
    ):
        self.repository = repository

    async def get_or_create(
        self,
        conversation_id: str,
    ) -> Conversation:
        """Load a conversation or return a new empty one with the requested ID."""
        conversation = await self.repository.get(
            conversation_id
        )

        if conversation is None:
            conversation = Conversation(
                id=conversation_id
            )

        return conversation

    async def save(
        self,
        conversation: Conversation,
    ):
        """Persist the conversation through the configured repository."""
        await self.repository.save(
            conversation
        )