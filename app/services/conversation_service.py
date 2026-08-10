from app.models.conversation import Conversation
from app.repositories.base import ConversationRepository


class ConversationService:

    def __init__(
        self,
        repository: ConversationRepository,
    ):
        self.repository = repository

    async def get_or_create(
        self,
        conversation_id: str,
    ) -> Conversation:

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

        await self.repository.save(
            conversation
        )