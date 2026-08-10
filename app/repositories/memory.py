from app.models.conversation import Conversation
from app.repositories.base import ConversationRepository


class MemoryConversationRepository(ConversationRepository):

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}

    async def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    async def save(self, conversation: Conversation) -> None:
        self._conversations[conversation.id] = conversation