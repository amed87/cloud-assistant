from app.models.conversation import Conversation
from app.repositories.base import ConversationRepository


class MemoryConversationRepository(ConversationRepository):

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}

    async def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    async def save(self, conversation: Conversation) -> None:
        self._conversations[conversation.id] = conversation

    async def delete(self, conversation_id: str) -> None:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    # Truncate the conversation to a maximum number of turns so that it doesn't grow indefinitely. 
    # This is useful for memory management and performance.
    async def truncate(self, conversation_id: str, max_turns: int=5) -> None:
        if conversation_id in self._conversations and len(self._conversations[conversation_id].messages) > max_turns:
            self._conversations[conversation_id].messages = self._conversations[conversation_id].messages[-max_turns:]