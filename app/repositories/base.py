from abc import ABC, abstractmethod

from app.models.conversation import Conversation


class ConversationRepository(ABC):

    @abstractmethod
    async def get(self, conversation_id: str) -> Conversation | None:
        ...

    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        ...