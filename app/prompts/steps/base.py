from abc import ABC, abstractmethod

from app.models.conversation import Conversation
from app.models.message import ChatMessage


class PromptStep(ABC):

    @abstractmethod
    async def build(
        self,
        conversation: Conversation,
        messages: list[ChatMessage],
    ) -> None:
        """Erweitert die Prompt-Nachrichten."""
        pass