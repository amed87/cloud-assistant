from abc import ABC, abstractmethod

from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.services.rag_service import RagContext
import typing as t


class PromptStep(ABC):

    @abstractmethod
    async def build(
        self,
        conversation: Conversation,
        messages: list[ChatMessage],
        *_: t.Any,
    ) -> None:
        """Extend the prompt messages."""
        pass