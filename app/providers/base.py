from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.models.message import ChatMessage


class LLMProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
    ) -> str:
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        ...