from abc import ABC, abstractmethod


class PromptProvider(ABC):

    @abstractmethod
    async def system_prompt(self) -> str:
        """Return the current system prompt."""
        ...