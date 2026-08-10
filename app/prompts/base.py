from abc import ABC, abstractmethod


class PromptProvider(ABC):

    @abstractmethod
    async def system_prompt(self) -> str:
        """Liefert den aktuellen System Prompt."""
        ...