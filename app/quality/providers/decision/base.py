from abc import ABC, abstractmethod

class DecisionProvider(ABC):

    @abstractmethod
    async def decide(
        self,
        query: str,
        question: str,
        choices: list[str],
    ) -> dict[str, float]:
        """
        Evaluates a state (query) against a set of choices based on a question.
        Returns a dictionary mapping each choice to its probability score (0.0 - 1.0).
        """
        pass
