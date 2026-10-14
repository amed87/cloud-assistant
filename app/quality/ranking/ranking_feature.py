from abc import ABC
from abc import abstractmethod

from app.vectorstores.models import SearchResult


class RankingFeature(ABC):
    """Base class for candidate ranking features."""

    name: str

    @abstractmethod
    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:
        """Return ranking contribution."""
        raise NotImplementedError