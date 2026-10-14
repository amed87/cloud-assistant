from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass 

from app.vectorstores.models import SearchResult

@dataclass(slots=True)
class RankingFeature(ABC):
    """Base class for candidate ranking features."""
    weight: float

    @abstractmethod
    def score(
        self,
        query: str,
        candidate: SearchResult,
    ) -> float:
        """Return ranking contribution."""
        raise NotImplementedError