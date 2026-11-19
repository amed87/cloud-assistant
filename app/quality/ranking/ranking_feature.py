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

    async def prime(
        self,
        query: str,
        query_embedding: list[float] | None = None,
    ) -> None:
        """Optional asynchronous preparation before the synchronous scoring loop.

        Features with expensive per-query work (decision-model calls,
        facet classification) override this; the default is a no-op.
        ``query_embedding`` is the query embedding already computed for
        vector search, so features must not embed the query a second time.
        """
        return None