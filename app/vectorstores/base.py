from abc import ABC, abstractmethod
from typing import Any

from app.vectorstores.models import (
    SearchResult,
    VectorDocument,
)


class VectorStore(ABC):

    @abstractmethod
    async def add(
        self,
        documents: list[VectorDocument],
    ) -> None:
        """
        Speichert Dokumente im Vector Store.
        """
        pass

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 5,
        intent: str = "general",
    ) -> list[SearchResult]:
        """
        Liefert die ähnlichsten Dokumente.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: list[str],
    ) -> None:
        """
        Löscht Dokumente.
        """
        pass

    @abstractmethod
    def get(
        self
    ) -> dict[str, Any]:
        """Fetch the entire database as a string."""
        pass