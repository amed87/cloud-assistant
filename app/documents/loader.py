from abc import ABC, abstractmethod

from app.documents.models import Document


class DocumentLoader(ABC):

    @abstractmethod
    async def load(
        self,
        path: str,
    ) -> Document:
        """Lädt ein Dokument aus einer Datei."""
        pass