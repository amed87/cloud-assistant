from abc import ABC, abstractmethod
from pathlib import Path
from asyncio import to_thread

from app.documents.models import Document, FAQDocument, FAQEntry


class DocumentLoader(ABC):

    @abstractmethod
    async def load(
        self,
        path: str,
    ) -> Document:
        """Lädt ein Dokument aus einer Datei."""
        pass

class FAQLoader(DocumentLoader):

    async def load(
            self,
            path: str,
        ) -> FAQDocument:
            """Lädt ein FAQ-Dokument aus einer Datei."""
            file_path = Path(path)

            entries = await self._read_file(
                file_path,
            )

            return FAQDocument(
                id=file_path.stem,
                source=str(file_path),
                entries=entries,
            )

    async def _read_file(
            self,
            path: Path,
        ) -> list[FAQEntry]:
            """Liest die FAQ-Einträge aus einer Datei."""
            return await to_thread(
            self._parse,
            path,
            encoding="utf-8",
        )       

    @abstractmethod
    def _parse( 
        self,
        path: Path,
        encoding: str,
    ) -> list[FAQEntry]:
        """Parst die FAQ-Einträge aus einer Datei."""
        pass  