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
        """Load a document from a file."""
        pass

class FAQLoader(DocumentLoader):

    async def load(
            self,
            path: str,
        ) -> FAQDocument:
            """Load an FAQ document from a file."""
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
            """Read FAQ entries from a file."""
            return await to_thread(
            self._parse,
            path,
            encoding="utf-8-sig",
        )       

    @abstractmethod
    def _parse( 
        self,
        path: Path,
        encoding: str,
    ) -> list[FAQEntry]:
        """Parse FAQ entries from a file."""
        pass  