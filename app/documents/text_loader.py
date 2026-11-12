from pathlib import Path

from app.documents.loader import DocumentLoader
from app.documents.models import TextDocument


class TextLoader(DocumentLoader):
    """Load a UTF-8 text file into a document model."""

    async def load(
        self,
        path: str,
    ) -> TextDocument:
        """Read ``path`` without blocking the async event loop."""
        file_path = Path(path)

        content = await self._read_file(
            file_path,
        )

        return TextDocument(
            id=file_path.stem,
            source=str(file_path),
            content=content,
        )

    async def _read_file(
        self,
        path: Path,
    ) -> str:
        """Perform the blocking filesystem read in a worker thread."""
        import asyncio

        return await asyncio.to_thread(
            path.read_text,
            encoding="utf-8",
        )