from pathlib import Path

from app.documents.loader import DocumentLoader
from app.documents.models import Document


class TextLoader(DocumentLoader):

    async def load(
        self,
        path: str,
    ) -> Document:

        file_path = Path(path)

        content = await self._read_file(
            file_path,
        )

        return Document(
            id=file_path.stem,
            source=str(file_path),
            content=content,
        )

    async def _read_file(
        self,
        path: Path,
    ) -> str:

        import asyncio

        return await asyncio.to_thread(
            path.read_text,
            encoding="utf-8",
        )