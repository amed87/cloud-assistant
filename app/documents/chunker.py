from app.documents.models import DocumentChunk, TextDocument


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0:
            raise ValueError("overlap must not be negative")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size",
            )
        
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(
        self,
        document: TextDocument,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        start = 0
        index = 0

        while start < len(document.content):
            end = start + self._chunk_size

            chunks.append(
                DocumentChunk(
                    id=f"{document.id}-{index}",
                    source=document.source,
                    chunk_index=index,
                    content=document.content[start:end],
                )
            )

            start += self._chunk_size - self._overlap
            index += 1

        return chunks