from app.documents.models import (
    Document,
    DocumentChunk,
)


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        chunks: list[DocumentChunk] = []

        start = 0
        index = 0

        while start < len(document.content):

            end = start + self.chunk_size

            chunks.append(
                DocumentChunk(
                    id=f"{document.id}-{index}",
                    source=document.source,
                    chunk_index=index,
                    content=document.content[start:end],
                )
            )

            start += self.chunk_size - self.overlap
            index += 1

        return chunks