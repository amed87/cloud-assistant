import pytest

from app.documents.chunker import DocumentChunker
from app.documents.models import TextDocument


def test_chunker_creates_overlapping_chunks() -> None:
    document = TextDocument(
        id="document-1",
        source="test.txt",
        content="abcdefghij",
    )
    chunker = DocumentChunker(chunk_size=6, overlap=2)

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "abcdef",
        "efghij",
        "ij",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.id for chunk in chunks] == [
        "document-1-0",
        "document-1-1",
        "document-1-2",
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)

def test_chunker_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        DocumentChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunker_handles_empty_content() -> None:
    document = TextDocument(
        id="document-1",
        source="test.txt",
        content="",
    )
    chunker = DocumentChunker(chunk_size=10, overlap=2)

    assert chunker.chunk(document) == []