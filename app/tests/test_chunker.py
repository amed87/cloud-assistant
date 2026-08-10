import asyncio

from app.documents.chunker import DocumentChunker
from app.documents.text_loader import TextLoader


async def main():

    loader = TextLoader()

    document = await loader.load(
        "data/test.txt",
    )

    chunker = DocumentChunker(
        chunk_size=50,
        overlap=10,
    )

    chunks = chunker.chunk(document)

    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:
        print()
        print("ID:", chunk.id)
        print("Index:", chunk.chunk_index)
        print("Content:", chunk.content)


asyncio.run(main())