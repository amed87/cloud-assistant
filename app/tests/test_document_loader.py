import asyncio

from app.documents.text_loader import TextLoader


async def main():

    loader = TextLoader()

    document = await loader.load(
        "data/test.txt",
    )

    print("ID:", document.id)
    print("Source:", document.source)
    print("Content:", document.content)


asyncio.run(main())