import asyncio

from app.embeddings.ollama_provider import (
    OllamaEmbeddingProvider,
)


async def main():

    provider = OllamaEmbeddingProvider()

    embedding = await provider.embed(
        "Was ist Kubernetes?"
    )

    print(len(embedding))
    print(embedding[:10])


asyncio.run(main())