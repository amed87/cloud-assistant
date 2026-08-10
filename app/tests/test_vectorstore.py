import asyncio

from app.embeddings.ollama_provider import (
    OllamaEmbeddingProvider,
)
from app.vectorstores.chroma import (
    ChromaVectorStore,
)
from app.vectorstores.models import (
    VectorDocument,
)


async def main():

    embedding_provider = OllamaEmbeddingProvider()

    store = ChromaVectorStore()

    embedding = await embedding_provider.embed(
        "Kubernetes orchestriert Container."
    )

    await store.add(
        [
            VectorDocument(
                id="1",
                content="Kubernetes orchestriert Container.",
                embedding=embedding,
                metadata={
                    "source": "test",
                },
            )
        ]
    )

    results = await store.search(
        embedding,
    )

    print(results)


asyncio.run(main())