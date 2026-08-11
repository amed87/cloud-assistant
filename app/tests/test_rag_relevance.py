import asyncio

from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.services.rag_service import RagService
from app.vectorstores.chroma import ChromaVectorStore


QUESTIONS = [
    "Was ist Kubernetes?",
    "Wofür wird Kubernetes verwendet?",
    "Was ist ein Kubernetes Cluster?",
    "Wie funktioniert Kubernetes?",
    "Was ist Docker?",
    "Wie backe ich einen Schokoladenkuchen?",
    "Wie repariere ich ein Fahrrad?",
    "Was ist die Hauptstadt von Frankreich?",
]


async def main():

    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()

    rag_service = RagService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    for question in QUESTIONS:

        print()
        print("=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        context = await rag_service.retrieve(
            question,
        )

        if not context.documents:
            print("NO RELEVANT DOCUMENTS")
            continue

        for document in context.documents:

            print(
                f"Score: {document.score:.4f}"
            )

            print(
                f"ID: {document.id}"
            )

            print(
                f"Content: {document.content[:200]}"
            )


if __name__ == "__main__":
    asyncio.run(main())