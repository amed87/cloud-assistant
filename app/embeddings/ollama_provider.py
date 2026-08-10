from ollama import AsyncClient

from app.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(
    EmbeddingProvider,
):

    def __init__(
        self,
        model: str = "nomic-embed-text",
    ):
        self.model = model
        self.client = AsyncClient()

    async def embed(
        self,
        text: str,
    ) -> list[float]:

        response = await self.client.embed(
            model=self.model,
            input=text,
        )

        return response.embeddings[0]