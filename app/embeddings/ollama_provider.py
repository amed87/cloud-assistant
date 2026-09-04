from ollama import AsyncClient

from app.config.settings import get_settings
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
        self.threads = int(get_settings().ollama_num_threads)
        self.ctx_size = int(get_settings().OLLAMA_NUM_CTX)

    async def embed(
        self,
        text: str,
    ) -> list[float]:

        response = await self.client.embed(
            model=self.model,
            input=text,
            keep_alive=-1,
            options={
                'num_thread': self.threads,
                'num_ctx': self.ctx_size,
                'num_keep': 0,
            }
        )

        return response.embeddings[0]