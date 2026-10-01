import logging

from httpx import ConnectError, TimeoutException
from ollama import AsyncClient, ResponseError

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingProvider
from app.exceptions.provider import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


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

        try:
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
        except ConnectError as ex:
            logger.exception("Unable to connect to the embedding provider.")
            raise ProviderUnavailableError(
                "The embedding provider is not reachable."
            ) from ex
        except TimeoutException as ex:
            logger.exception("Embedding provider request timed out.")
            raise ProviderTimeoutError(
                "The embedding provider did not respond in time."
            ) from ex
        except ResponseError as ex:
            logger.exception("Embedding provider returned an error.")
            raise ProviderError(
                "The embedding could not be generated."
            ) from ex
        except Exception as ex:
            logger.exception("Unexpected embedding provider error.")
            raise ProviderError(
                "Unexpected embedding provider error."
            ) from ex

        return response.embeddings[0]