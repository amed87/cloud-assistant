import logging
import time
from collections.abc import AsyncIterator

from httpx import ConnectError, ReadTimeout
from ollama import AsyncClient, ResponseError

from app.config.settings import get_settings
from app.exceptions.provider import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.models.message import ChatMessage
from app.providers.base import LLMProvider


logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):

    def __init__(self):
        settings = get_settings()

        self.model = settings.ollama_model
        self.client = AsyncClient(
            host=settings.ollama_host,
        )
        self.threads = int(settings.ollama_num_threads)
        self.ctx_size = int(settings.OLLAMA_NUM_CTX)

    def _to_ollama_messages(
        self,
        messages: list[ChatMessage],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    async def chat(
        self,
        messages: list[ChatMessage],
    ) -> str:

        ollama_messages = self._to_ollama_messages(messages)

        start = time.perf_counter()

        try:

            logger.info(
                "Sending request to Ollama (%s)",
                self.model,
            )

            response = await self.client.chat(
                model=self.model,
                messages=ollama_messages,
                keep_alive=-1,
                options={
                    'num_thread': self.threads,
                    'num_ctx': self.ctx_size,
                    'num_keep': 0,
                }
            )

            duration = time.perf_counter() - start

            logger.info(
                "Ollama answered in %.2f seconds",
                duration,
            )

            return response.message.content

        except ConnectError as ex:

            logger.exception(
                "Unable to connect to Ollama."
            )

            raise ProviderUnavailableError(
                "Ollama server is not reachable."
            ) from ex

        except ReadTimeout as ex:

            logger.exception(
                "Ollama request timed out."
            )

            raise ProviderTimeoutError(
                "The language model did not respond in time."
            ) from ex

        except ResponseError as ex:

            logger.exception(
                "Ollama returned an error."
            )

            raise ProviderError(
                "The response could not be generated."
            ) from ex

        except Exception as ex:

            logger.exception(
                "Unexpected provider error."
            )

            raise ProviderError(
                "Unexpected provider error."
            ) from ex

    async def stream_chat(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:

        ollama_messages = self._to_ollama_messages(messages)

        start = time.perf_counter()

        try:

            logger.info(
                "Starting Ollama stream (%s)",
                self.model,
            )

            stream = await self.client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
            )

            async for chunk in stream:

                content = chunk.message.content

                if content:
                    yield content

            duration = time.perf_counter() - start

            logger.info(
                "Streaming finished in %.2f seconds",
                duration,
            )

        except ConnectError as ex:

            logger.exception(
                "Unable to connect to Ollama."
            )

            raise ProviderUnavailableError(
                "Ollama server is not reachable."
            ) from ex

        except ReadTimeout as ex:

            logger.exception(
                "Ollama streaming timed out."
            )

            raise ProviderTimeoutError(
                "The language model did not respond in time."
            ) from ex

        except ResponseError as ex:

            logger.exception(
                "Ollama returned an error while streaming."
            )

            raise ProviderError(
                "The response could not be generated."
            ) from ex

        except Exception as ex:

            logger.exception(
                "Unexpected provider streaming error."
            )

            raise ProviderError(
                "Unexpected provider error."
            ) from ex