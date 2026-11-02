import logging
from typing import Any

from httpx import ConnectError, TimeoutException
from ollama import AsyncClient, ResponseError

from app.config.settings import get_settings
from app.quality.providers.decision.base import DecisionProvider
from app.exceptions.provider import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.quality.providers.decision.models import DecisionResult

logger = logging.getLogger(__name__)

class OllamaDecisionProvider(
    DecisionProvider,
):

    def __init__(
        self,
        model: str = "laya",
    ):
        self.model = model
        self.client = AsyncClient()
        self.threads = int(get_settings().ollama_num_threads)
        self.ctx_size = int(get_settings().OLLAMA_NUM_CTX)

    async def decide(
        self,
        query: str,
        questions: dict[str, str],
        choices: list[str],
    ) -> dict[str, DecisionResult]:
        requests = {
            name: {
                "type": "choice", 
                "instructions": question, 
                "criteria": {choice: choice for choice in choices}
                } 
            for name, question in questions.items()
            }

        try:
            NONE = "None of the above options."
            choices.append(NONE)

            # system_one ist der native Ollama-Endpunkt für Entscheidungsmodelle
            response = await self.client.systemone(
                model=self.model,
                state=f"Nutzer-Suchanfrage: {query}",
                questions=requests,
            )
        except ConnectError as ex:
            logger.exception("Unable to connect to the decision provider.")
            raise ProviderUnavailableError(
                "The decision provider is not reachable."
            ) from ex
        except TimeoutException as ex:
            logger.exception("Decision provider request timed out.")
            raise ProviderTimeoutError(
                "The decision provider did not respond in time."
            ) from ex
        except ResponseError as ex:
            logger.exception("Decision provider returned an error.")
            raise ProviderError(
                "The decision could not be processed."
            ) from ex
        except Exception as ex:
            logger.exception("Unexpected decision provider error.")
            raise ProviderError(
                "Unexpected decision provider error."
            ) from ex

        results = {}
        for key, answer in response.answers.items():
            probs = answer.probabilities
            probs.pop(NONE, 0.0)
            result = DecisionResult(
                probabilities=probs,
                confidence=answer.confidence,
            )
            results[key] = result

        return results
