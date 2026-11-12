"""Verify Ollama decision requests map configured prompts and provider failures."""

from types import SimpleNamespace

import pytest
from httpx import ConnectError, TimeoutException

from app.exceptions.provider import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.quality.providers.decision import ollama_provider


class FakeDecisionClient:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    async def systemone(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            answers={
                "matched_faq": SimpleNamespace(
                    probabilities={
                        "Candidate question": 0.8,
                        "None of the above options.": 0.2,
                    },
                    confidence=0.8,
                )
            }
        )


@pytest.mark.asyncio
async def test_decide_maps_prompts_and_removes_none_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDecisionClient()
    monkeypatch.setattr(ollama_provider, "AsyncClient", lambda: client)
    provider = ollama_provider.OllamaDecisionProvider(model="test-model")
    choices = ["Candidate question"]

    results = await provider.decide(
        query="User question",
        questions={"matched_faq": "Check if this is a match"},
        choices=choices,
    )

    assert choices == ["Candidate question"]
    assert client.request == {
        "model": "test-model",
        "state": "Nutzer-Suchanfrage: User question",
        "questions": {
            "matched_faq": {
                "type": "choice",
                "instructions": "Check if this is a match",
                "criteria": {"Candidate question": "Candidate question"},
            }
        },
    }
    assert results["matched_faq"].probabilities == {"Candidate question": 0.8}
    assert results["matched_faq"].confidence == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_error", "expected_error"),
    [
        (ConnectError("connection details"), ProviderUnavailableError),
        (TimeoutException("timeout details"), ProviderTimeoutError),
        (RuntimeError("backend details"), ProviderError),
    ],
)
async def test_decide_translates_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    source_error: Exception,
    expected_error: type[ProviderError],
) -> None:
    class FailingDecisionClient:
        request: dict[str, object] | None = None

        async def systemone(self, **kwargs: object) -> None:
            self.request = kwargs
            raise source_error

    client = FailingDecisionClient()
    monkeypatch.setattr(
        ollama_provider,
        "AsyncClient",
        lambda: client,
    )
    provider = ollama_provider.OllamaDecisionProvider()

    with pytest.raises(expected_error) as raised:
        await provider.decide(
            query="User question",
            questions={"matched_faq": "Check if this is a match"},
            choices=["Candidate question"],
        )

    assert raised.value.__cause__ is source_error
    assert client.request is not None
    assert client.request["state"] == "Nutzer-Suchanfrage: User question"
