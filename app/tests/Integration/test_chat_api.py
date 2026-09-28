from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router
from app.dependencies.chat import get_chat_service
from app.dependencies.quick_chat import get_quick_chat_service


class FakeChatService:
    async def ask(self, conversation_id: str, question: str) -> str:
        return "Chat-Antwort"

    async def stream_answer(
        self,
        conversation_id: str,
        question: str,
    ) -> AsyncIterator[str]:
        yield "Teil "
        yield "eins"


class FakeQuickChatService:
    async def ask(self, conversation_id: str, question: str) -> str:
        return "Quick-Antwort"


def create_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_chat_service] = FakeChatService
    app.dependency_overrides[get_quick_chat_service] = FakeQuickChatService

    return TestClient(app)


def test_chat_endpoint_returns_answer() -> None:
    response = create_client().post(
        "/chat",
        json={
            "conversation_id": "conversation-1",
            "message": "Hallo",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Chat-Antwort"}


def test_quick_chat_endpoint_returns_answer() -> None:
    response = create_client().post(
        "/chat/quick",
        json={
            "conversation_id": "conversation-1",
            "message": "Hallo",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Quick-Antwort"}


def test_stream_chat_endpoint_returns_stream() -> None:
    response = create_client().post(
        "/chat/stream",
        json={
            "conversation_id": "conversation-1",
            "message": "Hallo",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Teil eins"