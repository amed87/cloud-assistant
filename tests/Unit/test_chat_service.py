import pytest

from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.models.role import Role
from app.services.chat_service import ChatService
from app.services.rag_service import RagContext


class FakeProvider:
    async def chat(self, messages: list[ChatMessage]) -> str:
        return "Normale Antwort"

    async def stream_chat(self, messages: list[ChatMessage]):
        yield "Teil "
        yield "eins"


class FakeRepository:
    def __init__(self) -> None:
        self.conversations: dict[str, Conversation] = {}

    async def get(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    async def save(self, conversation: Conversation) -> None:
        self.conversations[conversation.id] = conversation

    async def truncate(self, conversation_id: str, max_turns: int) -> None:
        pass


class FakePromptPipeline:
    async def build(
        self,
        conversation: Conversation,
        rag_context: RagContext | None = None,
    ) -> list[ChatMessage]:
        return conversation.messages


class FakeRagService:
    async def retrieve(self, question: str) -> RagContext:
        return RagContext(documents=[])


def create_service() -> tuple[ChatService, FakeRepository]:
    repository = FakeRepository()

    service = ChatService(
        provider=FakeProvider(),
        repository=repository,
        prompt_pipeline=FakePromptPipeline(),
        rag_service=FakeRagService(),
    )

    return service, repository


@pytest.mark.asyncio
async def test_ask_returns_and_saves_answer() -> None:
    service, repository = create_service()

    answer = await service.ask("conversation-1", "Hallo")

    assert answer == "Normale Antwort"

    conversation = repository.conversations["conversation-1"]

    assert conversation.messages == [
        ChatMessage(role=Role.USER, content="Hallo"),
        ChatMessage(role=Role.ASSISTANT, content="Normale Antwort"),
    ]


@pytest.mark.asyncio
async def test_stream_answer_yields_and_saves_complete_answer() -> None:
    service, repository = create_service()

    tokens = [
        token
        async for token in service.stream_answer(
            "conversation-1",
            "Hallo",
        )
    ]

    assert tokens == ["Teil ", "eins"]

    conversation = repository.conversations["conversation-1"]

    assert conversation.messages[-1] == ChatMessage(
        role=Role.ASSISTANT,
        content="Teil eins",
    )