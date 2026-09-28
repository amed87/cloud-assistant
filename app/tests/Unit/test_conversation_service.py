import pytest

from app.models.conversation import Conversation
from app.repositories.memory import MemoryConversationRepository
from app.services.conversation_service import ConversationService


@pytest.fixture
def service() -> ConversationService:
    return ConversationService(MemoryConversationRepository())


@pytest.mark.asyncio
async def test_get_or_create_creates_missing_conversation(
    service: ConversationService,
) -> None:
    conversation = await service.get_or_create("conversation-1")

    assert conversation.id == "conversation-1"
    assert conversation.messages == []


@pytest.mark.asyncio
async def test_get_or_create_returns_existing_conversation(
    service: ConversationService,
) -> None:
    existing = Conversation(id="conversation-1")
    await service.save(existing)

    result = await service.get_or_create("conversation-1")

    assert result is existing


@pytest.mark.asyncio
async def test_save_persists_conversation(
    service: ConversationService,
) -> None:
    conversation = Conversation(id="conversation-1")

    await service.save(conversation)

    result = await service.get_or_create("conversation-1")

    assert result is conversation