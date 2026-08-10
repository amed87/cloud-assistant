import logging
from collections.abc import AsyncIterator

from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.models.role import Role

from app.providers.base import LLMProvider
from app.repositories.base import ConversationRepository

from app.prompts.builder import PromptBuilder
from app.services.rag_service import RagService


logger = logging.getLogger(__name__)


class ChatService:

    def __init__(
        self,
        provider: LLMProvider,
        repository: ConversationRepository,
        prompt_builder: PromptBuilder,
        rag_service: RagService,
    ):
        self.provider = provider
        self.repository = repository
        self.prompt_builder = prompt_builder
        self.rag_service = rag_service

    async def ask(
        self,
        conversation_id: str,
        question: str,
    ) -> str:

        conversation = await self._get_or_create_conversation(
            conversation_id
        )

        conversation.messages.append(
            ChatMessage(
                role=Role.USER,
                content=question,
            )
        )

        rag_context = await self.rag_service.retrieve(
            question,
        )

        messages = await self.prompt_builder.build(
            conversation,
            rag_context=rag_context,
        )

        answer = await self.provider.chat(
            messages,
        )

        conversation.messages.append(
            ChatMessage(
                role=Role.ASSISTANT,
                content=answer,
            )
        )

        await self.repository.save(
            conversation,
        )

        return answer

    async def stream_answer(
        self,
        conversation_id: str,
        question: str,
    ) -> AsyncIterator[str]:

        conversation = await self._get_or_create_conversation(
            conversation_id
        )

        conversation.messages.append(
            ChatMessage(
                role=Role.USER,
                content=question,
            )
        )

        rag_context = await self.rag_service.retrieve(
            question,
        )

        messages = await self.prompt_builder.build(
            conversation,
            rag_context=rag_context,
        )

        answer = ""

        async for token in self.provider.stream_chat(
            messages,
        ):
            answer += token
            yield token

        conversation.messages.append(
            ChatMessage(
                role=Role.ASSISTANT,
                content=answer,
            )
        )

        await self.repository.save(
            conversation,
        )

    async def _get_or_create_conversation(
        self,
        conversation_id: str,
    ) -> Conversation:

        conversation = await self.repository.get(
            conversation_id,
        )

        if conversation is None:
            conversation = Conversation(
                id=conversation_id,
            )

        return conversation