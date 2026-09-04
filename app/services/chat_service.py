import logging
from collections.abc import AsyncIterator

from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.models.role import Role
from app.config.settings import get_settings

from app.providers.base import LLMProvider
from app.repositories.base import ConversationRepository

from app.prompts.pipeline import PromptPipeline
from app.services.rag_service import RagService


logger = logging.getLogger(__name__)


class ChatService:

    def __init__(
        self,
        provider: LLMProvider,
        repository: ConversationRepository,
        prompt_pipeline: PromptPipeline,
        rag_service: RagService,
    ):
        self.provider = provider
        self.repository = repository
        self.prompt_pipeline = prompt_pipeline
        self.rag_service = rag_service
        self.max_turns = get_settings().NUMBER_OF_TURNS

    async def ask(
        self,
        conversation_id: str,
        question: str,
    ) -> str:

        conversation = await self._get_or_create_conversation(
            conversation_id
        )

        await self.repository.truncate(
            conversation_id,
            max_turns=self.max_turns
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

        messages = await self.prompt_pipeline.build(
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

        await self.repository.truncate(
            conversation_id,
            max_turns=self.max_turns
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

        messages = await self.prompt_pipeline.build(
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