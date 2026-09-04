from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.prompts.steps.base import PromptStep
from app.services.rag_service import RagContext
import typing as t


class ConversationStep(PromptStep):

    async def build(
        self,
        conversation: Conversation,
        messages: list[ChatMessage],
        *_: t.Any,
    ) -> None:

        messages.extend(
            conversation.messages
        )