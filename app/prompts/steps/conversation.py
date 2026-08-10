from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.prompts.steps.base import PromptStep


class ConversationStep(PromptStep):

    async def build(
        self,
        conversation: Conversation,
        messages: list[ChatMessage],
    ) -> None:

        messages.extend(
            conversation.messages
        )