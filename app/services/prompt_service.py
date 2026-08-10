from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.prompts.builder import PromptBuilder


class PromptService:

    def __init__(
        self,
        prompt_builder: PromptBuilder,
    ):
        self.prompt_builder = prompt_builder

    async def build(
        self,
        conversation: Conversation,
    ) -> list[ChatMessage]:

        return await self.prompt_builder.build(
            conversation
        )