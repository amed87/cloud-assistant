from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.models.role import Role
from app.prompts.base import PromptProvider
from app.prompts.steps.base import PromptStep


class SystemPromptStep(PromptStep):

    def __init__(
        self,
        prompt_provider: PromptProvider,
    ):
        self.prompt_provider = prompt_provider

    async def build(
        self,
        conversation: Conversation,
        messages: list[ChatMessage],
    ) -> None:

        system_prompt = await self.prompt_provider.system_prompt()

        messages.append(
            ChatMessage(
                role=Role.SYSTEM,
                content=system_prompt,
            )
        )