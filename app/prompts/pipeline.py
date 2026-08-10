from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.prompts.steps.base import PromptStep


class PromptPipeline:

    def __init__(
        self,
        steps: list[PromptStep],
    ):
        self.steps = steps

    async def build(
        self,
        conversation: Conversation,
    ) -> list[ChatMessage]:

        messages: list[ChatMessage] = []

        for step in self.steps:

            await step.build(
                conversation,
                messages,
            )

        return messages