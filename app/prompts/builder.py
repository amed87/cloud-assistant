from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.models.role import Role
from app.prompts.base import PromptProvider
from app.services.rag_service import RagContext


class PromptBuilder:

    def __init__(
        self,
        prompt_provider: PromptProvider,
    ):
        self.prompt_provider = prompt_provider

    async def build(
        self,
        conversation: Conversation,
        rag_context: RagContext | None = None,
    ) -> list[ChatMessage]:

        system_prompt = (
            await self.prompt_provider.system_prompt()
        )

        if rag_context is not None and rag_context.text:

            system_prompt += (
                "\n\n"
                "Nutze den folgenden Kontext zur Beantwortung "
                "der Frage. Wenn der Kontext keine ausreichende "
                "Information enthält, sage dies offen.\n\n"
                "KONTEXT:\n"
                f"{rag_context.text}"
            )

        return [
            ChatMessage(
                role=Role.SYSTEM,
                content=system_prompt,
            ),
            *conversation.messages,
        ]