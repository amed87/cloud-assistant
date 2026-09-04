from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.models.role import Role
from app.prompts.base import PromptProvider
from app.prompts.steps.base import PromptStep
from app.services.rag_service import RagContext


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
        rag_context: RagContext | None = None,
    ) -> None:

        system_prompt = await self.prompt_provider.system_prompt()

        if rag_context is not None and rag_context.text:        
                    system_prompt += (
                        "\n\n"
                        "Nutze NUR den folgenden Kontext zur Beantwortung "
                        "der Frage. Wenn der Kontext keine ausreichende "
                        "Information enthält, sage 'Das weiß ich leider nicht.'.\n\n"
                        "KONTEXT:\n"
                        f"{rag_context.text}"
                    )

        messages.append(
            ChatMessage(
                role=Role.SYSTEM,
                content=system_prompt,
            )
        )