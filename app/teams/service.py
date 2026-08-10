from app.services.chat_service import ChatService
from app.teams.models import TeamsMessage, TeamsResponse


class TeamsService:

    def __init__(
        self,
        chat_service: ChatService,
    ):
        self.chat_service = chat_service

    async def process_message(
        self,
        message: TeamsMessage,
    ) -> TeamsResponse:

        answer = await self.chat_service.ask(
            conversation_id=message.conversation_id,
            question=message.text,
        )

        return TeamsResponse(
            conversation_id=message.conversation_id,
            text=answer,
        )