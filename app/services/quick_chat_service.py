import logging
from app.services.rag_service import RagService


logger = logging.getLogger(__name__)

# Chat service ohne KI-generierte Antworten, sondern nur mit Embedding-Suchen,
# für schnelle Antworten.
class QuickChatService:

    def __init__(
        self,
        rag_service: RagService,
    ):
        self.rag_service = rag_service

    async def ask(
        self,
        conversation_id: str,
        question: str,
    ) -> str:
        return await self.rag_service.quick_retrieve(question)
