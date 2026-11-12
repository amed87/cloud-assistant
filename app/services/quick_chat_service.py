import logging
from app.services.rag_service import RagService


logger = logging.getLogger(__name__)

# Chat service without AI-generated responses; it relies on embedding search
# to return answers quickly.
class QuickChatService:
    """Answer chat requests directly from FAQ retrieval without LLM generation."""

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
        """Return the best accepted FAQ answer or the configured fallback."""
        return await self.rag_service.quick_retrieve(question)
