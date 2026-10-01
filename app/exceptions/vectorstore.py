from app.exceptions.base import FaqBotError


class VectorStoreError(FaqBotError):
    """Error raised while accessing the vector store."""