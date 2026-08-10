from pydantic import BaseModel, Field
from uuid import uuid4

from app.models.message import ChatMessage


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[ChatMessage] = Field(default_factory=list)