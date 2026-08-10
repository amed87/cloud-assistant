from pydantic import BaseModel

from app.models.role import Role


class ChatMessage(BaseModel):
    role: Role
    content: str