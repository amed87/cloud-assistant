from pydantic import BaseModel


class TeamsMessage(BaseModel):
    conversation_id: str
    user_id: str
    user_name: str
    text: str


class TeamsResponse(BaseModel):
    conversation_id: str
    text: str