from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies.chat import get_chat_service
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    answer = await service.ask(
        conversation_id=request.conversation_id,
        question=request.message,
    )

    return ChatResponse(answer=answer)

@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    return StreamingResponse(
        service.stream_answer(
            conversation_id=request.conversation_id,
            question=request.message,
        ),
        media_type="text/plain",
    )