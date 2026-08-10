from fastapi import APIRouter, Body, Depends

from app.dependencies.teams import get_teams_service
from app.teams.mapper import TeamsMapper
from app.teams.service import TeamsService

router = APIRouter(
    prefix="/teams",
    tags=["Teams"],
)

mapper = TeamsMapper()


@router.post("/messages")
async def receive_message(
    payload: dict = Body(...),
    service: TeamsService = Depends(get_teams_service),
):

    message = mapper.from_payload(payload)

    response = await service.process_message(
        message,
    )

    return response