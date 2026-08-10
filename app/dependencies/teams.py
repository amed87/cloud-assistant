from app.dependencies.chat import get_chat_service
from app.teams.service import TeamsService


def get_teams_service() -> TeamsService:
    return TeamsService(
        chat_service=get_chat_service(),
    )