from app.teams.service import TeamsService


class TeamsController:

    def __init__(
        self,
        service: TeamsService,
    ):
        self.service = service