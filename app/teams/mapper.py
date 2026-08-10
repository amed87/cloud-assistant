from app.teams.models import TeamsMessage


class TeamsMapper:

    def from_payload(
        self,
        payload: dict,
    ) -> TeamsMessage:

        conversation = payload.get("conversation", {})
        user = payload.get("from", {})

        return TeamsMessage(
            conversation_id=conversation.get("id", ""),
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            text=payload.get("text", ""),
        )