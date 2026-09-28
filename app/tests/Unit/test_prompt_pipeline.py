import pytest

from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.prompts.default import DefaultPromptProvider
from app.prompts.pipeline import PromptPipeline
from app.prompts.steps.conversation import ConversationStep
from app.prompts.steps.system_prompt import SystemPromptStep


@pytest.mark.asyncio
async def test_prompt_pipeline_builds_system_prompt_and_conversation() -> None:
    conversation = Conversation(
        id="1",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="system", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ],
    )

    pipeline = PromptPipeline(
        [
            SystemPromptStep(DefaultPromptProvider()),
            ConversationStep(),
        ]
    )

    prompt = await pipeline.build(conversation)

    assert len(prompt) == len(conversation.messages) + 1
    assert prompt[0].role == "system"
    assert "Impuls GmbH" in prompt[0].content
    assert prompt[1:] == conversation.messages