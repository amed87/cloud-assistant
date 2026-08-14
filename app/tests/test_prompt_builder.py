import asyncio

from app.models.conversation import Conversation
from app.models.message import ChatMessage
from app.prompts.builder import PromptBuilder
from app.prompts.default import DefaultPromptProvider
from app.prompts.pipeline import PromptPipeline
from app.prompts.steps.conversation import ConversationStep
from app.prompts.steps.system_prompt import SystemPromptStep


async def test_prompt_builder_builds_expected_prompt_with_default_system_prompt(prompt_builder: PromptBuilder):
    conversation = Conversation(id="1", messages=[
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="system", content="Hi there!"),
        ChatMessage(role="user", content="How are you?"),
    ])
    prompt = await prompt_builder.build(conversation)

    expected = (
        [ChatMessage(role= 'system', content='\nDu bist ein Mitarbeiter vom Impuls GmbH.\n\nRegeln:\n\n- Antworte ausschließlich auf Deutsch.\n- Gib präzise technische Antworten.\n- Wenn du dir nicht sicher bist, sage das.\n- Erkläre komplexe Themen schrittweise.\n'), 
         ChatMessage(role= 'user', content='Hello'), 
         ChatMessage(role= 'system', content='Hi there!'), 
         ChatMessage(role= 'user', content='How are you?'),
         ChatMessage(role= 'user', content='Hello'),
         ChatMessage(role= 'system', content='Hi there!'),
         ChatMessage(role= 'user', content='How are you?')]
    )
    
    print(f"Prompt: {prompt}\nExpected: {expected}")
    assert all(p.content == e.content and p.role == e.role for p, e in zip(prompt, expected))


async def main():
    prompt_provider = DefaultPromptProvider()
    pipeline = PromptPipeline(
        [
            SystemPromptStep(prompt_provider),
            ConversationStep(),
        ]
    )
    prompt_builder = PromptBuilder(
        pipeline,
    )

    await test_prompt_builder_builds_expected_prompt_with_default_system_prompt(prompt_builder)
    print("Prompt builder test passed!")

if __name__ == "__main__":
    asyncio.run(main())