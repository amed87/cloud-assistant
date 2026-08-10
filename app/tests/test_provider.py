import asyncio

from app.models.message import ChatMessage
from app.models.role import Role
from app.providers.ollama_provider import OllamaProvider


async def main():
    provider = OllamaProvider()

    messages = [
        ChatMessage(
            role=Role.USER,
            content="Erkläre Kubernetes in zwei Sätzen."
        )
    ]

    async for token in provider.stream_chat(messages):
        print(token, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())