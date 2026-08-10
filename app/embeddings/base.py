from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Erzeugt einen Embedding-Vektor für einen Text.
        """
        pass

    async def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Standardimplementierung für mehrere Texte.
        Provider können diese Methode überschreiben.
        """

        embeddings = []

        for text in texts:
            embeddings.append(
                await self.embed(text)
            )

        return embeddings