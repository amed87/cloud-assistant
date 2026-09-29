import pytest
from app.documents.chunker import DocumentChunker
from app.documents.models import TextDocument
from app.documents.faq_importer import FAQImporter
from app.documents.models import FAQDocument, FAQEntry
from app.documents.text_importer import TextImporter
from app.vectorstores.models import VectorDocument

class MockEmbeddingProvider:
    def __init__(self):
        self.embedded_texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return [1.0, 2.0, 3.0]

class MockVectorStore:

    def __init__(self, documents: list[VectorDocument] | None = None):
        self.documents = list(documents or [])

    def get(self) -> dict[str, list]:
        return {
            "ids": [document.id for document in self.documents],
            "metadatas": [document.metadata for document in self.documents],
        }

    async def add(
        self,
        documents: list[VectorDocument],
    ) -> None:

        self.documents.extend(documents)

    async def delete(self, ids: list[str]) -> None:
        self.documents = [document for document in self.documents if document.id not in ids]

@pytest.mark.asyncio
async def test_text_importer_imports_document() -> None:
    document = TextDocument(
        id="test",
        source="test.txt",
        content=(
            "Kubernetes ist eine Plattform zur "
            "Orchestrierung von Containern."
        ),
    )

    chunker = DocumentChunker(
        chunk_size=1000,
        overlap=200,
    )

    vector_store = MockVectorStore()

    importer = TextImporter(
        chunker=chunker,
        embedding_provider=MockEmbeddingProvider(),
        vector_store=vector_store,
    )

    await importer.import_data(document)

    assert len(vector_store.documents) == 1
    assert vector_store.documents[0].content == document.content
    assert vector_store.documents[0].embedding == [1.0, 2.0, 3.0]

@pytest.mark.asyncio
async def test_faq_importer_imports_entries() -> None:
    faq = FAQDocument(
        id="faq",
        source="faq.csv",
        entries=[
            FAQEntry(
                id="entry-1",
                question="Was ist Kubernetes?",
                answer="Eine Plattform zur Container-Orchestrierung.",
                question_type="Definition",
                keywords=["kubernetes", "container"],
            ),
        ],
    )

    vector_store = MockVectorStore()

    importer = FAQImporter(
        embedding_provider=MockEmbeddingProvider(),
        vector_store=vector_store,
    )

    await importer.import_data(faq)

    assert len(vector_store.documents) == 1

    vector_document = vector_store.documents[0]

    assert vector_document.content == faq.entries[0].question
    assert vector_document.embedding == [1.0, 2.0, 3.0]
    assert vector_document.metadata["answer"] == faq.entries[0].answer
    assert vector_document.metadata["keywords"] == faq.entries[0].keywords


@pytest.mark.asyncio
async def test_faq_importer_imports_new_ids_and_deletes_obsolete_faq_ids() -> None:
    faq = FAQDocument(
        id="faq",
        source="data/faq.csv",
        entries=[
            FAQEntry(
                id="entry-keep",
                question="Bestehender Eintrag",
                answer="Antwort",
                question_type="Definition",
            ),
            FAQEntry(
                id="entry-new",
                question="Neuer Eintrag",
                answer="Neue Antwort",
                question_type="Definition",
            ),
        ],
    )
    vector_store = MockVectorStore([
        VectorDocument(
            id="entry-keep",
            content="Bestehender Eintrag",
            embedding=[1.0],
            metadata={"source": "data\\faq.csv"},
        ),
        VectorDocument(
            id="entry-obsolete",
            content="Entfernter Eintrag",
            embedding=[1.0],
            metadata={"source": "data\\faq.csv"},
        ),
        VectorDocument(
            id="other-source-entry",
            content="Textimport",
            embedding=[1.0],
            metadata={"source": "handbuch.txt"},
        ),
    ])
    embedding_provider = MockEmbeddingProvider()
    importer = FAQImporter(embedding_provider, vector_store)

    await importer.import_data(faq)

    assert {document.id for document in vector_store.documents} == {
        "entry-keep",
        "entry-new",
        "other-source-entry",
    }
    assert embedding_provider.embedded_texts == ["Neuer Eintrag"]