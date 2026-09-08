from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    source: str


class TextDocument(Document):
    content: str


class DocumentChunk(BaseModel):
    id: str
    source: str
    chunk_index: int
    content: str


class FAQEntry(BaseModel):
    id: str
    question: str
    answer: str
    keywords: list[str] = Field(default_factory=list)


class FAQDocument(Document):
    entries: list[FAQEntry]