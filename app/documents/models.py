from pydantic import BaseModel


class Document(BaseModel):
    id: str
    source: str
    content: str


class DocumentChunk(BaseModel):
    id: str

    source: str

    chunk_index: int

    content: str