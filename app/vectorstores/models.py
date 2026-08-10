from typing import Any

from pydantic import BaseModel, Field


class VectorDocument(BaseModel):
    id: str
    content: str
    embedding: list[float]

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SearchResult(BaseModel):
    id: str
    content: str
    score: float

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )