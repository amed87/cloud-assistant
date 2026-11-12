from pydantic import BaseModel, Field
from typing import Any, List, Dict

class FAQHealthCheckRequest(BaseModel):
    """Request body for the FAQ health-check endpoint."""
    
    file_path: str | None = None

class SelfRetrievalFailure(BaseModel):
    """A self-retrieval failure containing the FAQ question."""
    id: str
    question: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DuplicatePair(BaseModel):
    """A semantically similar FAQ pair and its questions."""
    id_1: str
    question_1: str
    id_2: str
    question_2: str
    distance: float
    notes: str | None = None
    semantics_issue: bool = True  # A low distance indicates a semantic issue.
    intent_separatable: bool = False  # Different question types can be separated by the reranker.

class LowMarginQuery(BaseModel):
    """A query with a small score margin between ranks one and two."""
    query_id: str
    query_question: str
    rank_1_id: str
    rank_1_question: str
    rank_1_score: float
    rank_2_id: str
    rank_2_question: str
    rank_2_score: float
    margin: float

class SelfRetrievalReport(BaseModel):
    total_documents: int
    passed_count: int
    rate: float
    failed_entries: List[SelfRetrievalFailure] = Field(
        default_factory=list,
        description="Self-retrieval failures, including their questions."
    )
    
    def passes(self, min_rate: float = 0.95) -> bool:
        return self.rate >= min_rate

class DuplicateReport(BaseModel):
    duplicate_pairs: List[DuplicatePair] = Field(
        default_factory=list,
        description="Semantically similar FAQ pairs, including their questions."
    )
    affected_count: int = 0
    
    def passes(self, max_duplicates: int = 0) -> bool:
        return len(self.duplicate_pairs) <= max_duplicates

class RetrievalMarginReport(BaseModel):
    avg_margin: float = 0.0
    min_margin: float = 0.0
    below_threshold_count: int = 0
    low_margin_entries: List[LowMarginQuery] = Field(
        default_factory=list,
        description="Queries with a margin below 0.1, including their questions."
    )
    
    def passes(self, min_margin: float = 0.1) -> bool:
        return self.below_threshold_count == 0

class EmbeddingHealthReport(BaseModel):
    total_vectors: int = 0
    nan_count: int = 0
    zero_count: int = 0
    extreme_values_count: int = 0
    health_rate: float = 1.0
    unhealthy_ids: List[str] = Field(
        default_factory=list,
        description="IDs with problematic embeddings."
    )
    
    def passes(self, min_health_rate: float = 1.0) -> bool:
        return self.health_rate >= min_health_rate

class FAQHealthCheckReport(BaseModel):
    status: str  # "PASS" | "FAIL" | "WARNING"
    self_retrieval: SelfRetrievalReport
    duplicates: DuplicateReport
    retrieval_margins: RetrievalMarginReport
    embedding_health: EmbeddingHealthReport
    messages: List[str] = Field(default_factory=list)
    checked_at: str = ""

    class Config:
        arbitrary_types_allowed = True