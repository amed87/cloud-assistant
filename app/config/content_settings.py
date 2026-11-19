from functools import lru_cache
from pathlib import Path
import re
from app.config.settings import get_settings

import yaml
from pydantic import BaseModel


class CentroidFacet(BaseModel):
    """One classification facet over centroid embeddings.

    Declared under ``centroid_facets`` in content_config.yaml. Reranker
    matching compares the query-side label against the FAQ metadata field
    ``metadata_key``; ``dm_prompt`` is the question text for the
    decision-model fallback call when the centroid cascade is uncertain.
    """

    metadata_key: str
    fallback: str = ""
    multi_label: bool = False
    dm_prompt: str = ""


class ContentConfig(BaseModel):
    """User-facing text and prompt configuration loaded from YAML."""

    standard_fallback: str = "Keine relevanten Informationen gefunden."
    no_answer_fallback: str = "Keine Antwort gefunden."
    fragetyp_filter: dict[str, re.Pattern] = {}
    fragetyp_fallback: str = "Generell"
    centroid_facets: dict[str, CentroidFacet] = {}
    decision_model_prompt: str = "Welche dieser Fragen hat dieselbe Bedeutung wie die Anfrage?"
    decision_model_prompts: dict[str, str] = {
        "matched_faq": 'Which candidate question is a structurally valid paraphrase that shares the identical information requirement, constraints and intent as the user query?'
        }
@lru_cache
def load_content_config(path: str = get_settings().CONTENT_CONFIG_FILE) -> ContentConfig:
    """Load and validate the content configuration from ``path``."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Content configuration not found: {file.resolve()}. "
            "Please create the file (see config/chatbot_texts.example.yaml)."
        )
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return ContentConfig(**data)

content_config = load_content_config()