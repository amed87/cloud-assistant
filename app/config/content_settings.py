from functools import lru_cache
from pathlib import Path
from app.config.settings import get_settings

import yaml
from pydantic import BaseModel

class ContentConfig(BaseModel):
    standard_fallback: str = "Keine relevanten Informationen gefunden."
    no_answer_fallback: str = "Keine Antwort gefunden."

@lru_cache
def load_content_config(path: str = get_settings().CONTENT_CONFIG_FILE) -> ContentConfig:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Content configuration not found: {file.resolve()}. "
            "Please create the file (see config/chatbot_texts.example.yaml)."
        )
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return ContentConfig(**data)