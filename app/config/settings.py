from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Cloud Assistant"
    app_version: str = "0.1.0"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma:2b"

    log_level: str = "INFO"

    CHROMA_PATH: str = "./data/chroma"
    CHROMA_COLLECTION: str = "cloud-assistant"

    RAG_TOP_K: int = 5
    RAG_MIN_SCORE: float = 0.3

    enable_profiler: bool = True
    profiler_sample_rate: float = 0.1
    profiler_output_path: str = "./load-profiler.html"
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()