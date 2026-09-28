from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Cloud Assistant"
    app_version: str = "0.1.0"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma:2b"
    ollama_num_threads: str = "4"
    OLLAMA_NUM_CTX: str = "2048"

    log_level: str = "INFO"

    FAQ_INPUT_FILE: str = "./data/faq.csv"
    CONTENT_CONFIG_FILE: str = "./content_config.yaml"

    NUMBER_OF_TURNS: int = 5

    CHROMA_PATH: str = "./data/chroma"
    CHROMA_COLLECTION: str = "cloud-assistant"

    RAG_TOP_K: int = 5
    RAG_MIN_SCORE: float = 0.3
    RAG_MIN_SCORE_GAP: float = 0.1

    ENABLE_PROFILER: bool = True
    PROFILER_SAMPLE_RATE: float = 0.1
    PROFILER_OUTPUT_PATH: str = "./load-profiler.html"
    PROFILER_OUTPUT_FORMAT: str = "html"
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()