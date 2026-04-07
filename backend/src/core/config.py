from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Engineering RAG Assistant"
    app_env: str = "development"

    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    deepseek_api_key: str = "test-deepseek-key"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    data_directory: Path = Path("data")
    uploads_directory: Path = Path("data/uploads")
    processed_directory: Path = Path("data/processed")
    faiss_index_directory: Path = Path("storage/faiss")
    chat_history_db_path: Path = Path("storage/chat_history.sqlite3")

    retriever_top_k: int = Field(default=4, ge=1, le=50)
    rerank_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    rerank_lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    chunk_max_word_pieces: int = Field(default=220, ge=8, le=512)
    chunk_overlap_word_pieces: int = Field(default=40, ge=0, le=128)

    chat_history_max_messages: int = Field(default=8, ge=0, le=100)
    retrieval_history_user_messages: int = Field(default=3, ge=0, le=20)
    pdf_min_text_chars: int = Field(default=120, ge=0)
    pdf_garbled_char_ratio: float = Field(default=0.35, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for directory in (
            self.data_directory,
            self.uploads_directory,
            self.processed_directory,
            self.faiss_index_directory,
            self.chat_history_db_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

