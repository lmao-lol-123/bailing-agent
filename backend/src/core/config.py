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
    index_state_directory: Path = Path("storage/index")
    index_name: str = "default"
    chat_history_db_path: Path = Path("storage/chat_history.sqlite3")

    retriever_top_k: int = Field(default=4, ge=1, le=50)
    rerank_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    rerank_lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    hybrid_lexical_enabled: bool = True
    hybrid_dense_k: int = Field(default=30, ge=1, le=100)
    hybrid_lexical_k: int = Field(default=30, ge=1, le=100)
    hybrid_fusion_top_k: int = Field(default=20, ge=1, le=100)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=200)
    rerank_mode: str = "heuristic"
    rerank_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_cross_encoder_top_k: int = Field(default=20, ge=1, le=100)

    query_routing_enabled: bool = True
    query_multi_enabled: bool = True
    query_multi_max_variants: int = Field(default=4, ge=1, le=6)
    query_route_exploration_token_threshold: int = Field(default=5, ge=1, le=20)
    query_route_precision_literal_ratio_threshold: float = Field(default=0.34, ge=0.0, le=1.0)
    query_route_structure_pattern_enabled: bool = True

    targeted_retrieval_enabled: bool = True
    targeted_retry_enabled: bool = True
    targeted_retry_max_attempts: int = Field(default=1, ge=0, le=3)
    targeted_min_results_ratio: float = Field(default=0.75, ge=0.1, le=1.0)
    targeted_soft_bias_weight: float = Field(default=0.06, ge=0.0, le=0.5)
    targeted_hard_min_candidates: int = Field(default=2, ge=1, le=20)
    targeted_retry_dense_multiplier: float = Field(default=1.5, ge=1.0, le=4.0)
    targeted_retry_lexical_multiplier: float = Field(default=1.5, ge=1.0, le=4.0)

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
            self.index_state_directory,
            self.chat_history_db_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
