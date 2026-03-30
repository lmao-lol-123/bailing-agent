from __future__ import annotations

from functools import lru_cache

from src.core.config import Settings, get_settings
from src.generate.service import AnswerService
from src.ingest.service import IngestionService
from src.retrieve.providers import build_chat_client, build_embedding_model
from src.retrieve.store import VectorStoreService


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_model = build_embedding_model(settings)
        self.chat_client = build_chat_client(settings)
        self.vector_store = VectorStoreService(settings=settings, embeddings=self.embedding_model)
        self.ingestion_service = IngestionService(
            settings=settings,
            embeddings=self.embedding_model,
            vector_store=self.vector_store,
        )
        self.answer_service = AnswerService(
            settings=settings,
            chat_client=self.chat_client,
            vector_store=self.vector_store,
        )


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer(get_settings())
