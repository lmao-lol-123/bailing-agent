from __future__ import annotations

from functools import lru_cache

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings, get_settings
from backend.src.generate.service import AnswerService
from backend.src.ingest.service import IngestionService
from backend.src.retrieve.providers import build_chat_client, build_embedding_model
from backend.src.retrieve.index_manager import IndexManager
from backend.src.retrieve.store import VectorStoreService


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_model = build_embedding_model(settings)
        self.chat_client = build_chat_client(settings)
        self.vector_store = VectorStoreService(settings=settings, embeddings=self.embedding_model)
        self.index_manager = IndexManager(settings=settings, embeddings=self.embedding_model, vector_store=self.vector_store)
        self.vector_store.set_index_manager(self.index_manager)
        self.chat_history_store = ChatHistoryStore(settings)
        self.ingestion_service = IngestionService(
            settings=settings,
            embeddings=self.embedding_model,
            vector_store=self.vector_store,
            index_manager=self.index_manager,
        )
        self.answer_service = AnswerService(
            settings=settings,
            chat_client=self.chat_client,
            vector_store=self.vector_store,
            chat_history_store=self.chat_history_store,
        )


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer(get_settings())







