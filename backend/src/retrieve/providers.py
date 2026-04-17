from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from backend.src.core.config import Settings


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        self.max_seq_length = int(getattr(self._model, "max_seq_length", 256) or 256)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def count_word_pieces(self, text: str) -> int:
        tokenizer: Any = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return len(text.split())
        return len(tokenizer.encode(text, add_special_tokens=False))

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def build_embedding_model(settings: Settings) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(settings.sentence_transformer_model)


def build_chat_client(settings: Settings) -> AsyncOpenAI:
    base_url = settings.deepseek_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=base_url,
    )
