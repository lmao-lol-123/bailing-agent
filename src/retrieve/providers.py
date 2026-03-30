from __future__ import annotations

from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings

from src.core.config import Settings


def build_embedding_model(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )


def build_chat_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url.rstrip("/") + "/v1",
    )

