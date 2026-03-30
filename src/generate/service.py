from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.encoders import jsonable_encoder

from src.core.config import Settings
from src.retrieve.store import VectorStoreService
from src.generate.prompts import SYSTEM_PROMPT, build_user_prompt


class AnswerService:
    def __init__(self, settings: Settings, chat_client: Any, vector_store: VectorStoreService) -> None:
        self._settings = settings
        self._chat_client = chat_client
        self._vector_store = vector_store

    async def stream_answer(self, question: str, top_k: int | None = None) -> AsyncIterator[dict[str, Any]]:
        documents = self._vector_store.similarity_search(question, k=top_k)
        citations = self._vector_store.build_citations(documents)

        if not documents:
            yield {"event": "token", "data": {"text": "我无法从提供的资料中确认答案。"}}
            yield {"event": "sources", "data": {"citations": []}}
            yield {"event": "done", "data": {"grounded": False}}
            return

        stream = await self._chat_client.chat.completions.create(
            model=self._settings.deepseek_model,
            temperature=0.1,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, documents)},
            ],
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield {"event": "token", "data": {"text": delta}}

        yield {"event": "sources", "data": {"citations": jsonable_encoder(citations)}}
        yield {"event": "done", "data": {"grounded": True}}
