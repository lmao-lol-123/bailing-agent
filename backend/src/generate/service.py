from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.encoders import jsonable_encoder

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import ChatMessage, ChatRole, RetrievalFilter
from backend.src.generate.prompts import SYSTEM_PROMPT, build_user_prompt
from backend.src.retrieve.store import VectorStoreService


class AnswerService:
    def __init__(
        self,
        settings: Settings,
        chat_client: Any,
        vector_store: VectorStoreService,
        chat_history_store: ChatHistoryStore,
    ) -> None:
        self._settings = settings
        self._chat_client = chat_client
        self._vector_store = vector_store
        self._chat_history_store = chat_history_store

    async def stream_answer(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        history_messages = self._chat_history_store.list_messages(
            session_id=session_id,
            limit=self._settings.chat_history_max_messages,
        )
        retrieval_query = self._build_retrieval_query(question=question, history_messages=history_messages)
        documents = self._vector_store.similarity_search(
            retrieval_query,
            k=top_k,
            metadata_filter=metadata_filter,
        )
        citations = self._vector_store.build_citations(documents)

        if not documents:
            fallback_text = "我无法从已索引文档中找到足够可靠的上下文来回答这个问题。"
            self._persist_turn(
                session_id=session_id,
                question=question,
                answer=fallback_text,
                grounded=False,
                citations=[],
            )
            yield {"event": "token", "data": {"text": fallback_text}}
            yield {"event": "sources", "data": {"citations": []}}
            yield {"event": "done", "data": {"grounded": False, "session_id": session_id}}
            return

        stream = await self._chat_client.chat.completions.create(
            model=self._settings.deepseek_model,
            temperature=0.1,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self._build_history_messages(history_messages),
                {"role": "user", "content": build_user_prompt(question, documents)},
            ],
        )

        answer_parts: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                answer_parts.append(delta)
                yield {"event": "token", "data": {"text": delta}}

        answer_text = "".join(answer_parts)
        self._persist_turn(
            session_id=session_id,
            question=question,
            answer=answer_text,
            grounded=True,
            citations=citations,
        )
        yield {"event": "sources", "data": {"citations": jsonable_encoder(citations)}}
        yield {"event": "done", "data": {"grounded": True, "session_id": session_id}}

    def _build_history_messages(self, history_messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role.value, "content": message.content} for message in history_messages]

    def _build_retrieval_query(self, question: str, history_messages: list[ChatMessage]) -> str:
        user_messages = [message.content for message in history_messages if message.role is ChatRole.USER]
        if not user_messages:
            return question

        recent_user_messages = user_messages[-self._settings.retrieval_history_user_messages :]
        return "\n".join([*recent_user_messages, question])

    def _persist_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        grounded: bool,
        citations: list[Any],
    ) -> None:
        self._chat_history_store.append_message(
            session_id=session_id,
            role=ChatRole.USER,
            content=question,
        )
        self._chat_history_store.append_message(
            session_id=session_id,
            role=ChatRole.ASSISTANT,
            content=answer,
            grounded=grounded,
            citations=list(citations),
        )
