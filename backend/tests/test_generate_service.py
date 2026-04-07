from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import ChatRole, RetrievalFilter, SourceType
from backend.src.generate.service import AnswerService
from backend.tests.conftest import FakeChatClient


class StubVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.queries: list[tuple[str, int | None, RetrievalFilter | None]] = []

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[Document]:
        self.queries.append((query, k, metadata_filter))
        return self._documents

    @staticmethod
    def build_citations(documents: list[Document]) -> list[dict]:
        return [
            {
                "index": 1,
                "doc_id": "doc-1",
                "source_name": "guide.md",
                "source_uri_or_path": "guide.md",
                "page_or_section": "1",
                "snippet": "FastAPI streams answers using SSE.",
            }
        ]


def _chat_db_path() -> Path:
    base = Path("backend/.pytest-tmp") / "chat_history_tests"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{uuid4().hex}.sqlite3"


@pytest.mark.asyncio
async def test_answer_service_streams_tokens_and_sources() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-stream")

    service = AnswerService(
        settings=settings,
        chat_client=FakeChatClient(["Hello", " world"]),
        vector_store=StubVectorStore(
            [
                Document(
                    page_content="FastAPI streams answers using SSE.",
                    metadata={
                        "doc_id": "doc-1",
                        "source_name": "guide.md",
                        "source_uri_or_path": "guide.md",
                        "page_or_section": "1",
                    },
                )
            ]
        ),
        chat_history_store=history_store,
    )

    events = [
        event
        async for event in service.stream_answer(
            session_id=session_id,
            question="How do we stream answers?",
        )
    ]

    assert [event["event"] for event in events] == ["token", "token", "sources", "done"]
    assert events[0]["data"]["text"] == "Hello"
    assert events[2]["data"]["citations"][0]["source_name"] == "guide.md"


@pytest.mark.asyncio
async def test_answer_service_uses_chat_history_for_context_and_persists_turns() -> None:
    settings = Settings(
        chat_history_db_path=_chat_db_path(),
        chat_history_max_messages=6,
    )
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-follow-up")
    history_store.append_message(session_id=session_id, role=ChatRole.USER, content="How do we stream answers?")
    history_store.append_message(session_id=session_id, role=ChatRole.ASSISTANT, content="Use SSE.", grounded=True)

    vector_store = StubVectorStore(
        [
            Document(
                page_content="FastAPI streams answers using SSE.",
                metadata={
                    "doc_id": "doc-1",
                    "source_name": "guide.md",
                    "source_uri_or_path": "guide.md",
                    "page_or_section": "1",
                },
            )
        ]
    )
    chat_client = FakeChatClient(["Follow-up", " answer"])
    service = AnswerService(
        settings=settings,
        chat_client=chat_client,
        vector_store=vector_store,
        chat_history_store=history_store,
    )
    metadata_filter = RetrievalFilter(doc_types=[SourceType.MARKDOWN])

    events = [
        event
        async for event in service.stream_answer(
            session_id=session_id,
            question="What should the client set as the response media type?",
            top_k=3,
            metadata_filter=metadata_filter,
        )
    ]

    assert [event["event"] for event in events] == ["token", "token", "sources", "done"]
    assert "How do we stream answers?" in vector_store.queries[0][0]
    assert vector_store.queries[0][1] == 3
    assert vector_store.queries[0][2] == metadata_filter
    sent_messages = chat_client.completions.calls[0]["messages"]
    assert sent_messages[1]["role"] == "user"
    assert sent_messages[1]["content"] == "How do we stream answers?"
    assert sent_messages[2]["role"] == "assistant"
    assert sent_messages[2]["content"] == "Use SSE."

    stored_messages = history_store.list_messages(session_id)
    assert [message.role for message in stored_messages] == [
        ChatRole.USER,
        ChatRole.ASSISTANT,
        ChatRole.USER,
        ChatRole.ASSISTANT,
    ]
    assert stored_messages[-1].content == "Follow-up answer"
