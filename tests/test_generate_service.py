from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.core.config import Settings
from src.generate.service import AnswerService
from tests.conftest import FakeChatClient


class StubVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
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


@pytest.mark.asyncio
async def test_answer_service_streams_tokens_and_sources() -> None:
    service = AnswerService(
        settings=Settings(),
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
    )

    events = [event async for event in service.stream_answer("How do we stream answers?")]

    assert [event["event"] for event in events] == ["token", "token", "sources", "done"]
    assert events[0]["data"]["text"] == "Hello"
    assert events[2]["data"]["citations"][0]["source_name"] == "guide.md"

