from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.dependencies import get_container
from src.core.models import DocumentSummary, IngestResult, SourceType


class FakeIngestionService:
    def save_upload(self, file_name: str, payload: bytes) -> Path:
        return Path(file_name)

    def ingest_saved_file(self, file_path: Path, force_mineru: bool = False) -> IngestResult:
        return IngestResult(
            source_name=file_path.name,
            source_type=SourceType.TXT,
            source_uri_or_path=str(file_path),
            doc_id="doc-1",
            documents_loaded=1,
            chunks_indexed=1,
        )

    def ingest_url(self, url: str) -> IngestResult:
        return IngestResult(
            source_name="example.com",
            source_type=SourceType.WEB,
            source_uri_or_path=url,
            doc_id="doc-url",
            documents_loaded=1,
            chunks_indexed=1,
        )


class FakeVectorStore:
    def list_documents(self) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                doc_id="doc-1",
                source_name="guide.md",
                source_type=SourceType.MARKDOWN,
                source_uri_or_path="guide.md",
                chunk_count=2,
                page_or_sections=["1", "2"],
            )
        ]


class FakeAnswerService:
    async def stream_answer(self, question: str, top_k: int | None = None):
        yield {"event": "token", "data": {"text": "Hello"}}
        yield {"event": "sources", "data": {"citations": [{"index": 1, "source_name": "guide.md"}]}}
        yield {"event": "done", "data": {"grounded": True}}


class FakeContainer:
    def __init__(self) -> None:
        self.ingestion_service = FakeIngestionService()
        self.vector_store = FakeVectorStore()
        self.answer_service = FakeAnswerService()


def test_api_endpoints() -> None:
    app.dependency_overrides[get_container] = lambda: FakeContainer()
    client = TestClient(app)

    frontend_response = client.get("/")
    health_response = client.get("/health")
    ingest_response = client.post(
        "/ingest/files",
        files=[("files", ("notes.txt", b"hello world", "text/plain"))],
    )
    url_response = client.post("/ingest/url", json={"url": "https://example.com"})
    documents_response = client.get("/documents")
    stream_response = client.post("/ask/stream", json={"question": "hello"})

    assert frontend_response.status_code == 200
    assert "工程文档问答工作台" in frontend_response.text
    assert health_response.status_code == 200
    assert ingest_response.json()["results"][0]["source_name"] == "notes.txt"
    assert url_response.json()["results"][0]["source_type"] == "web"
    assert documents_response.json()["documents"][0]["source_name"] == "guide.md"
    assert "event: token" in stream_response.text
    assert "event: sources" in stream_response.text

    app.dependency_overrides.clear()

