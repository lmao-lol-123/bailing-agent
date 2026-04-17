from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.core.dependencies import get_container
from backend.src.core.models import (
    ChatMessage,
    ChatRole,
    ChatSessionSummary,
    DocumentSummary,
    IngestResult,
    RetrievalFilter,
    SessionFileStatus,
    SessionFileSummary,
    SourceType,
)


class FakeIngestionService:
    def __init__(self) -> None:
        self.deleted_doc_ids: list[str] = []
        self.deleted_session_files: list[tuple[str, str]] = []
        self.recovered_session_files: list[tuple[str, str]] = []
        self.deleted_sessions: list[str] = []

    def save_upload(self, file_name: str, payload: bytes) -> Path:
        return Path(file_name)

    def ingest_saved_file(
        self,
        file_path: Path,
        force_mineru: bool = False,
        *,
        scope: str = "global",
        session_id: str | None = None,
        is_sensitive: bool | None = None,
        mineru_mode: str | None = None,
        source_name: str | None = None,
        doc_id_override: str | None = None,
    ) -> IngestResult:
        return IngestResult(
            source_name=source_name or file_path.name,
            source_type=SourceType.TXT,
            source_uri_or_path=str(file_path),
            doc_id=doc_id_override or "doc-1",
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

    def ingest_session_upload(
        self,
        *,
        session_id: str,
        file_name: str,
        payload: bytes,
        force_mineru: bool = False,
        is_sensitive: bool | None = None,
        mineru_mode: str | None = None,
    ) -> SessionFileSummary:
        return SessionFileSummary(
            file_id="file-1",
            session_id=session_id,
            doc_id="doc-session",
            source_name=file_name,
            source_uri_or_path=file_name,
            doc_type=SourceType.TXT,
            status=SessionFileStatus.INDEXED,
        )

    def delete_documents(self, doc_ids: list[str]) -> None:
        self.deleted_doc_ids.extend(doc_ids)

    def delete_session(self, *, session_id: str) -> None:
        self.deleted_sessions.append(session_id)

    def delete_session_file(self, *, session_id: str, file_id: str) -> SessionFileSummary | None:
        self.deleted_session_files.append((session_id, file_id))
        return SessionFileSummary(
            file_id=file_id,
            session_id=session_id,
            doc_id="doc-session",
            source_name="notes.txt",
            source_uri_or_path="notes.txt",
            doc_type=SourceType.TXT,
            status=SessionFileStatus.INDEXED,
        )

    def recover_session_file_deletion(self, *, session_id: str, file_id: str) -> str:
        self.recovered_session_files.append((session_id, file_id))
        return "already_deleted"


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


class FakeChatHistoryStore:
    def __init__(self) -> None:
        self.created_session_ids: list[str] = []
        self.renamed_sessions: list[tuple[str, str]] = []
        self.deleted_sessions: list[str] = []
        self.session_files = [
            SessionFileSummary(
                file_id="file-1",
                session_id="session-cookie",
                doc_id="doc-session",
                source_name="notes.txt",
                source_uri_or_path="notes.txt",
                doc_type=SourceType.TXT,
                status=SessionFileStatus.INDEXED,
            )
        ]

    def ensure_session(self, session_id: str | None) -> str:
        resolved = session_id or "session-cookie"
        self.created_session_ids.append(resolved)
        return resolved

    def rename_session(self, session_id: str, title: str) -> None:
        self.renamed_sessions.append((session_id, title))

    def delete_session(self, session_id: str) -> list[SessionFileSummary]:
        self.deleted_sessions.append(session_id)
        return [item for item in self.session_files if item.session_id == session_id]

    def list_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        return [
            ChatMessage(
                session_id=session_id,
                role=ChatRole.USER,
                content="hello",
            )
        ]

    def list_sessions(self) -> list[ChatSessionSummary]:
        return [
            ChatSessionSummary(
                session_id="session-cookie",
                title="hello",
                message_count=1,
            )
        ]

    def list_session_files(self, session_id: str) -> list[SessionFileSummary]:
        return [item for item in self.session_files if item.session_id == session_id]


class FakeAnswerService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None, RetrievalFilter | None]] = []

    async def stream_answer(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ):
        self.calls.append((session_id, question, top_k, metadata_filter))
        yield {"event": "token", "data": {"text": "Hello"}}
        yield {"event": "sources", "data": {"citations": [{"index": 1, "source_name": "guide.md"}]}}
        yield {"event": "done", "data": {"grounded": True, "session_id": session_id}}


class FakeContainer:
    def __init__(self) -> None:
        self.ingestion_service = FakeIngestionService()
        self.vector_store = FakeVectorStore()
        self.chat_history_store = FakeChatHistoryStore()
        self.answer_service = FakeAnswerService()


def test_api_endpoints() -> None:
    fake_container = FakeContainer()
    app.dependency_overrides[get_container] = lambda: fake_container
    client = TestClient(app)

    frontend_response = client.get("/")
    health_response = client.get("/health")
    ingest_response = client.post(
        "/ingest/files",
        files=[("files", ("notes.txt", b"hello world", "text/plain"))],
    )
    url_response = client.post("/ingest/url", json={"url": "https://example.com"})
    documents_response = client.get("/documents")
    stream_response = client.post(
        "/ask/stream",
        json={
            "question": "hello",
            "metadata_filter": {"doc_types": ["markdown"]},
        },
    )
    sessions_response = client.get("/sessions")
    rename_response = client.patch("/sessions/session-cookie", json={"title": "Renamed"})
    history_response = client.get("/sessions/session-cookie/messages")
    session_files_response = client.get("/sessions/session-cookie/files")
    session_file_upload_response = client.post(
        "/sessions/session-cookie/files",
        files={"file": ("notes.txt", b"hello session", "text/plain")},
    )
    session_file_delete_response = client.delete("/sessions/session-cookie/files/file-1")
    session_file_recover_response = client.post(
        "/sessions/session-cookie/files/file-1/recover-delete"
    )
    delete_response = client.delete("/sessions/session-cookie")

    assert frontend_response.status_code == 200
    assert health_response.status_code == 200
    assert ingest_response.json()["results"][0]["source_name"] == "notes.txt"
    assert url_response.json()["results"][0]["source_type"] == "web"
    assert documents_response.json()["documents"][0]["source_name"] == "guide.md"
    assert "event: token" in stream_response.text
    assert "event: sources" in stream_response.text
    assert sessions_response.json()["sessions"][0]["title"] == "hello"
    assert rename_response.json()["status"] == "ok"
    assert fake_container.chat_history_store.renamed_sessions == [("session-cookie", "Renamed")]
    assert stream_response.headers["set-cookie"].startswith("rag_session_id=session-cookie;")
    assert fake_container.answer_service.calls[0] == (
        "session-cookie",
        "hello",
        None,
        RetrievalFilter(doc_types=[SourceType.MARKDOWN]),
    )
    assert history_response.json()["messages"][0]["session_id"] == "session-cookie"
    assert session_files_response.json()["files"][0]["file_id"] == "file-1"
    assert session_files_response.json()["files"][0]["status"] == "indexed"
    assert session_file_upload_response.json()["doc_id"] == "doc-session"
    assert session_file_upload_response.json()["status"] == "indexed"
    assert session_file_delete_response.json()["status"] == "ok"
    assert session_file_recover_response.json()["status"] == "already_deleted"
    assert fake_container.ingestion_service.deleted_session_files == [("session-cookie", "file-1")]
    assert fake_container.ingestion_service.recovered_session_files == [
        ("session-cookie", "file-1")
    ]
    assert delete_response.json()["status"] == "ok"
    assert fake_container.chat_history_store.deleted_sessions == []
    assert fake_container.ingestion_service.deleted_sessions == ["session-cookie"]
    assert fake_container.ingestion_service.deleted_doc_ids == []

    follow_up_response = client.post("/ask/stream", json={"question": "follow up"})

    assert fake_container.answer_service.calls[1] == ("session-cookie", "follow up", None, None)
    assert "event: done" in follow_up_response.text

    app.dependency_overrides.clear()
