from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import (
    IngestedChunk,
    NormalizedDocument,
    RetrievalFilter,
    SessionFileStatus,
    SourceType,
)
from backend.src.generate.service import AnswerService
from backend.src.ingest.service import (
    IngestionService,
    SessionFileCleanupError,
    SessionFileRecoveryStateError,
)
from backend.src.retrieve.index_manager import DocumentDeleteStepError
from backend.tests.conftest import FakeChatClient


class SessionScopedVectorStore:
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

    def similarity_search_with_trace(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> tuple[list[Document], dict]:
        return self.similarity_search(query=query, k=k, metadata_filter=metadata_filter), {
            "request": {"route_name": "general", "retry_reason": "none"},
            "retrieval": {},
            "generation": {},
        }

    @staticmethod
    def build_citations(documents: list[Document]) -> list[dict]:
        return [
            {
                "index": 1,
                "doc_id": "doc-session",
                "source_name": "session-guide.md",
                "source_uri_or_path": "session-guide.md",
                "page_or_section": "1",
                "snippet": "Session-scoped document.",
            }
        ]


def _chat_db_path() -> Path:
    base = Path("backend/.pytest-tmp") / "session_file_tests"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{uuid4().hex}.sqlite3"


def _ingest_settings() -> Settings:
    case_dir = Path("backend/.pytest-tmp") / f"session_ingest_{uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        chat_history_db_path=case_dir / "chat.sqlite3",
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
    )
    settings.ensure_directories()
    return settings


@pytest.mark.asyncio
async def test_answer_service_limits_retrieval_to_session_files() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-files")
    history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-session",
        source_name="session-guide.md",
        source_uri_or_path="session-guide.md",
        doc_type=SourceType.MARKDOWN,
    )
    vector_store = SessionScopedVectorStore(
        [
            Document(
                page_content="Session-scoped document.",
                metadata={
                    "doc_id": "doc-session",
                    "source_name": "session-guide.md",
                    "source_uri_or_path": "session-guide.md",
                    "page_or_section": "1",
                },
            )
        ]
    )
    service = AnswerService(
        settings=settings,
        chat_client=FakeChatClient(["Session answer"]),
        vector_store=vector_store,
        chat_history_store=history_store,
    )

    _ = [
        event
        async for event in service.stream_answer(
            session_id=session_id, question="What is in the session file?"
        )
    ]

    assert vector_store.queries[0][2] == RetrievalFilter(doc_ids=["doc-session"])


@pytest.mark.asyncio
async def test_answer_service_intersects_session_scope_with_request_filter() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-files-filter")
    history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-session",
        source_name="session-guide.md",
        source_uri_or_path="session-guide.md",
        doc_type=SourceType.MARKDOWN,
    )
    vector_store = SessionScopedVectorStore([])
    service = AnswerService(
        settings=settings,
        chat_client=FakeChatClient(["No answer"]),
        vector_store=vector_store,
        chat_history_store=history_store,
    )

    _ = [
        event
        async for event in service.stream_answer(
            session_id=session_id,
            question="What is in the session file?",
            metadata_filter=RetrievalFilter(doc_ids=["doc-session", "doc-other"]),
        )
    ]

    assert vector_store.queries[0][2] == RetrievalFilter(doc_ids=["doc-session"])


@pytest.mark.asyncio
async def test_answer_service_ignores_non_indexed_session_files() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-files-pending")
    pending_file = history_store.create_session_file_upload(
        session_id=session_id,
        source_name="pending.txt",
        source_uri_or_path="pending.txt",
    )
    history_store.update_session_file(
        session_id=session_id,
        file_id=pending_file.file_id,
        status=SessionFileStatus.PARSED,
    )

    vector_store = SessionScopedVectorStore([])
    service = AnswerService(
        settings=settings,
        chat_client=FakeChatClient(["No answer"]),
        vector_store=vector_store,
        chat_history_store=history_store,
    )

    events = [
        event
        async for event in service.stream_answer(
            session_id=session_id, question="Any indexed docs?"
        )
    ]

    assert vector_store.queries[0][2] == RetrievalFilter(
        doc_ids=["__session_scope_no_indexed_files__"]
    )
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "rejected"


class FakeIndexManager:
    def __init__(self) -> None:
        self.deleted_doc_ids: list[str] = []
        self.delete_error_step: str | None = None
        self.active_documents: dict[str, str] = {}

    def index_chunks(self, chunks) -> None:
        if chunks:
            first_chunk = chunks[0]
            self.active_documents[first_chunk.doc_id] = str(first_chunk.source_uri_or_path)
        return None

    def delete_document(self, doc_id: str) -> bool:
        self.deleted_doc_ids.append(doc_id)
        self.active_documents.pop(doc_id, None)
        return True

    def delete_document_with_steps(self, doc_id: str):
        if self.delete_error_step:
            raise DocumentDeleteStepError(
                step=self.delete_error_step,
                message=f"[step={self.delete_error_step}] delete failed",
            )
        self.deleted_doc_ids.append(doc_id)
        self.active_documents.pop(doc_id, None)
        return None

    def get_document_summary(self, doc_id: str):
        source_uri_or_path = self.active_documents.get(doc_id)
        if source_uri_or_path is None:
            return None
        return type(
            "DocumentSummary",
            (),
            {"doc_id": doc_id, "source_uri_or_path": source_uri_or_path},
        )()

    def count_active_documents_by_source_uri_or_path(self, source_uri_or_path: str) -> int:
        return sum(1 for value in self.active_documents.values() if value == source_uri_or_path)


def _build_ingestion_service(
    settings: Settings, history_store: ChatHistoryStore
) -> IngestionService:
    return IngestionService(
        settings=settings,
        embeddings=object(),
        vector_store=object(),  # type: ignore[arg-type]
        index_manager=FakeIndexManager(),  # type: ignore[arg-type]
        chat_history_store=history_store,
    )


def test_ingestion_service_session_upload_status_flow(monkeypatch) -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-upload-flow")
    service = _build_ingestion_service(settings, history_store)

    documents = [
        NormalizedDocument(
            doc_id="doc-upload",
            source_type=SourceType.TXT,
            source_name="notes.txt",
            source_uri_or_path="notes.txt",
            page_or_section="1",
            page="1",
            title="Notes",
            section_path=["Notes"],
            doc_type=SourceType.TXT,
            content="hello",
            metadata={},
        )
    ]
    chunks = [
        IngestedChunk(
            chunk_id="chunk-1",
            doc_id="doc-upload",
            source_name="notes.txt",
            source_uri_or_path="notes.txt",
            page_or_section="1",
            page="1",
            title="Notes",
            section_path=["Notes"],
            doc_type=SourceType.TXT,
            content="hello",
            metadata={"chunk_id": "chunk-1", "child_content": "hello"},
        )
    ]
    monkeypatch.setattr(
        service._router,
        "load_file",
        lambda file_path, force_mineru=False, **kwargs: (documents, False),
    )
    monkeypatch.setattr(service._chunking, "chunk_documents", lambda docs: chunks)

    summary = service.ingest_session_upload(
        session_id=session_id, file_name="notes.txt", payload=b"hello"
    )

    assert summary.status is SessionFileStatus.INDEXED
    assert summary.doc_id == "doc-upload"
    assert summary.page_count == 1


def test_ingestion_service_session_upload_marks_failed_on_parse_error(monkeypatch) -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-upload-fail")
    service = _build_ingestion_service(settings, history_store)

    monkeypatch.setattr(
        service._router,
        "load_file",
        lambda file_path, force_mineru=False, **kwargs: (_ for _ in ()).throw(
            RuntimeError("parse failed")
        ),
    )

    summary = service.ingest_session_upload(
        session_id=session_id, file_name="broken.pdf", payload=b"bad"
    )

    assert summary.status is SessionFileStatus.FAILED
    assert summary.error_code == "parse_failed"
    assert summary.error_message is not None


def test_ingestion_service_delete_session_file_keeps_record_when_cleanup_fails(monkeypatch) -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-delete-fail")
    file_summary = history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-delete",
        source_name="notes.txt",
        source_uri_or_path="notes.txt",
        doc_type=SourceType.TXT,
    )
    service = _build_ingestion_service(settings, history_store)
    service._index_manager.delete_error_step = "parent_store_delete"

    with pytest.raises(SessionFileCleanupError):
        service.delete_session_file(session_id=session_id, file_id=file_summary.file_id)

    current = history_store.get_session_file(session_id=session_id, file_id=file_summary.file_id)
    assert current is not None
    assert current.status is SessionFileStatus.RECOVER_PENDING
    assert current.error_code == "delete_cleanup_failed"
    assert current.error_message is not None
    assert "parent_store_delete" in current.error_message


def test_ingestion_service_delete_failed_session_file_directly_removes_record() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-delete-direct-failed")
    uploaded = history_store.create_session_file_upload(
        session_id=session_id,
        source_name="broken.pdf",
        source_uri_or_path="broken.pdf",
    )
    history_store.update_session_file(
        session_id=session_id,
        file_id=uploaded.file_id,
        status=SessionFileStatus.FAILED,
        error_code="parse_failed",
        error_message="parse failed",
    )
    service = _build_ingestion_service(settings, history_store)

    deleted = service.delete_session_file(session_id=session_id, file_id=uploaded.file_id)

    assert deleted is not None
    assert deleted.file_id == uploaded.file_id
    assert history_store.get_session_file(session_id=session_id, file_id=uploaded.file_id) is None
    assert service._index_manager.deleted_doc_ids == []


def test_ingestion_service_recover_delete_is_idempotent_when_missing() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-recover-missing")
    service = _build_ingestion_service(settings, history_store)

    outcome = service.recover_session_file_deletion(session_id=session_id, file_id="missing")

    assert outcome == "already_deleted"


def test_ingestion_service_recover_delete_requires_recover_pending_state() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-recover-state")
    summary = history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-state",
        source_name="notes.txt",
        source_uri_or_path="notes.txt",
        doc_type=SourceType.TXT,
    )
    service = _build_ingestion_service(settings, history_store)

    with pytest.raises(SessionFileRecoveryStateError):
        service.recover_session_file_deletion(session_id=session_id, file_id=summary.file_id)


def test_ingestion_service_recover_delete_successfully_cleans_pending_record() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-recover-success")
    summary = history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-success",
        source_name="notes.txt",
        source_uri_or_path="notes.txt",
        doc_type=SourceType.TXT,
    )
    history_store.update_session_file(
        session_id=session_id,
        file_id=summary.file_id,
        status=SessionFileStatus.RECOVER_PENDING,
        error_code="delete_cleanup_failed",
        error_message="[step=parent_store_delete] delete failed",
    )
    service = _build_ingestion_service(settings, history_store)

    outcome = service.recover_session_file_deletion(session_id=session_id, file_id=summary.file_id)

    assert outcome == "ok"
    assert history_store.get_session_file(session_id=session_id, file_id=summary.file_id) is None


def test_ingestion_service_sanitizes_upload_filenames() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    service = _build_ingestion_service(settings, history_store)

    saved_upload = service.save_upload(r"..\..\secret\plan?.txt", b"hello")
    saved_session_upload = service.save_session_upload(
        "session-safe",
        "../../nested\\unsafe:name.txt",
        b"hello",
    )

    expected_object_dir = (settings.uploads_directory / "objects").resolve(strict=False)
    assert saved_upload.parent == expected_object_dir
    assert saved_upload.name.endswith(".txt")
    assert len(saved_upload.stem) == 64
    assert ".." not in saved_upload.name

    assert saved_session_upload.parent == expected_object_dir
    assert saved_session_upload.name.endswith(".txt")
    assert len(saved_session_upload.stem) == 64
    assert ".." not in saved_session_upload.name


def test_ingestion_service_reuses_object_storage_for_duplicate_payloads() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    service = _build_ingestion_service(settings, history_store)

    first_path = service.save_upload("first.pdf", b"same payload")
    second_path = service.save_upload("second.pdf", b"same payload")
    session_path = service.save_session_upload("session-a", "third.pdf", b"same payload")

    assert first_path == second_path
    assert second_path == session_path
    assert first_path.exists()


def test_ingestion_service_delete_session_file_removes_persisted_artifacts() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-delete-artifacts")
    service = _build_ingestion_service(settings, history_store)

    upload_path = service.save_session_upload(session_id, "notes.txt", b"hello")
    processed_path = settings.processed_directory / f"{upload_path.stem}.normalized.json"
    processed_path.write_text("{}", encoding="utf-8")

    uploaded = history_store.create_session_file_upload(
        session_id=session_id,
        source_name="notes.txt",
        source_uri_or_path=str(upload_path),
    )
    history_store.update_session_file(
        session_id=session_id,
        file_id=uploaded.file_id,
        status=SessionFileStatus.FAILED,
        error_code="parse_failed",
        error_message="parse failed",
    )

    service.delete_session_file(session_id=session_id, file_id=uploaded.file_id)

    assert not upload_path.exists()
    assert not processed_path.exists()
    assert not upload_path.parent.exists()


def test_ingestion_service_delete_session_removes_all_session_artifacts() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-delete-all")
    service = _build_ingestion_service(settings, history_store)

    upload_path = service.save_session_upload(session_id, "notes.txt", b"hello")
    processed_path = settings.processed_directory / f"{upload_path.stem}.normalized.json"
    processed_path.write_text("{}", encoding="utf-8")
    history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-delete-all",
        source_name="notes.txt",
        source_uri_or_path=str(upload_path),
        doc_type=SourceType.TXT,
    )

    service.delete_session(session_id=session_id)

    assert not upload_path.exists()
    assert not processed_path.exists()
    assert not upload_path.parent.exists()
    assert service._index_manager.deleted_doc_ids == ["doc-delete-all"]
    assert history_store.list_session_files(session_id) == []
    assert all(item.session_id != session_id for item in history_store.list_sessions())


def test_ingestion_service_delete_session_file_preserves_shared_doc_and_object_until_last_reference() -> None:
    settings = _ingest_settings()
    history_store = ChatHistoryStore(settings)
    session_id = history_store.ensure_session("session-shared")
    service = _build_ingestion_service(settings, history_store)

    upload_path = service.save_upload("shared.pdf", b"shared payload")
    processed_path = settings.processed_directory / "doc-shared.normalized.json"
    processed_path.write_text("{}", encoding="utf-8")
    first = history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-shared",
        source_name="shared-a.pdf",
        source_uri_or_path=str(upload_path),
        doc_type=SourceType.PDF,
    )
    second = history_store.register_session_file(
        session_id=session_id,
        doc_id="doc-shared",
        source_name="shared-b.pdf",
        source_uri_or_path=str(upload_path),
        doc_type=SourceType.PDF,
    )
    service._index_manager.active_documents["doc-shared"] = str(upload_path)

    service.delete_session_file(session_id=session_id, file_id=first.file_id)

    assert upload_path.exists()
    assert processed_path.exists()
    assert service._index_manager.deleted_doc_ids == []
    assert history_store.get_session_file(session_id=session_id, file_id=second.file_id) is not None

    service.delete_session_file(session_id=session_id, file_id=second.file_id)

    assert not upload_path.exists()
    assert not processed_path.exists()
    assert service._index_manager.deleted_doc_ids == ["doc-shared"]
