from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from backend.scripts.cleanup_data_storage import cleanup_data_storage
from backend.src.core.config import Settings
from backend.src.core.models import (
    DocumentSummary,
    SessionFileStatus,
    SessionFileSummary,
    SourceType,
)


class FakeChatHistoryStore:
    def __init__(self, session_files: list[SessionFileSummary]) -> None:
        self._session_files = session_files

    def list_all_session_files(self) -> list[SessionFileSummary]:
        return list(self._session_files)

    def update_session_file(self, *, session_id: str, file_id: str, **kwargs):
        for index, session_file in enumerate(self._session_files):
            if session_file.session_id != session_id or session_file.file_id != file_id:
                continue
            self._session_files[index] = session_file.model_copy(update=kwargs)
            return self._session_files[index]
        return None


class FakeIndexManager:
    def __init__(self, documents: list[DocumentSummary]) -> None:
        self._documents = documents
        self.rebuild_count = 0

    def list_documents(self) -> list[DocumentSummary]:
        return list(self._documents)

    def rewrite_document_source_uri(self, *, doc_id: str, source_uri_or_path: str) -> bool:
        for index, document in enumerate(self._documents):
            if document.doc_id != doc_id:
                continue
            self._documents[index] = document.model_copy(
                update={"source_uri_or_path": source_uri_or_path}
            )
            return True
        return False

    def rebuild(self) -> int:
        self.rebuild_count += 1
        return len(self._documents)


def _case_settings() -> Settings:
    case_dir = Path("backend/.pytest-tmp") / f"cleanup-script-{uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
        chat_history_db_path=case_dir / "chat.sqlite3",
    )
    settings.ensure_directories()
    return settings


def test_cleanup_data_storage_dry_run_reports_without_mutation() -> None:
    settings = _case_settings()
    legacy_upload = settings.uploads_directory / "12345678-1234-1234-1234-123456789abc-guide.pdf"
    duplicate_upload = settings.uploads_directory / "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa-guide.pdf"
    legacy_upload.write_bytes(b"same payload")
    duplicate_upload.write_bytes(b"same payload")
    legacy_snapshot = settings.processed_directory / f"{legacy_upload.stem}.normalized.json"
    legacy_snapshot.write_text("{}", encoding="utf-8")
    stale_dir = settings.processed_directory / "legacy-dir"
    stale_dir.mkdir()
    session_file = SessionFileSummary(
        file_id="file-1",
        session_id="session-1",
        doc_id="doc-legacy",
        source_name="guide.pdf",
        source_uri_or_path=str(legacy_upload),
        doc_type=SourceType.PDF,
        status=SessionFileStatus.INDEXED,
    )
    index_manager = FakeIndexManager(
        [
            DocumentSummary(
                doc_id="doc-legacy",
                source_name="guide.pdf",
                source_type=SourceType.PDF,
                source_uri_or_path=str(legacy_upload),
                chunk_count=1,
                page_or_sections=[],
            )
        ]
    )
    container = SimpleNamespace(
        settings=settings,
        chat_history_store=FakeChatHistoryStore([session_file]),
        index_manager=index_manager,
    )

    report = cleanup_data_storage(container=container, apply=False)

    assert report["uploads_scanned"] == 2
    assert report["session_files_updated"] == 1
    assert report["documents_updated"] == 1
    assert legacy_upload.exists()
    assert duplicate_upload.exists()
    assert legacy_snapshot.exists()
    assert stale_dir.exists()
    assert index_manager.rebuild_count == 0


def test_cleanup_data_storage_apply_migrates_and_cleans_legacy_artifacts() -> None:
    settings = _case_settings()
    legacy_upload = settings.uploads_directory / "12345678-1234-1234-1234-123456789abc-guide.pdf"
    duplicate_upload = settings.uploads_directory / "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa-guide.pdf"
    legacy_upload.write_bytes(b"same payload")
    duplicate_upload.write_bytes(b"same payload")
    legacy_snapshot = settings.processed_directory / f"{legacy_upload.stem}.normalized.json"
    legacy_snapshot.write_text("{}", encoding="utf-8")
    stale_snapshot = settings.processed_directory / "orphan.normalized.json"
    stale_snapshot.write_text("{}", encoding="utf-8")
    stale_dir = settings.processed_directory / "legacy-dir"
    stale_dir.mkdir()
    session_file = SessionFileSummary(
        file_id="file-1",
        session_id="session-1",
        doc_id="doc-legacy",
        source_name="guide.pdf",
        source_uri_or_path=str(legacy_upload),
        doc_type=SourceType.PDF,
        status=SessionFileStatus.INDEXED,
    )
    chat_history_store = FakeChatHistoryStore([session_file])
    index_manager = FakeIndexManager(
        [
            DocumentSummary(
                doc_id="doc-legacy",
                source_name="guide.pdf",
                source_type=SourceType.PDF,
                source_uri_or_path=str(legacy_upload),
                chunk_count=1,
                page_or_sections=[],
            )
        ]
    )
    container = SimpleNamespace(
        settings=settings,
        chat_history_store=chat_history_store,
        index_manager=index_manager,
    )

    report = cleanup_data_storage(container=container, apply=True)

    object_dir = settings.uploads_directory / "objects"
    object_files = [path.resolve(strict=False) for path in object_dir.glob("*.pdf")]
    assert len(object_files) == 1
    assert not legacy_upload.exists()
    assert not duplicate_upload.exists()
    assert chat_history_store.list_all_session_files()[0].source_uri_or_path == str(object_files[0])
    assert index_manager.list_documents()[0].source_uri_or_path == str(object_files[0])
    assert (settings.processed_directory / "doc-legacy.normalized.json").exists()
    assert not legacy_snapshot.exists()
    assert not stale_snapshot.exists()
    assert not stale_dir.exists()
    assert index_manager.rebuild_count == 1
    assert report["deleted_legacy_uploads"] == 2
