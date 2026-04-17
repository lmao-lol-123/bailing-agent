from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import ChatRole, Citation, SessionFileStatus, SourceType


def _chat_db_path() -> Path:
    base = Path("backend/.pytest-tmp") / "chat_history_tests"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{uuid4().hex}.sqlite3"


def test_chat_history_store_persists_messages() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)

    session_id = store.ensure_session(None)
    store.append_message(
        session_id=session_id, role=ChatRole.USER, content="How do I build the index?"
    )
    store.append_message(
        session_id=session_id,
        role=ChatRole.ASSISTANT,
        content="Run python -m scripts.build_index.",
        grounded=True,
        retrieval_query="How do I build the index?",
        debug_trace={"route_names": ["general"], "result_count": 1},
        citations=[
            Citation(
                index=1,
                doc_id="doc-1",
                source_name="README.md",
                source_uri_or_path="README.md",
                page_or_section="CLI",
                snippet="python -m scripts.build_index",
            )
        ],
    )

    messages = store.list_messages(session_id)

    assert len(messages) == 2
    assert messages[0].role is ChatRole.USER
    assert messages[0].content == "How do I build the index?"
    assert messages[1].role is ChatRole.ASSISTANT
    assert messages[1].grounded is True
    assert messages[1].citations[0].source_name == "README.md"
    assert messages[1].retrieval_query == "How do I build the index?"
    assert messages[1].debug_trace == {"route_names": ["general"], "result_count": 1}


def test_chat_history_store_lists_sessions_by_recent_update() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)

    first_session_id = store.ensure_session(None)
    store.append_message(first_session_id, ChatRole.USER, "How do I ingest a PDF?")
    second_session_id = store.ensure_session(None)
    store.append_message(second_session_id, ChatRole.USER, "Explain the FAISS retrieval flow")

    sessions = store.list_sessions()

    assert [session.session_id for session in sessions] == [second_session_id, first_session_id]
    assert sessions[0].title == "Explain the FAISS retrieval flow"
    assert sessions[0].message_count == 1
    assert sessions[0].updated_at is not None


def test_chat_history_store_renames_and_deletes_session() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)

    session_id = store.ensure_session(None)
    store.append_message(session_id, ChatRole.USER, "Original title")

    store.rename_session(session_id, "Renamed session")

    sessions = store.list_sessions()
    assert sessions[0].title == "Renamed session"

    store.delete_session(session_id)

    assert store.list_messages(session_id) == []
    assert store.list_sessions() == []


def test_chat_history_store_registers_and_deletes_session_files() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)

    session_id = store.ensure_session(None)
    file_summary = store.register_session_file(
        session_id=session_id,
        doc_id="doc-session",
        source_name="session-guide.md",
        source_uri_or_path="session-guide.md",
        doc_type=SourceType.MARKDOWN,
    )

    listed_files = store.list_session_files(session_id)
    deleted_file = store.delete_session_file(session_id, file_summary.file_id)

    assert len(listed_files) == 1
    assert listed_files[0].doc_id == "doc-session"
    assert listed_files[0].status is SessionFileStatus.INDEXED
    assert deleted_file is not None
    assert deleted_file.file_id == file_summary.file_id
    assert store.list_session_files(session_id) == []


def test_chat_history_store_updates_session_file_status() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)
    session_id = store.ensure_session(None)

    file_summary = store.create_session_file_upload(
        session_id=session_id,
        source_name="notes.txt",
        source_uri_or_path="notes.txt",
    )
    parsed_summary = store.update_session_file(
        session_id=session_id,
        file_id=file_summary.file_id,
        status=SessionFileStatus.PARSED,
        page_count=3,
    )
    indexed_summary = store.update_session_file(
        session_id=session_id,
        file_id=file_summary.file_id,
        status=SessionFileStatus.INDEXED,
        doc_id="doc-2",
        doc_type=SourceType.TXT,
    )
    recover_pending_summary = store.update_session_file(
        session_id=session_id,
        file_id=file_summary.file_id,
        status=SessionFileStatus.RECOVER_PENDING,
        error_code="delete_cleanup_failed",
        error_message="[step=parent_store_delete] failed",
    )

    assert parsed_summary is not None
    assert parsed_summary.status is SessionFileStatus.PARSED
    assert parsed_summary.page_count == 3
    assert indexed_summary is not None
    assert indexed_summary.status is SessionFileStatus.INDEXED
    assert indexed_summary.doc_id == "doc-2"
    assert recover_pending_summary is not None
    assert recover_pending_summary.status is SessionFileStatus.RECOVER_PENDING
    assert recover_pending_summary.error_code == "delete_cleanup_failed"


def test_chat_history_store_lists_query_traces() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)
    session_id = store.ensure_session(None)

    store.append_message(session_id=session_id, role=ChatRole.USER, content="q")
    store.append_message(
        session_id=session_id,
        role=ChatRole.ASSISTANT,
        content="a",
        debug_trace={"request": {"route_name": "general"}, "retrieval": {}, "generation": {}},
    )

    traces = store.list_query_traces(session_id)

    assert len(traces) == 1
    assert traces[0]["request"]["route_name"] == "general"
