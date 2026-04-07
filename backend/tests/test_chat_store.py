from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import ChatRole, Citation


def _chat_db_path() -> Path:
    base = Path("backend/.pytest-tmp") / "chat_history_tests"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{uuid4().hex}.sqlite3"


def test_chat_history_store_persists_messages() -> None:
    settings = Settings(chat_history_db_path=_chat_db_path())
    store = ChatHistoryStore(settings)

    session_id = store.ensure_session(None)
    store.append_message(session_id=session_id, role=ChatRole.USER, content="How do I build the index?")
    store.append_message(
        session_id=session_id,
        role=ChatRole.ASSISTANT,
        content="Run python -m scripts.build_index.",
        grounded=True,
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




