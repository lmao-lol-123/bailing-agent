from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.src.core.config import Settings
from backend.src.core.models import (
    ChatMessage,
    ChatRole,
    ChatSessionSummary,
    Citation,
    SessionFileStatus,
    SessionFileSummary,
    SourceType,
)


class ChatHistoryStore:
    def __init__(self, settings: Settings) -> None:
        self._db_path = Path(settings.chat_history_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_session_title_column(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    grounded INTEGER,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
                )
                """
            )
            self._ensure_message_columns(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_files (
                    file_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    doc_id TEXT,
                    source_name TEXT NOT NULL,
                    source_uri_or_path TEXT NOT NULL,
                    doc_type TEXT,
                    page_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
                )
                """
            )
            self._ensure_session_file_columns(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id ON chat_messages(session_id, id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_files_session_id_created_at ON session_files(session_id, created_at)"
            )

    @staticmethod
    def _ensure_session_title_column(connection: sqlite3.Connection) -> None:
        columns = [
            row["name"] for row in connection.execute("PRAGMA table_info(chat_sessions)").fetchall()
        ]
        if "title" not in columns:
            connection.execute("ALTER TABLE chat_sessions ADD COLUMN title TEXT")

    @staticmethod
    def _ensure_message_columns(connection: sqlite3.Connection) -> None:
        columns = [
            row["name"] for row in connection.execute("PRAGMA table_info(chat_messages)").fetchall()
        ]
        if "retrieval_query" not in columns:
            connection.execute("ALTER TABLE chat_messages ADD COLUMN retrieval_query TEXT")
        if "metadata_filter_json" not in columns:
            connection.execute(
                "ALTER TABLE chat_messages ADD COLUMN metadata_filter_json TEXT NOT NULL DEFAULT 'null'"
            )
        if "debug_trace_json" not in columns:
            connection.execute(
                "ALTER TABLE chat_messages ADD COLUMN debug_trace_json TEXT NOT NULL DEFAULT 'null'"
            )

    @staticmethod
    def _ensure_session_file_columns(connection: sqlite3.Connection) -> None:
        columns = [
            row["name"] for row in connection.execute("PRAGMA table_info(session_files)").fetchall()
        ]
        if "page_count" not in columns:
            connection.execute("ALTER TABLE session_files ADD COLUMN page_count INTEGER")
        if "status" not in columns:
            connection.execute(
                "ALTER TABLE session_files ADD COLUMN status TEXT NOT NULL DEFAULT 'uploaded'"
            )
        if "error_code" not in columns:
            connection.execute("ALTER TABLE session_files ADD COLUMN error_code TEXT")
        if "error_message" not in columns:
            connection.execute("ALTER TABLE session_files ADD COLUMN error_message TEXT")
        if "updated_at" not in columns:
            connection.execute(
                "ALTER TABLE session_files ADD COLUMN updated_at TEXT NOT NULL DEFAULT '1970-01-01T00:00:00+00:00'"
            )

    def ensure_session(self, session_id: str | None) -> str:
        resolved_session_id = session_id or uuid4().hex
        timestamp = _utc_now_iso()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (session_id, title, created_at, updated_at)
                VALUES (?, NULL, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (resolved_session_id, timestamp, timestamp),
            )

        return resolved_session_id

    def append_message(
        self,
        session_id: str,
        role: ChatRole,
        content: str,
        grounded: bool | None = None,
        citations: list[Citation | dict[str, object]] | None = None,
        retrieval_query: str | None = None,
        metadata_filter: dict[str, object] | None = None,
        debug_trace: dict[str, object] | None = None,
    ) -> None:
        timestamp = _utc_now_iso()
        normalized_citations = [Citation.model_validate(citation) for citation in (citations or [])]
        serialized_citations = json.dumps(
            [citation.model_dump(mode="json") for citation in normalized_citations],
            ensure_ascii=False,
        )
        serialized_metadata_filter = (
            json.dumps(metadata_filter, ensure_ascii=False)
            if metadata_filter is not None
            else "null"
        )
        serialized_debug_trace = (
            json.dumps(debug_trace, ensure_ascii=False) if debug_trace is not None else "null"
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    session_id, role, content, grounded, citations_json, retrieval_query, metadata_filter_json, debug_trace_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role.value,
                    content,
                    None if grounded is None else int(grounded),
                    serialized_citations,
                    retrieval_query,
                    serialized_metadata_filter,
                    serialized_debug_trace,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )

    def rename_session(self, session_id: str, title: str) -> None:
        normalized_title = _build_session_title(title)
        timestamp = _utc_now_iso()

        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (normalized_title, timestamp, session_id),
            )

    def delete_session(self, session_id: str) -> list[SessionFileSummary]:
        session_files = self.list_session_files(session_id)
        with self._connect() as connection:
            connection.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        return session_files

    def list_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        if limit is None:
            query = (
                "SELECT session_id, role, content, grounded, citations_json, retrieval_query, debug_trace_json, created_at "
                "FROM chat_messages WHERE session_id = ? ORDER BY id ASC"
            )
            parameters: tuple[object, ...] = (session_id,)
        else:
            query = (
                "SELECT session_id, role, content, grounded, citations_json, retrieval_query, debug_trace_json, created_at FROM ("
                "SELECT session_id, role, content, grounded, citations_json, retrieval_query, debug_trace_json, created_at, id "
                "FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC"
            )
            parameters = (session_id, limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        messages: list[ChatMessage] = []
        for row in rows:
            citations = [
                Citation.model_validate(item) for item in json.loads(row["citations_json"])
            ]
            created_at = datetime.fromisoformat(row["created_at"])
            grounded_value = row["grounded"]
            debug_trace = json.loads(row["debug_trace_json"]) if row["debug_trace_json"] else None
            messages.append(
                ChatMessage(
                    session_id=row["session_id"],
                    role=ChatRole(row["role"]),
                    content=row["content"],
                    grounded=None if grounded_value is None else bool(grounded_value),
                    citations=citations,
                    retrieval_query=row["retrieval_query"],
                    debug_trace=debug_trace if isinstance(debug_trace, dict) else None,
                    created_at=created_at,
                )
            )
        return messages

    def list_sessions(self) -> list[ChatSessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    session.session_id,
                    session.title,
                    session.updated_at,
                    session.rowid AS session_row_id,
                    COUNT(message.id) AS message_count,
                    (
                        SELECT first_user_message.content
                        FROM chat_messages AS first_user_message
                        WHERE first_user_message.session_id = session.session_id
                          AND first_user_message.role = ?
                        ORDER BY first_user_message.id ASC
                        LIMIT 1
                    ) AS fallback_title
                FROM chat_sessions AS session
                LEFT JOIN chat_messages AS message
                    ON message.session_id = session.session_id
                GROUP BY session.session_id, session.title, session.updated_at, session.rowid
                ORDER BY session.updated_at DESC, session_row_id DESC
                """,
                (ChatRole.USER.value,),
            ).fetchall()

        return [
            ChatSessionSummary(
                session_id=row["session_id"],
                title=_build_session_title(row["title"] or row["fallback_title"]),
                message_count=int(row["message_count"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def list_query_traces(
        self, session_id: str, limit: int | None = None
    ) -> list[dict[str, object]]:
        if limit is None:
            query = (
                "SELECT debug_trace_json FROM chat_messages "
                "WHERE session_id = ? AND role = ? AND debug_trace_json IS NOT NULL AND debug_trace_json != 'null' "
                "ORDER BY id ASC"
            )
            parameters: tuple[object, ...] = (session_id, ChatRole.ASSISTANT.value)
        else:
            query = (
                "SELECT debug_trace_json FROM ("
                "SELECT debug_trace_json, id FROM chat_messages "
                "WHERE session_id = ? AND role = ? AND debug_trace_json IS NOT NULL AND debug_trace_json != 'null' "
                "ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC"
            )
            parameters = (session_id, ChatRole.ASSISTANT.value, limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        traces: list[dict[str, object]] = []
        for row in rows:
            try:
                trace_value = json.loads(str(row["debug_trace_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(trace_value, dict):
                traces.append(trace_value)
        return traces

    def register_session_file(
        self,
        *,
        session_id: str,
        doc_id: str,
        source_name: str,
        source_uri_or_path: str,
        doc_type: SourceType,
        file_id: str | None = None,
    ) -> SessionFileSummary:
        resolved_file_id = file_id or uuid4().hex
        timestamp = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_files (
                    file_id, session_id, doc_id, source_name, source_uri_or_path, doc_type,
                    page_count, status, error_code, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?)
                """,
                (
                    resolved_file_id,
                    session_id,
                    doc_id,
                    source_name,
                    source_uri_or_path,
                    doc_type.value,
                    SessionFileStatus.INDEXED.value,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
        return self.get_session_file(
            session_id=session_id, file_id=resolved_file_id
        ) or SessionFileSummary(
            file_id=resolved_file_id,
            session_id=session_id,
            doc_id=doc_id,
            source_name=source_name,
            source_uri_or_path=source_uri_or_path,
            doc_type=doc_type,
            status=SessionFileStatus.INDEXED,
            created_at=datetime.fromisoformat(timestamp),
            updated_at=datetime.fromisoformat(timestamp),
        )

    def create_session_file_upload(
        self,
        *,
        session_id: str,
        source_name: str,
        source_uri_or_path: str,
        file_id: str | None = None,
    ) -> SessionFileSummary:
        resolved_file_id = file_id or uuid4().hex
        timestamp = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_files (
                    file_id, session_id, doc_id, source_name, source_uri_or_path, doc_type,
                    page_count, status, error_code, error_message, created_at, updated_at
                )
                VALUES (?, ?, NULL, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?)
                """,
                (
                    resolved_file_id,
                    session_id,
                    source_name,
                    source_uri_or_path,
                    SessionFileStatus.UPLOADED.value,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
        return self.get_session_file(
            session_id=session_id, file_id=resolved_file_id
        ) or SessionFileSummary(
            file_id=resolved_file_id,
            session_id=session_id,
            source_name=source_name,
            source_uri_or_path=source_uri_or_path,
            status=SessionFileStatus.UPLOADED,
            created_at=datetime.fromisoformat(timestamp),
            updated_at=datetime.fromisoformat(timestamp),
        )

    def update_session_file(
        self,
        *,
        session_id: str,
        file_id: str,
        status: SessionFileStatus,
        doc_id: str | None = None,
        doc_type: SourceType | None = None,
        page_count: int | None = None,
        source_name: str | None = None,
        source_uri_or_path: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> SessionFileSummary | None:
        existing = self.get_session_file(session_id=session_id, file_id=file_id)
        if existing is None:
            return None

        resolved_doc_id = existing.doc_id if doc_id is None else doc_id
        resolved_doc_type = existing.doc_type if doc_type is None else doc_type
        resolved_page_count = existing.page_count if page_count is None else page_count
        resolved_source_name = existing.source_name if source_name is None else source_name
        resolved_source_uri = (
            existing.source_uri_or_path if source_uri_or_path is None else source_uri_or_path
        )
        resolved_error_code = error_code
        resolved_error_message = error_message
        if status not in {SessionFileStatus.FAILED, SessionFileStatus.RECOVER_PENDING}:
            resolved_error_code = None
            resolved_error_message = None
        timestamp = _utc_now_iso()

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE session_files
                SET doc_id = ?, source_name = ?, source_uri_or_path = ?, doc_type = ?,
                    page_count = ?, status = ?, error_code = ?, error_message = ?, updated_at = ?
                WHERE session_id = ? AND file_id = ?
                """,
                (
                    resolved_doc_id,
                    resolved_source_name,
                    resolved_source_uri,
                    resolved_doc_type.value if resolved_doc_type else None,
                    resolved_page_count,
                    status.value,
                    resolved_error_code,
                    resolved_error_message,
                    timestamp,
                    session_id,
                    file_id,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
        return self.get_session_file(session_id=session_id, file_id=file_id)

    def get_session_file(self, *, session_id: str, file_id: str) -> SessionFileSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_id, session_id, doc_id, source_name, source_uri_or_path, doc_type,
                       page_count, status, error_code, error_message, created_at, updated_at
                FROM session_files
                WHERE session_id = ? AND file_id = ?
                """,
                (session_id, file_id),
            ).fetchone()
        if row is None:
            return None
        return self._session_file_from_row(row)

    def list_session_files(self, session_id: str) -> list[SessionFileSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT file_id, session_id, doc_id, source_name, source_uri_or_path, doc_type,
                       page_count, status, error_code, error_message, created_at, updated_at
                FROM session_files
                WHERE session_id = ?
                ORDER BY created_at ASC, file_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._session_file_from_row(row) for row in rows]

    def list_all_session_files(self) -> list[SessionFileSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT file_id, session_id, doc_id, source_name, source_uri_or_path, doc_type,
                       page_count, status, error_code, error_message, created_at, updated_at
                FROM session_files
                ORDER BY created_at ASC, file_id ASC
                """
            ).fetchall()
        return [self._session_file_from_row(row) for row in rows]

    def delete_session_file(self, session_id: str, file_id: str) -> SessionFileSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_id, session_id, doc_id, source_name, source_uri_or_path, doc_type,
                       page_count, status, error_code, error_message, created_at, updated_at
                FROM session_files
                WHERE session_id = ? AND file_id = ?
                """,
                (session_id, file_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "DELETE FROM session_files WHERE session_id = ? AND file_id = ?",
                (session_id, file_id),
            )
        return self._session_file_from_row(row)

    @staticmethod
    def _session_file_from_row(row: sqlite3.Row) -> SessionFileSummary:
        doc_type_raw = row["doc_type"]
        status_raw = row["status"] or SessionFileStatus.UPLOADED.value
        return SessionFileSummary(
            file_id=row["file_id"],
            session_id=row["session_id"],
            doc_id=str(row["doc_id"]) if row["doc_id"] is not None else None,
            source_name=row["source_name"],
            source_uri_or_path=row["source_uri_or_path"],
            doc_type=SourceType(doc_type_raw) if doc_type_raw else None,
            page_count=int(row["page_count"]) if row["page_count"] is not None else None,
            status=SessionFileStatus(str(status_raw)),
            error_code=str(row["error_code"]) if row["error_code"] is not None else None,
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_session_title(title: str | None) -> str:
    normalized_title = (title or "").strip()
    if not normalized_title:
        return "新对话"
    return normalized_title[:32]
