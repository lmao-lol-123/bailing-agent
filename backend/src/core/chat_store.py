from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.src.core.config import Settings
from backend.src.core.models import ChatMessage, ChatRole, ChatSessionSummary, Citation


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
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id ON chat_messages(session_id, id)"
            )

    @staticmethod
    def _ensure_session_title_column(connection: sqlite3.Connection) -> None:
        columns = [row["name"] for row in connection.execute("PRAGMA table_info(chat_sessions)").fetchall()]
        if "title" not in columns:
            connection.execute("ALTER TABLE chat_sessions ADD COLUMN title TEXT")

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
    ) -> None:
        timestamp = _utc_now_iso()
        normalized_citations = [Citation.model_validate(citation) for citation in (citations or [])]
        serialized_citations = json.dumps(
            [citation.model_dump() for citation in normalized_citations],
            ensure_ascii=False,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, grounded, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role.value,
                    content,
                    None if grounded is None else int(grounded),
                    serialized_citations,
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

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    def list_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        if limit is None:
            query = (
                "SELECT session_id, role, content, grounded, citations_json, created_at "
                "FROM chat_messages WHERE session_id = ? ORDER BY id ASC"
            )
            parameters: tuple[object, ...] = (session_id,)
        else:
            query = (
                "SELECT session_id, role, content, grounded, citations_json, created_at FROM ("
                "SELECT session_id, role, content, grounded, citations_json, created_at, id "
                "FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC"
            )
            parameters = (session_id, limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        messages: list[ChatMessage] = []
        for row in rows:
            citations = [Citation.model_validate(item) for item in json.loads(row["citations_json"])]
            created_at = datetime.fromisoformat(row["created_at"])
            grounded_value = row["grounded"]
            messages.append(
                ChatMessage(
                    session_id=row["session_id"],
                    role=ChatRole(row["role"]),
                    content=row["content"],
                    grounded=None if grounded_value is None else bool(grounded_value),
                    citations=citations,
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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_session_title(title: str | None) -> str:
    normalized_title = (title or "").strip()
    if not normalized_title:
        return "新对话"
    return normalized_title[:32]


