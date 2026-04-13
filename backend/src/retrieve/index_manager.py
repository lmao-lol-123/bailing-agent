from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import DocumentSummary, IngestedChunk, SourceType
from backend.src.ingest.parent_store import JsonParentStore

if TYPE_CHECKING:
    from backend.src.retrieve.store import VectorStoreService

_INDEX_SCHEMA_VERSION = 1
_CHUNK_SCHEMA_VERSION = 1
_PARENT_STORE_VERSION = 1


@dataclass(frozen=True)
class IndexWriteResult:
    doc_id: str
    chunk_count: int
    added_chunk_ids: list[str] = field(default_factory=list)
    removed_chunk_ids: list[str] = field(default_factory=list)
    skipped: bool = False
    created: bool = False
    updated: bool = False


@dataclass(frozen=True)
class ActiveChunkRecord:
    chunk_id: str
    doc_id: str
    retrieval_text: str
    child_content: str
    metadata: dict[str, Any]
    source_block_type: str | None = None
    page: str | None = None
    section_path: list[str] = field(default_factory=list)

    def to_document(self) -> Document:
        return Document(page_content=self.retrieval_text, metadata=dict(self.metadata), id=self.chunk_id)


class IndexManager:
    def __init__(self, settings: Settings, embeddings: object, vector_store: VectorStoreService) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._parent_store = JsonParentStore(settings.processed_directory)
        self._index_root = settings.index_state_directory / settings.index_name
        self._manifest_path = self._index_root / "manifest.json"
        self._db_path = self._index_root / "index.sqlite3"
        self._index_root.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._ensure_manifest_exists()

    def index_chunks(self, chunks: list[IngestedChunk]) -> IndexWriteResult:
        if not chunks:
            return IndexWriteResult(doc_id="", chunk_count=0, skipped=True)

        self._assert_manifest_compatible()
        doc_id = chunks[0].doc_id
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError("IndexManager only supports one doc_id per write call")

        fingerprint = self._build_content_fingerprint(chunks)
        now = self._utcnow()
        existing_doc = self._fetch_active_document(doc_id)
        if existing_doc and str(existing_doc["content_fingerprint"]) == fingerprint:
            self._record_ingest_run(doc_id=doc_id, run_type="noop", chunk_count=len(chunks), status="skipped")
            return IndexWriteResult(doc_id=doc_id, chunk_count=int(existing_doc["chunk_count"]), skipped=True)

        existing_chunk_ids = self.get_active_chunk_ids(doc_id)
        documents = self._chunks_to_documents(chunks)
        if existing_chunk_ids:
            self._vector_store.delete_ids(existing_chunk_ids)
        self._vector_store.add_documents(documents=documents, ids=[chunk.chunk_id for chunk in chunks])

        parent_rows = self._collect_parent_rows(chunks)
        ingest_id = str(uuid4())
        with self._connect() as connection:
            existing_parent_ids: list[str] = []
            if existing_chunk_ids:
                existing_parent_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT parent_chunk_id FROM index_chunks WHERE doc_id = ? AND parent_chunk_id IS NOT NULL AND chunk_status='active'",
                        (doc_id,),
                    ).fetchall()
                    if row[0]
                ]
                placeholders = ",".join("?" for _ in existing_chunk_ids)
                connection.execute(
                    f"UPDATE index_chunks SET chunk_status='deleted', updated_at=?, deleted_at=? WHERE chunk_id IN ({placeholders})",
                    (now, now, *existing_chunk_ids),
                )
                if existing_parent_ids:
                    parent_placeholders = ",".join("?" for _ in existing_parent_ids)
                    connection.execute(
                        f"UPDATE index_parents SET parent_status='deleted', updated_at=?, deleted_at=? WHERE parent_chunk_id IN ({parent_placeholders})",
                        (now, now, *existing_parent_ids),
                    )

            connection.execute(
                """
                INSERT INTO index_documents (
                    doc_id, source_name, source_uri_or_path, source_type, doc_type, content_fingerprint,
                    latest_ingest_id, chunk_count, doc_status, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_name=excluded.source_name,
                    source_uri_or_path=excluded.source_uri_or_path,
                    source_type=excluded.source_type,
                    doc_type=excluded.doc_type,
                    content_fingerprint=excluded.content_fingerprint,
                    latest_ingest_id=excluded.latest_ingest_id,
                    chunk_count=excluded.chunk_count,
                    doc_status='active',
                    updated_at=excluded.updated_at,
                    deleted_at=NULL
                """,
                (
                    doc_id,
                    chunks[0].source_name,
                    chunks[0].source_uri_or_path,
                    chunks[0].doc_type.value if chunks[0].doc_type else chunks[0].metadata.get("source_type") or "txt",
                    chunks[0].doc_type.value if chunks[0].doc_type else chunks[0].metadata.get("doc_type") or "txt",
                    fingerprint,
                    ingest_id,
                    len(chunks),
                    now,
                    now,
                ),
            )

            for chunk in chunks:
                metadata_json = json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True)
                section_path_json = json.dumps(chunk.section_path, ensure_ascii=False)
                child_content = str(chunk.metadata.get("child_content") or "")
                connection.execute(
                    """
                    INSERT INTO index_chunks (
                        chunk_id, doc_id, parent_chunk_id, parent_store_ref, source_block_id, source_block_type,
                        page, page_or_section, title, section_path_json, retrieval_text, child_content,
                        metadata_json, retrieval_text_hash, child_content_hash, chunk_status,
                        created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        doc_id=excluded.doc_id,
                        parent_chunk_id=excluded.parent_chunk_id,
                        parent_store_ref=excluded.parent_store_ref,
                        source_block_id=excluded.source_block_id,
                        source_block_type=excluded.source_block_type,
                        page=excluded.page,
                        page_or_section=excluded.page_or_section,
                        title=excluded.title,
                        section_path_json=excluded.section_path_json,
                        retrieval_text=excluded.retrieval_text,
                        child_content=excluded.child_content,
                        metadata_json=excluded.metadata_json,
                        retrieval_text_hash=excluded.retrieval_text_hash,
                        child_content_hash=excluded.child_content_hash,
                        chunk_status='active',
                        updated_at=excluded.updated_at,
                        deleted_at=NULL
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.metadata.get("parent_chunk_id"),
                        chunk.metadata.get("parent_store_ref"),
                        chunk.metadata.get("source_block_id"),
                        chunk.metadata.get("source_block_type"),
                        chunk.page,
                        chunk.page_or_section,
                        chunk.title,
                        section_path_json,
                        chunk.content,
                        child_content,
                        metadata_json,
                        hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                        hashlib.sha256(child_content.encode("utf-8")).hexdigest() if child_content else "",
                        now,
                        now,
                    ),
                )

            for parent_row in parent_rows:
                connection.execute(
                    """
                    INSERT INTO index_parents (
                        parent_chunk_id, doc_id, parent_store_ref, parent_content_hash, source_block_ids_json,
                        parent_status, created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                    ON CONFLICT(parent_chunk_id) DO UPDATE SET
                        doc_id=excluded.doc_id,
                        parent_store_ref=excluded.parent_store_ref,
                        parent_content_hash=excluded.parent_content_hash,
                        source_block_ids_json=excluded.source_block_ids_json,
                        parent_status='active',
                        updated_at=excluded.updated_at,
                        deleted_at=NULL
                    """,
                    (
                        parent_row["parent_chunk_id"],
                        parent_row["doc_id"],
                        parent_row["parent_store_ref"],
                        parent_row["parent_content_hash"],
                        parent_row["source_block_ids_json"],
                        now,
                        now,
                    ),
                )

            connection.execute(
                "INSERT INTO ingest_runs (ingest_id, doc_id, run_type, chunk_count, started_at, completed_at, status, schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ingest_id, doc_id, "update" if existing_doc else "append", len(chunks), now, now, "completed", _INDEX_SCHEMA_VERSION),
            )

        self._touch_manifest()
        return IndexWriteResult(
            doc_id=doc_id,
            chunk_count=len(chunks),
            added_chunk_ids=[chunk.chunk_id for chunk in chunks],
            removed_chunk_ids=existing_chunk_ids,
            created=existing_doc is None,
            updated=existing_doc is not None,
        )

    def delete_document(self, doc_id: str) -> bool:
        chunk_ids = self.get_active_chunk_ids(doc_id)
        if not chunk_ids:
            return False

        now = self._utcnow()
        self._vector_store.delete_ids(chunk_ids)
        with self._connect() as connection:
            connection.execute(
                "UPDATE index_documents SET doc_status='deleted', updated_at=?, deleted_at=? WHERE doc_id=?",
                (now, now, doc_id),
            )
            connection.execute(
                "UPDATE index_chunks SET chunk_status='deleted', updated_at=?, deleted_at=? WHERE doc_id=? AND chunk_status='active'",
                (now, now, doc_id),
            )
            connection.execute(
                "UPDATE index_parents SET parent_status='deleted', updated_at=?, deleted_at=? WHERE doc_id=? AND parent_status='active'",
                (now, now, doc_id),
            )
            connection.execute(
                "INSERT INTO ingest_runs (ingest_id, doc_id, run_type, chunk_count, started_at, completed_at, status, schema_version) VALUES (?, ?, 'delete', ?, ?, ?, 'completed', ?)",
                (str(uuid4()), doc_id, len(chunk_ids), now, now, _INDEX_SCHEMA_VERSION),
            )
        self._parent_store.delete_records(doc_id)
        self._touch_manifest()
        return True

    def rebuild(self) -> int:
        manifest = self.get_manifest()
        documents = self._load_active_documents_for_rebuild()
        self._vector_store.rebuild_documents(documents=documents, ids=[str(getattr(document, "id", "")) for document in documents])
        manifest["schema_version"] = _INDEX_SCHEMA_VERSION
        manifest["chunk_schema_version"] = _CHUNK_SCHEMA_VERSION
        manifest["parent_store_version"] = _PARENT_STORE_VERSION
        manifest["embedding_model"] = self._settings.sentence_transformer_model
        manifest["embedding_dimension"] = self._embedding_dimension()
        manifest["updated_at"] = self._utcnow()
        self._write_manifest(manifest)
        return len(documents)

    def list_documents(self) -> list[DocumentSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id, source_name, source_type, source_uri_or_path, chunk_count FROM index_documents WHERE doc_status='active' ORDER BY source_name ASC"
            ).fetchall()
            summaries: list[DocumentSummary] = []
            for row in rows:
                chunk_rows = connection.execute(
                    "SELECT page, page_or_section, section_path_json FROM index_chunks WHERE doc_id=? AND chunk_status='active'",
                    (row["doc_id"],),
                ).fetchall()
                locations = sorted(
                    {
                        location
                        for chunk_row in chunk_rows
                        for location in [
                            self._format_location(
                                page=chunk_row["page"],
                                page_or_section=chunk_row["page_or_section"],
                                section_path_json=chunk_row["section_path_json"],
                            )
                        ]
                        if location
                    }
                )
                summaries.append(
                    DocumentSummary(
                        doc_id=str(row["doc_id"]),
                        source_name=str(row["source_name"]),
                        source_type=SourceType(str(row["source_type"])),
                        source_uri_or_path=str(row["source_uri_or_path"]),
                        chunk_count=int(row["chunk_count"]),
                        page_or_sections=locations,
                    )
                )
        return summaries

    def get_active_chunk_ids(self, doc_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM index_chunks WHERE doc_id=? AND chunk_status='active' ORDER BY chunk_id ASC",
                (doc_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def list_active_chunk_records(self) -> list[ActiveChunkRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, doc_id, retrieval_text, child_content, metadata_json, source_block_type, page, section_path_json FROM index_chunks WHERE chunk_status='active' ORDER BY doc_id ASC, chunk_id ASC"
            ).fetchall()
        records: list[ActiveChunkRecord] = []
        for row in rows:
            metadata = json.loads(str(row["metadata_json"]))
            records.append(
                ActiveChunkRecord(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    retrieval_text=str(row["retrieval_text"]),
                    child_content=str(row["child_content"] or ""),
                    metadata=metadata,
                    source_block_type=str(row["source_block_type"]) if row["source_block_type"] else None,
                    page=str(row["page"]) if row["page"] is not None else None,
                    section_path=self._decode_section_path(row["section_path_json"]),
                )
            )
        return records

    def get_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return self._current_manifest(created_at=self._utcnow())
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _load_active_documents_for_rebuild(self) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, retrieval_text, metadata_json FROM index_chunks WHERE chunk_status='active' ORDER BY doc_id ASC, chunk_id ASC"
            ).fetchall()
        documents: list[Document] = []
        for row in rows:
            metadata = json.loads(str(row["metadata_json"]))
            documents.append(Document(page_content=str(row["retrieval_text"]), metadata=metadata, id=str(row["chunk_id"])))
        return documents

    def _collect_parent_rows(self, chunks: list[IngestedChunk]) -> list[dict[str, Any]]:
        parents: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            metadata = chunk.metadata or {}
            parent_chunk_id = metadata.get("parent_chunk_id")
            if not parent_chunk_id:
                continue
            parents[str(parent_chunk_id)] = {
                "parent_chunk_id": str(parent_chunk_id),
                "doc_id": chunk.doc_id,
                "parent_store_ref": str(metadata.get("parent_store_ref") or ""),
                "parent_content_hash": str(metadata.get("parent_content_hash") or ""),
                "source_block_ids_json": json.dumps(metadata.get("parent_source_block_ids") or [], ensure_ascii=False),
            }
        return list(parents.values())

    def _chunks_to_documents(self, chunks: list[IngestedChunk]) -> list[Document]:
        return [Document(page_content=chunk.content, metadata=chunk.metadata, id=chunk.chunk_id) for chunk in chunks]

    def _build_content_fingerprint(self, chunks: list[IngestedChunk]) -> str:
        normalized_items = []
        for chunk in sorted(
            chunks,
            key=lambda item: (
                str(item.metadata.get("source_block_id") or ""),
                int(item.metadata.get("child_index") or 0),
                item.page or "",
                item.content,
            ),
        ):
            normalized_items.append(
                json.dumps(
                    {
                        "source_block_id": chunk.metadata.get("source_block_id"),
                        "source_block_type": chunk.metadata.get("source_block_type"),
                        "child_index": chunk.metadata.get("child_index"),
                        "child_count": chunk.metadata.get("child_count"),
                        "parent_chunk_id": chunk.metadata.get("parent_chunk_id"),
                        "parent_store_ref": chunk.metadata.get("parent_store_ref"),
                        "section_path": chunk.section_path,
                        "page": chunk.page,
                        "page_or_section": chunk.page_or_section,
                        "retrieval_text": chunk.content,
                        "child_content": chunk.metadata.get("child_content"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return hashlib.sha256("\n".join(normalized_items).encode("utf-8")).hexdigest()

    def _fetch_active_document(self, doc_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM index_documents WHERE doc_id=? AND doc_status='active'",
                (doc_id,),
            ).fetchone()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_documents (
                    doc_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_uri_or_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL,
                    latest_ingest_id TEXT,
                    chunk_count INTEGER NOT NULL,
                    doc_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS index_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_chunk_id TEXT,
                    parent_store_ref TEXT,
                    source_block_id TEXT,
                    source_block_type TEXT,
                    page TEXT,
                    page_or_section TEXT,
                    title TEXT,
                    section_path_json TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    child_content TEXT,
                    metadata_json TEXT NOT NULL,
                    retrieval_text_hash TEXT NOT NULL,
                    child_content_hash TEXT,
                    chunk_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS index_parents (
                    parent_chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_store_ref TEXT,
                    parent_content_hash TEXT,
                    source_block_ids_json TEXT NOT NULL,
                    parent_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS ingest_runs (
                    ingest_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_index_chunks_doc_status ON index_chunks(doc_id, chunk_status);
                CREATE INDEX IF NOT EXISTS idx_index_parents_doc_status ON index_parents(doc_id, parent_status);
                CREATE INDEX IF NOT EXISTS idx_index_documents_status ON index_documents(doc_status);
                """
            )

    def _ensure_manifest_exists(self) -> None:
        if self._manifest_path.exists():
            return
        self._write_manifest(self._current_manifest(created_at=self._utcnow()))

    def _assert_manifest_compatible(self) -> None:
        manifest = self.get_manifest()
        if int(manifest.get("schema_version", 0)) != _INDEX_SCHEMA_VERSION:
            raise ValueError("Index schema version mismatch; rebuild required")
        if str(manifest.get("embedding_model", "")) != self._settings.sentence_transformer_model:
            raise ValueError("Embedding model mismatch; rebuild required")

    def _touch_manifest(self) -> None:
        manifest = self.get_manifest()
        manifest["updated_at"] = self._utcnow()
        self._write_manifest(manifest)

    def _current_manifest(self, *, created_at: str) -> dict[str, Any]:
        return {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "index_name": self._settings.index_name,
            "embedding_model": self._settings.sentence_transformer_model,
            "embedding_dimension": self._embedding_dimension(),
            "vector_backend": "faiss",
            "vector_id_strategy": "chunk_id",
            "chunk_schema_version": _CHUNK_SCHEMA_VERSION,
            "parent_store_version": _PARENT_STORE_VERSION,
            "created_at": created_at,
            "updated_at": created_at,
        }

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self._manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_ingest_run(self, *, doc_id: str, run_type: str, chunk_count: int, status: str) -> None:
        now = self._utcnow()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingest_runs (ingest_id, doc_id, run_type, chunk_count, started_at, completed_at, status, schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid4()), doc_id, run_type, chunk_count, now, now, status, _INDEX_SCHEMA_VERSION),
            )

    def _embedding_dimension(self) -> int:
        vector = self._embeddings.embed_query("index manifest probe")
        return len(vector)

    def _format_location(self, *, page: str | None, page_or_section: str | None, section_path_json: str | None) -> str | None:
        section_path = self._decode_section_path(section_path_json)
        page_value = page or page_or_section
        if section_path and page_value:
            return f"{' > '.join(section_path)} | p{page_value}"
        if section_path:
            return " > ".join(section_path)
        if page_value:
            return str(page_value)
        return None

    def _decode_section_path(self, raw_value: str | None) -> list[str]:
        if not raw_value:
            return []
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()
