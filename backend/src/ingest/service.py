from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from uuid import uuid4

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import IngestResult, SessionFileStatus, SessionFileSummary
from backend.src.ingest.chunking import StructureAwareChunkingService
from backend.src.ingest.loaders import DocumentLoaderRouter, dump_normalized_documents
from backend.src.ingest.storage import build_object_storage_path, sha256_bytes, sha256_file
from backend.src.retrieve.index_manager import DocumentDeleteStepError, IndexManager
from backend.src.retrieve.store import VectorStoreService


class SessionFileCleanupError(RuntimeError):
    def __init__(self, *, step: str, message: str) -> None:
        self.step = step
        super().__init__(message)


class SessionFileRecoveryStateError(RuntimeError):
    pass


_WINDOWS_RESERVED_FILENAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        embeddings: object,
        vector_store: VectorStoreService,
        index_manager: IndexManager,
        chat_history_store: ChatHistoryStore,
    ) -> None:
        self._settings = settings
        self._router = DocumentLoaderRouter(settings)
        self._chunking = StructureAwareChunkingService(
            settings=settings, embeddings=embeddings, persist_parent_store=True
        )
        self._vector_store = vector_store
        self._index_manager = index_manager
        self._chat_history_store = chat_history_store

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
        documents, used_mineru = self._router.load_file(
            file_path=file_path,
            force_mineru=force_mineru,
            is_sensitive=is_sensitive,
            mineru_mode=mineru_mode,
            source_name=source_name,
            doc_id_override=doc_id_override,
        )
        self._apply_scope_metadata(documents, scope=scope, session_id=session_id)
        processed_path = self._snapshot_path_for_doc_id(documents[0].doc_id)
        dump_normalized_documents(documents, processed_path)
        chunks = self._chunking.chunk_documents(documents)
        index_result = self._index_manager.index_chunks(chunks)
        return IngestResult(
            source_name=documents[0].source_name,
            source_type=documents[0].source_type,
            source_uri_or_path=str(file_path),
            doc_id=documents[0].doc_id,
            documents_loaded=len(documents),
            chunks_indexed=len(index_result.added_chunk_ids),
            rebuild_triggered=index_result.updated,
            used_mineru=used_mineru,
        )

    def ingest_url(self, url: str) -> IngestResult:
        documents = self._router.load_url(url)
        self._apply_scope_metadata(documents, scope="global", session_id=None)
        snapshot_path = self._settings.processed_directory / f"url-{uuid4()}.normalized.json"
        dump_normalized_documents(documents, snapshot_path)
        chunks = self._chunking.chunk_documents(documents)
        index_result = self._index_manager.index_chunks(chunks)
        return IngestResult(
            source_name=documents[0].source_name,
            source_type=documents[0].source_type,
            source_uri_or_path=url,
            doc_id=documents[0].doc_id,
            documents_loaded=len(documents),
            chunks_indexed=len(index_result.added_chunk_ids),
            rebuild_triggered=index_result.updated,
        )

    def save_upload(self, file_name: str, payload: bytes) -> Path:
        return self._store_object_from_payload(file_name=file_name, payload=payload)

    def save_session_upload(self, session_id: str, file_name: str, payload: bytes) -> Path:
        return self._store_object_from_payload(file_name=file_name, payload=payload)

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
        saved_path = self.save_session_upload(session_id, file_name, payload)
        doc_id = self._build_managed_doc_id(
            file_path=saved_path,
            session_id=session_id,
        )
        session_file = self._chat_history_store.create_session_file_upload(
            session_id=session_id,
            source_name=file_name,
            source_uri_or_path=str(saved_path),
        )
        try:
            documents, _ = self._router.load_file(
                file_path=saved_path,
                force_mineru=force_mineru,
                is_sensitive=is_sensitive,
                mineru_mode=mineru_mode,
                source_name=file_name,
                doc_id_override=doc_id,
            )
            self._apply_scope_metadata(documents, scope="session", session_id=session_id)
            page_count = self._estimate_page_count(documents)
            parsed_summary = self._chat_history_store.update_session_file(
                session_id=session_id,
                file_id=session_file.file_id,
                status=SessionFileStatus.PARSED,
                page_count=page_count,
                source_name=file_name,
                source_uri_or_path=str(saved_path),
            )
            if parsed_summary is not None:
                session_file = parsed_summary

            processed_path = self._snapshot_path_for_doc_id(documents[0].doc_id)
            dump_normalized_documents(documents, processed_path)
            chunks = self._chunking.chunk_documents(documents)
            self._index_manager.index_chunks(chunks)
            indexed_summary = self._chat_history_store.update_session_file(
                session_id=session_id,
                file_id=session_file.file_id,
                status=SessionFileStatus.INDEXED,
                doc_id=documents[0].doc_id,
                doc_type=documents[0].source_type,
                page_count=page_count,
                source_name=file_name,
                source_uri_or_path=str(saved_path),
            )
            if indexed_summary is not None:
                return indexed_summary
            return SessionFileSummary(
                file_id=session_file.file_id,
                session_id=session_id,
                doc_id=documents[0].doc_id,
                source_name=file_name,
                source_uri_or_path=str(saved_path),
                doc_type=documents[0].source_type,
                page_count=page_count,
                status=SessionFileStatus.INDEXED,
                error_code=None,
                error_message=None,
                created_at=session_file.created_at,
                updated_at=session_file.updated_at,
            )
        except Exception as exc:
            failed_summary = self._chat_history_store.update_session_file(
                session_id=session_id,
                file_id=session_file.file_id,
                status=SessionFileStatus.FAILED,
                source_name=file_name,
                source_uri_or_path=str(saved_path),
                error_code=self._classify_upload_error(exc),
                error_message=str(exc)[:500],
            )
            if failed_summary is not None:
                return failed_summary
            return SessionFileSummary(
                file_id=session_file.file_id,
                session_id=session_id,
                source_name=file_name,
                source_uri_or_path=str(saved_path),
                status=SessionFileStatus.FAILED,
                error_code=self._classify_upload_error(exc),
                error_message=str(exc)[:500],
                created_at=session_file.created_at,
                updated_at=session_file.updated_at,
            )

    def delete_session_file(self, *, session_id: str, file_id: str) -> SessionFileSummary | None:
        session_file = self._chat_history_store.get_session_file(
            session_id=session_id, file_id=file_id
        )
        if session_file is None:
            return None

        if session_file.status is SessionFileStatus.RECOVER_PENDING:
            raise SessionFileRecoveryStateError(
                "File is in recover_pending state; call recover-delete endpoint."
            )

        try:
            self._cleanup_session_file_for_delete(session_file)
        except SessionFileCleanupError as exc:
            self._mark_recover_pending(
                session_id=session_id, file_id=file_id, error_message=str(exc), step=exc.step
            )
            raise

        return self._chat_history_store.delete_session_file(session_id, file_id)

    def delete_session(self, *, session_id: str) -> None:
        session_files = self._chat_history_store.list_session_files(session_id)
        excluded_entries = {
            (session_file.session_id, session_file.file_id) for session_file in session_files
        }
        for session_file in session_files:
            self._cleanup_session_file_for_delete_with_exclusions(
                session_file,
                exclude_entries=excluded_entries,
            )
        self._chat_history_store.delete_session(session_id)

    def recover_session_file_deletion(self, *, session_id: str, file_id: str) -> str:
        session_file = self._chat_history_store.get_session_file(
            session_id=session_id, file_id=file_id
        )
        if session_file is None:
            return "already_deleted"

        if session_file.status is not SessionFileStatus.RECOVER_PENDING:
            raise SessionFileRecoveryStateError(
                "Only recover_pending files can run recover-delete."
            )

        try:
            self._cleanup_session_file_for_delete(session_file)
        except SessionFileCleanupError as exc:
            self._mark_recover_pending(
                session_id=session_id, file_id=file_id, error_message=str(exc), step=exc.step
            )
            raise

        self._chat_history_store.delete_session_file(session_id, file_id)
        return "ok"

    def delete_documents(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            document_summary = self._index_manager.get_document_summary(doc_id)
            deleted = self._index_manager.delete_document(doc_id)
            if deleted and document_summary is not None:
                self._cleanup_deleted_document_artifacts(
                    doc_id=doc_id,
                    source_uri_or_path=document_summary.source_uri_or_path,
                )

    def recover_pending_session_file_deletions(
        self, *, session_id: str | None = None
    ) -> list[tuple[str, str, str]]:
        sessions = (
            [session_id]
            if session_id
            else [item.session_id for item in self._chat_history_store.list_sessions()]
        )
        outcomes: list[tuple[str, str, str]] = []
        for target_session_id in sessions:
            for session_file in self._chat_history_store.list_session_files(target_session_id):
                if session_file.status is not SessionFileStatus.RECOVER_PENDING:
                    continue
                try:
                    outcome = self.recover_session_file_deletion(
                        session_id=target_session_id, file_id=session_file.file_id
                    )
                except SessionFileCleanupError:
                    outcomes.append((target_session_id, session_file.file_id, "failed"))
                    continue
                except SessionFileRecoveryStateError:
                    outcomes.append((target_session_id, session_file.file_id, "skipped"))
                    continue
                outcomes.append((target_session_id, session_file.file_id, outcome))
        return outcomes

    def copy_local_file(self, source_path: Path) -> Path:
        return self._store_object_from_file(source_path)

    @staticmethod
    def _apply_scope_metadata(
        documents: list[object], *, scope: str, session_id: str | None
    ) -> None:
        for document in documents:
            metadata = getattr(document, "metadata", None)
            if metadata is None:
                continue
            metadata["scope"] = scope
            if session_id:
                metadata["session_id"] = session_id

    @staticmethod
    def _estimate_page_count(documents: list[object]) -> int | None:
        pages: set[str] = set()
        for document in documents:
            page = getattr(document, "page", None) or getattr(document, "page_or_section", None)
            if page is not None and str(page).strip():
                pages.add(str(page))
        if pages:
            return len(pages)
        return len(documents) if documents else None

    @staticmethod
    def _classify_upload_error(exc: Exception) -> str:
        message = str(exc).lower()
        if "mineru" in message or "loader" in message or "parse" in message or "decode" in message:
            return "parse_failed"
        if "index" in message or "faiss" in message or "embedding" in message:
            return "index_failed"
        return "upload_processing_failed"

    def _cleanup_index_for_session_file(
        self,
        session_file: SessionFileSummary,
        *,
        exclude_entries: set[tuple[str, str]] | None = None,
    ) -> None:
        if not session_file.doc_id:
            return
        if self._count_session_file_references(
            doc_id=session_file.doc_id,
            exclude_entries=exclude_entries,
        ):
            return
        try:
            self._index_manager.delete_document_with_steps(session_file.doc_id)
        except DocumentDeleteStepError as exc:
            raise SessionFileCleanupError(step=exc.step, message=str(exc)) from exc
        except Exception as exc:
            raise SessionFileCleanupError(step="unknown", message=f"[step=unknown] {exc}") from exc

    def _mark_recover_pending(
        self, *, session_id: str, file_id: str, error_message: str, step: str
    ) -> None:
        normalized_error = error_message.strip()
        if not normalized_error.startswith("[step="):
            normalized_error = f"[step={step}] {normalized_error}"
        self._chat_history_store.update_session_file(
            session_id=session_id,
            file_id=file_id,
            status=SessionFileStatus.RECOVER_PENDING,
            error_code="delete_cleanup_failed",
            error_message=normalized_error[:500],
        )

    def _cleanup_session_file_for_delete(self, session_file: SessionFileSummary) -> None:
        self._cleanup_session_file_for_delete_with_exclusions(
            session_file,
            exclude_entries={(session_file.session_id, session_file.file_id)},
        )

    def _cleanup_session_file_for_delete_with_exclusions(
        self,
        session_file: SessionFileSummary,
        *,
        exclude_entries: set[tuple[str, str]],
    ) -> None:
        if session_file.doc_id:
            self._cleanup_index_for_session_file(
                session_file,
                exclude_entries=exclude_entries,
            )
        self._cleanup_session_file_artifacts(
            session_file,
            exclude_entries=exclude_entries,
        )

    def _cleanup_session_file_artifacts(
        self,
        session_file: SessionFileSummary,
        *,
        exclude_entries: set[tuple[str, str]] | None = None,
    ) -> None:
        upload_path = self._resolve_path_within_base(
            raw_path=session_file.source_uri_or_path,
            base_directory=self._settings.uploads_directory,
        )
        if session_file.doc_id and not self._count_session_file_references(
            doc_id=session_file.doc_id,
            exclude_entries=exclude_entries,
        ):
            self._delete_doc_snapshot_if_present(session_file.doc_id)
            if upload_path is not None:
                legacy_processed_snapshot = self._legacy_snapshot_path(upload_path)
                self._delete_file_if_present(
                    legacy_processed_snapshot,
                    step="processed_snapshot_delete",
                )
        elif upload_path is not None and not self._count_source_path_references(
            source_uri_or_path=str(upload_path),
            exclude_entries=exclude_entries,
        ):
            self._delete_file_if_present(
                self._legacy_snapshot_path(upload_path),
                step="processed_snapshot_delete",
            )

        if upload_path is not None and not self._count_source_path_references(
            source_uri_or_path=str(upload_path),
            exclude_entries=exclude_entries,
        ):
            self._delete_file_if_present(upload_path, step="upload_file_delete")
            self._prune_empty_directories(
                upload_path.parent,
                stop_directory=self._settings.uploads_directory,
            )

    def _cleanup_deleted_document_artifacts(self, *, doc_id: str, source_uri_or_path: str) -> None:
        if not self._count_session_file_references(doc_id=doc_id):
            self._delete_doc_snapshot_if_present(doc_id)
            upload_path = self._resolve_path_within_base(
                raw_path=source_uri_or_path,
                base_directory=self._settings.uploads_directory,
            )
            if upload_path is not None:
                self._delete_file_if_present(
                    self._legacy_snapshot_path(upload_path),
                    step="processed_snapshot_delete",
                )
        else:
            upload_path = self._resolve_path_within_base(
                raw_path=source_uri_or_path,
                base_directory=self._settings.uploads_directory,
            )
            if upload_path is None:
                return
        if upload_path is None:
            return
        if not self._count_source_path_references(source_uri_or_path=str(upload_path)):
            self._delete_file_if_present(upload_path, step="upload_file_delete")
            self._prune_empty_directories(
                upload_path.parent,
                stop_directory=self._settings.uploads_directory,
            )

    @staticmethod
    def _sanitize_upload_filename(file_name: str | None) -> str:
        normalized_name = str(file_name or "").replace("\\", "/").split("/")[-1].strip()
        normalized_name = _WINDOWS_RESERVED_FILENAME_CHARS.sub("_", normalized_name)
        normalized_name = normalized_name.rstrip(". ")
        if normalized_name in {"", ".", ".."}:
            return "upload"
        return normalized_name

    @staticmethod
    def _build_upload_destination(*, base_directory: Path, file_name: str) -> Path:
        safe_name = IngestionService._sanitize_upload_filename(file_name)
        destination = (base_directory / f"{uuid4()}-{safe_name}").resolve(strict=False)
        base_path = base_directory.resolve(strict=False)
        try:
            destination.relative_to(base_path)
        except ValueError as exc:
            raise ValueError("Upload destination resolved outside the configured uploads directory") from exc
        return destination

    @staticmethod
    def _resolve_path_within_base(*, raw_path: str, base_directory: Path) -> Path | None:
        candidate = Path(raw_path).resolve(strict=False)
        base_path = base_directory.resolve(strict=False)
        try:
            candidate.relative_to(base_path)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _delete_file_if_present(path: Path, *, step: str) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise SessionFileCleanupError(step=step, message=f"[step={step}] {exc}") from exc

    @staticmethod
    def _prune_empty_directories(directory: Path, *, stop_directory: Path) -> None:
        current = directory.resolve(strict=False)
        stop_path = stop_directory.resolve(strict=False)
        while current != stop_path and stop_path in current.parents:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _snapshot_path_for_doc_id(self, doc_id: str) -> Path:
        return self._settings.processed_directory / f"{doc_id}.normalized.json"

    def _legacy_snapshot_path(self, upload_path: Path) -> Path:
        return self._settings.processed_directory.resolve(strict=False) / f"{upload_path.stem}.normalized.json"

    def _delete_doc_snapshot_if_present(self, doc_id: str) -> None:
        self._delete_file_if_present(
            self._snapshot_path_for_doc_id(doc_id),
            step="processed_snapshot_delete",
        )

    def _build_managed_doc_id(self, *, file_path: Path, session_id: str | None) -> str:
        source_type = self._router._detect_source_type(file_path)
        normalized_source = self._router._normalize_file_source_key(file_path)
        scope_key = (
            normalized_source
            if not session_id
            else f"{normalized_source}:session:{session_id.lower()}"
        )
        digest = hashlib.sha1(f"{source_type.value}:{scope_key}".encode("utf-8")).hexdigest()[:16]
        return f"doc-{digest}"

    def _store_object_from_payload(self, *, file_name: str, payload: bytes) -> Path:
        safe_name = self._sanitize_upload_filename(file_name)
        content_hash = sha256_bytes(payload)
        destination = build_object_storage_path(
            uploads_directory=self._settings.uploads_directory,
            content_hash=content_hash,
            file_name=safe_name,
        ).resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
        return destination

    def _store_object_from_file(self, source_path: Path) -> Path:
        content_hash = sha256_file(source_path)
        destination = build_object_storage_path(
            uploads_directory=self._settings.uploads_directory,
            content_hash=content_hash,
            suffix=source_path.suffix,
        ).resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_path, destination)
        return destination

    def _count_session_file_references(
        self,
        *,
        doc_id: str,
        exclude_entries: set[tuple[str, str]] | None = None,
    ) -> int:
        excluded = exclude_entries or set()
        return sum(
            1
            for session_file in self._chat_history_store.list_all_session_files()
            if session_file.doc_id == doc_id
            and (session_file.session_id, session_file.file_id) not in excluded
        )

    def _count_source_path_references(
        self,
        *,
        source_uri_or_path: str,
        exclude_entries: set[tuple[str, str]] | None = None,
    ) -> int:
        excluded = exclude_entries or set()
        session_file_refs = sum(
            1
            for session_file in self._chat_history_store.list_all_session_files()
            if session_file.source_uri_or_path == source_uri_or_path
            and (session_file.session_id, session_file.file_id) not in excluded
        )
        return session_file_refs + self._index_manager.count_active_documents_by_source_uri_or_path(
            source_uri_or_path
        )
