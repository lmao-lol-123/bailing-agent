from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from backend.src.core.dependencies import AppContainer, get_container
from backend.src.ingest.storage import (
    build_object_storage_path,
    is_object_storage_path,
    sha256_file,
)


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve(strict=False))


def cleanup_data_storage(*, container: AppContainer, apply: bool) -> dict[str, Any]:
    uploads_root = container.settings.uploads_directory.resolve(strict=False)
    processed_root = container.settings.processed_directory.resolve(strict=False)
    legacy_upload_paths = [
        path.resolve(strict=False)
        for path in uploads_root.rglob("*")
        if path.is_file() and not is_object_storage_path(path=path, uploads_directory=uploads_root)
    ]
    active_documents = container.index_manager.list_documents()
    session_files = container.chat_history_store.list_all_session_files()

    upload_migrations: list[dict[str, Any]] = []
    legacy_to_object: dict[str, str] = {}
    for legacy_path in sorted(legacy_upload_paths):
        content_hash = sha256_file(legacy_path)
        object_path = build_object_storage_path(
            uploads_directory=uploads_root,
            content_hash=content_hash,
            suffix=legacy_path.suffix,
        ).resolve(strict=False)
        legacy_key = _normalize_path(legacy_path)
        object_key = _normalize_path(object_path)
        legacy_to_object[legacy_key] = object_key
        upload_migrations.append(
            {
                "legacy_path": legacy_key,
                "object_path": object_key,
                "exists_already": object_path.exists(),
                "size": legacy_path.stat().st_size,
            }
        )
        if apply:
            object_path.parent.mkdir(parents=True, exist_ok=True)
            if not object_path.exists():
                object_path.write_bytes(legacy_path.read_bytes())

    updated_session_files = 0
    for session_file in session_files:
        legacy_key = _normalize_path(session_file.source_uri_or_path)
        object_key = legacy_to_object.get(legacy_key)
        if object_key is None or object_key == legacy_key:
            continue
        updated_session_files += 1
        if apply:
            container.chat_history_store.update_session_file(
                session_id=session_file.session_id,
                file_id=session_file.file_id,
                status=session_file.status,
                doc_id=session_file.doc_id,
                doc_type=session_file.doc_type,
                page_count=session_file.page_count,
                source_name=session_file.source_name,
                source_uri_or_path=object_key,
                error_code=session_file.error_code,
                error_message=session_file.error_message,
            )

    updated_documents = 0
    snapshot_migrations: list[dict[str, Any]] = []
    for document in active_documents:
        legacy_key = _normalize_path(document.source_uri_or_path)
        object_key = legacy_to_object.get(legacy_key)
        if object_key is not None and object_key != legacy_key:
            updated_documents += 1
            if apply:
                container.index_manager.rewrite_document_source_uri(
                    doc_id=document.doc_id,
                    source_uri_or_path=object_key,
                )

        legacy_snapshot = processed_root / f"{Path(legacy_key).stem}.normalized.json"
        target_snapshot = processed_root / f"{document.doc_id}.normalized.json"
        legacy_exists = legacy_snapshot.exists()
        target_exists = target_snapshot.exists()
        action = "noop"
        conflict = False
        if legacy_exists and not target_exists:
            action = "rename"
            if apply:
                legacy_snapshot.replace(target_snapshot)
        elif legacy_exists and target_exists:
            if legacy_snapshot.read_bytes() == target_snapshot.read_bytes():
                action = "delete_duplicate"
                if apply:
                    legacy_snapshot.unlink()
            else:
                action = "conflict"
                conflict = True
        snapshot_migrations.append(
            {
                "doc_id": document.doc_id,
                "legacy_snapshot": _normalize_path(legacy_snapshot),
                "target_snapshot": _normalize_path(target_snapshot),
                "action": action,
                "conflict": conflict,
            }
        )

    if apply and updated_documents:
        container.index_manager.rebuild()

    stale_processed_dirs = [
        path.resolve(strict=False)
        for path in processed_root.iterdir()
        if path.is_dir()
    ]
    stale_processed_snapshots = [
        path.resolve(strict=False)
        for path in processed_root.glob("*.normalized.json")
        if not path.name.startswith("doc-")
        and all(
            _normalize_path(path)
            != item["legacy_snapshot"]
            for item in snapshot_migrations
        )
    ]
    if apply:
        for directory in stale_processed_dirs:
            shutil.rmtree(directory, ignore_errors=False)
        for snapshot in stale_processed_snapshots:
            snapshot.unlink()

    deleted_legacy_uploads = 0
    if apply:
        active_source_paths = {
            _normalize_path(item.source_uri_or_path)
            for item in container.index_manager.list_documents()
        }
        active_source_paths.update(
            _normalize_path(item.source_uri_or_path)
            for item in container.chat_history_store.list_all_session_files()
        )
        for legacy_path in legacy_upload_paths:
            normalized_legacy = _normalize_path(legacy_path)
            if normalized_legacy in active_source_paths:
                continue
            legacy_path.unlink()
            deleted_legacy_uploads += 1

    return {
        "apply": apply,
        "uploads_scanned": len(legacy_upload_paths),
        "uploads_migrated": upload_migrations,
        "session_files_updated": updated_session_files,
        "documents_updated": updated_documents,
        "snapshots": snapshot_migrations,
        "stale_processed_dirs": [_normalize_path(path) for path in stale_processed_dirs],
        "stale_processed_snapshots": [
            _normalize_path(path) for path in stale_processed_snapshots
        ],
        "deleted_legacy_uploads": deleted_legacy_uploads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize legacy upload storage into content-addressed object storage."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the cleanup. Without this flag, the command only reports planned changes.",
    )
    args = parser.parse_args()
    report = cleanup_data_storage(container=get_container(), apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
