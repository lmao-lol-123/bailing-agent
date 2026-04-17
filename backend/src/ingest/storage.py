from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_object_storage_path(
    *, uploads_directory: Path, content_hash: str, file_name: str | None = None, suffix: str | None = None
) -> Path:
    resolved_suffix = (suffix or Path(str(file_name or "")).suffix or "").lower()
    return uploads_directory / "objects" / f"{content_hash}{resolved_suffix}"


def is_object_storage_path(*, path: Path, uploads_directory: Path) -> bool:
    uploads_root = uploads_directory.resolve(strict=False)
    object_root = (uploads_root / "objects").resolve(strict=False)
    candidate = path.resolve(strict=False)
    return candidate == object_root or object_root in candidate.parents
