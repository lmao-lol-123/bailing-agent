from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from backend.src.core.config import Settings
from backend.src.core.models import IngestResult
from backend.src.ingest.chunking import StructureAwareChunkingService
from backend.src.ingest.loaders import DocumentLoaderRouter, dump_normalized_documents
from backend.src.retrieve.store import VectorStoreService


class IngestionService:
    def __init__(self, settings: Settings, embeddings: object, vector_store: VectorStoreService) -> None:
        self._settings = settings
        self._router = DocumentLoaderRouter(settings)
        self._chunking = StructureAwareChunkingService(settings=settings, embeddings=embeddings)
        self._vector_store = vector_store

    def ingest_saved_file(self, file_path: Path, force_mineru: bool = False) -> IngestResult:
        documents, used_mineru = self._router.load_file(file_path=file_path, force_mineru=force_mineru)
        processed_path = self._settings.processed_directory / f"{file_path.stem}.normalized.json"
        dump_normalized_documents(documents, processed_path)
        chunks = self._chunking.chunk_documents(documents)
        self._vector_store.add_chunks(chunks)
        return IngestResult(
            source_name=file_path.name,
            source_type=documents[0].source_type,
            source_uri_or_path=str(file_path),
            doc_id=documents[0].doc_id,
            documents_loaded=len(documents),
            chunks_indexed=len(chunks),
            used_mineru=used_mineru,
        )

    def ingest_url(self, url: str) -> IngestResult:
        documents = self._router.load_url(url)
        snapshot_path = self._settings.processed_directory / f"url-{uuid4()}.normalized.json"
        dump_normalized_documents(documents, snapshot_path)
        chunks = self._chunking.chunk_documents(documents)
        self._vector_store.add_chunks(chunks)
        return IngestResult(
            source_name=documents[0].source_name,
            source_type=documents[0].source_type,
            source_uri_or_path=url,
            doc_id=documents[0].doc_id,
            documents_loaded=len(documents),
            chunks_indexed=len(chunks),
        )

    def save_upload(self, file_name: str, payload: bytes) -> Path:
        destination = self._settings.uploads_directory / f"{uuid4()}-{file_name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination

    def copy_local_file(self, source_path: Path) -> Path:
        destination = self._settings.uploads_directory / f"{uuid4()}-{source_path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return destination
