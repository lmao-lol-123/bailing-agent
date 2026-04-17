from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from backend.src.core.config import Settings
from backend.src.core.models import NormalizedDocument, SourceType
from backend.src.ingest.chunking import StructureAwareChunkingService
from backend.src.ingest.parent_store import JsonParentStore


@pytest.fixture()
def processed_dir() -> Path:
    root = Path("backend/test_runtime") / f"parent-store-{uuid4()}"
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    try:
        yield processed
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chunking_writes_parent_records_to_json_store(fake_embeddings, processed_dir) -> None:
    settings = Settings(
        processed_directory=processed_dir,
        chunk_max_word_pieces=24,
        chunk_overlap_word_pieces=4,
    )
    service = StructureAwareChunkingService(
        settings=settings, embeddings=fake_embeddings, persist_parent_store=True
    )
    long_text = " ".join(f"token{index}" for index in range(80))

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-parent-store",
                source_type=SourceType.TXT,
                source_name="notes.txt",
                source_uri_or_path="notes.txt",
                title="Notes",
                content=long_text,
                metadata={
                    "block_id": "block-long",
                    "block_type": "paragraph",
                    "layout_role": "body",
                },
            )
        ]
    )

    assert chunks
    parent_chunk_id = chunks[0].metadata["parent_chunk_id"]
    assert all(chunk.metadata["parent_stored"] is True for chunk in chunks)
    assert all(chunk.metadata["parent_content_available"] == "store" for chunk in chunks)
    assert all(
        chunk.metadata["parent_store_ref"] == f"doc-parent-store:{parent_chunk_id}"
        for chunk in chunks
    )
    assert all("parent_content" not in chunk.metadata for chunk in chunks)

    parent_store = JsonParentStore(settings.processed_directory)
    record = parent_store.load(parent_chunk_id)

    assert record is not None
    assert record.parent_chunk_id == parent_chunk_id
    assert record.doc_id == "doc-parent-store"
    assert record.parent_content == long_text
    assert record.source_block_ids == ["block-long"]


def test_parent_store_persists_assembled_parent_content(fake_embeddings, processed_dir) -> None:
    settings = Settings(
        processed_directory=processed_dir,
        chunk_max_word_pieces=96,
        chunk_overlap_word_pieces=8,
    )
    service = StructureAwareChunkingService(
        settings=settings, embeddings=fake_embeddings, persist_parent_store=True
    )

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-assembled-store",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                title="Paper",
                section_path=["Paper", "Figures"],
                content="Image block",
                metadata={
                    "block_id": "image-store",
                    "block_type": "image",
                    "layout_role": "body",
                    "block_order": 1,
                    "graph_edges": [
                        {"type": "has_caption", "target_block_id": "caption-store", "weight": 0.98}
                    ],
                },
            ),
            NormalizedDocument(
                doc_id="doc-assembled-store",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                title="Paper",
                section_path=["Paper", "Figures"],
                content="Figure 2. Stored caption context.",
                metadata={
                    "block_id": "caption-store",
                    "block_type": "caption",
                    "layout_role": "caption",
                    "block_order": 2,
                },
            ),
        ]
    )

    image_chunk = next(
        chunk for chunk in chunks if chunk.metadata["source_block_id"] == "image-store"
    )
    parent_store = JsonParentStore(settings.processed_directory)
    record = parent_store.load(image_chunk.metadata["parent_chunk_id"])

    assert record is not None
    assert record.source_block_ids == ["image-store", "caption-store"]
    assert "Figure 2. Stored caption context." in record.parent_content
    assert record.metadata["parent_assembly_strategy"] == "relation_limited"
