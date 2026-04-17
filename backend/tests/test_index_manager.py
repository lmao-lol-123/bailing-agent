from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from backend.src.core.config import Settings
from backend.src.core.models import IngestedChunk, SourceType
from backend.src.ingest.parent_store import JsonParentStore, ParentRecord
from backend.src.retrieve.index_manager import DocumentDeleteStepError, IndexManager
from backend.src.retrieve.store import VectorStoreService


def make_case_dir(name: str) -> Path:
    root = Path("backend/.pytest-tmp") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_chunk(
    *,
    doc_id: str,
    chunk_id: str,
    content: str,
    child_content: str,
    parent_chunk_id: str | None = None,
) -> IngestedChunk:
    parent_id = parent_chunk_id or f"parent-{doc_id}"
    return IngestedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_name="guide.md",
        source_uri_or_path="guide.md",
        page_or_section="1",
        page="1",
        title="Guide",
        section_path=["Guide", "Indexing"],
        doc_type=SourceType.MARKDOWN,
        content=content,
        metadata={
            "doc_id": doc_id,
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "page_or_section": "1",
            "page": "1",
            "title": "Guide",
            "section_path": ["Guide", "Indexing"],
            "doc_type": "markdown",
            "chunk_id": chunk_id,
            "chunk_level": "child",
            "source_block_id": f"block-{chunk_id}",
            "source_block_type": "paragraph",
            "parent_chunk_id": parent_id,
            "parent_store_ref": f"{doc_id}:{parent_id}",
            "parent_content_hash": f"hash-{parent_id}",
            "parent_source_block_ids": [f"block-{chunk_id}"],
            "child_index": 0,
            "child_count": 1,
            "child_content": child_content,
        },
    )


def test_index_manager_indexes_updates_and_rebuilds(fake_embeddings) -> None:
    case_dir = make_case_dir("index-manager")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
    )
    settings.ensure_directories()
    vector_store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    index_manager = IndexManager(
        settings=settings, embeddings=fake_embeddings, vector_store=vector_store
    )
    vector_store.set_index_manager(index_manager)

    created = index_manager.index_chunks(
        [
            make_chunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                content="retrieval text one",
                child_content="child text one",
            )
        ]
    )
    skipped = index_manager.index_chunks(
        [
            make_chunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                content="retrieval text one",
                child_content="child text one",
            )
        ]
    )
    updated = index_manager.index_chunks(
        [
            make_chunk(
                doc_id="doc-1",
                chunk_id="chunk-2",
                content="updated retrieval text",
                child_content="updated child text",
            )
        ]
    )
    rebuilt_count = index_manager.rebuild()

    assert created.created is True
    assert skipped.skipped is True
    assert updated.updated is True
    assert updated.removed_chunk_ids == ["chunk-1"]
    assert index_manager.get_active_chunk_ids("doc-1") == ["chunk-2"]
    assert rebuilt_count == 1
    assert index_manager.list_documents()[0].chunk_count == 1
    results = vector_store.similarity_search("updated retrieval", k=1)
    assert results[0].metadata["doc_id"] == "doc-1"


def test_index_manager_deletes_document_and_parent_store(fake_embeddings) -> None:
    case_dir = make_case_dir("index-manager-delete")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
    )
    settings.ensure_directories()
    vector_store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    index_manager = IndexManager(
        settings=settings, embeddings=fake_embeddings, vector_store=vector_store
    )
    vector_store.set_index_manager(index_manager)
    parent_store = JsonParentStore(settings.processed_directory)
    parent_store.save_records(
        "doc-delete",
        [
            ParentRecord(
                parent_chunk_id="parent-doc-delete",
                doc_id="doc-delete",
                source_block_ids=["block-delete"],
                parent_content="parent content",
                parent_wordpiece_count=2,
                parent_content_hash="hash-delete",
                section_path=["Guide"],
            )
        ],
    )

    index_manager.index_chunks(
        [
            make_chunk(
                doc_id="doc-delete",
                chunk_id="chunk-delete",
                content="delete retrieval text",
                child_content="delete child text",
                parent_chunk_id="parent-doc-delete",
            )
        ]
    )

    assert index_manager.delete_document("doc-delete") is True
    assert index_manager.list_documents() == []
    assert parent_store.load("parent-doc-delete") is None


def test_index_manager_delete_document_reports_faiss_step_failure(
    monkeypatch, fake_embeddings
) -> None:
    case_dir = make_case_dir("index-manager-delete-step-failure")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
    )
    settings.ensure_directories()
    vector_store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    index_manager = IndexManager(
        settings=settings, embeddings=fake_embeddings, vector_store=vector_store
    )
    vector_store.set_index_manager(index_manager)
    index_manager.index_chunks(
        [
            make_chunk(
                doc_id="doc-step",
                chunk_id="chunk-step",
                content="step retrieval",
                child_content="step child",
            )
        ]
    )
    monkeypatch.setattr(
        vector_store, "delete_ids", lambda ids: (_ for _ in ()).throw(RuntimeError("faiss error"))
    )

    with pytest.raises(DocumentDeleteStepError) as exc:
        index_manager.delete_document_with_steps("doc-step")

    assert exc.value.step == "faiss_delete"


def test_index_manager_blocks_incompatible_manifest(fake_embeddings) -> None:
    case_dir = make_case_dir("index-manager-manifest")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
    )
    settings.ensure_directories()
    vector_store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    index_manager = IndexManager(
        settings=settings, embeddings=fake_embeddings, vector_store=vector_store
    )
    manifest_path = settings.index_state_directory / settings.index_name / "manifest.json"
    manifest = index_manager.get_manifest()
    manifest["embedding_model"] = "other-model"
    manifest_path.write_text(
        __import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        index_manager.index_chunks(
            [
                make_chunk(
                    doc_id="doc-1",
                    chunk_id="chunk-1",
                    content="retrieval text one",
                    child_content="child text one",
                )
            ]
        )
