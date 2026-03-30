from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.core.config import Settings
from src.core.models import IngestedChunk
from src.retrieve.store import VectorStoreService


def make_case_dir(name: str) -> Path:
    root = Path("test_runtime") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_vector_store_adds_and_lists_documents(fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        chroma_persist_directory=case_dir / "chroma",
        chroma_collection_name="test-collection",
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    store.add_chunks(
        [
            IngestedChunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_name="guide.md",
                source_uri_or_path="guide.md",
                page_or_section="1",
                content="FastAPI streams answers using SSE.",
                metadata={
                    "doc_id": "doc-1",
                    "source_name": "guide.md",
                    "source_type": "markdown",
                    "source_uri_or_path": "guide.md",
                    "page_or_section": "1",
                },
            )
        ]
    )

    results = store.similarity_search("How does FastAPI stream SSE?", k=1)
    documents = store.list_documents()

    assert len(results) == 1
    assert results[0].metadata["doc_id"] == "doc-1"
    assert len(documents) == 1
    assert documents[0].chunk_count == 1
    assert documents[0].source_name == "guide.md"
