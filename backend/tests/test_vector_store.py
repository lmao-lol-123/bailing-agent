from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import IngestedChunk, RetrievalFilter, SourceType
from backend.src.retrieve.store import VectorStoreService


def make_case_dir(name: str) -> Path:
    root = Path("backend/.pytest-tmp") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_vector_store_adds_and_lists_documents(fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
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
                page="1",
                title="Guide",
                section_path=["Guide", "Streaming"],
                doc_type=SourceType.MARKDOWN,
                content="FastAPI streams answers using SSE.",
                metadata={
                    "doc_id": "doc-1",
                    "source_name": "guide.md",
                    "source_type": "markdown",
                    "source_uri_or_path": "guide.md",
                    "page_or_section": "1",
                    "title": "Guide",
                    "section_path": ["Guide", "Streaming"],
                    "page": "1",
                    "doc_type": "markdown",
                    "updated_at": "2026-04-03T10:00:00+00:00",
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


def test_vector_store_build_citations_deduplicates_same_location() -> None:
    documents = [
        Document(
            page_content="Chunk A",
            metadata={
                "doc_id": "doc-1",
                "source_name": "guide.md",
                "source_uri_or_path": "guide.md",
                "page_or_section": "1",
                "page": "1",
                "title": "Guide",
                "section_path": ["Guide", "Streaming"],
                "doc_type": "markdown",
                "updated_at": "2026-04-03T10:00:00+00:00",
            },
        ),
        Document(
            page_content="Chunk B",
            metadata={
                "doc_id": "doc-1",
                "source_name": "guide.md",
                "source_uri_or_path": "guide.md",
                "page_or_section": "1",
                "page": "1",
                "title": "Guide",
                "section_path": ["Guide", "Streaming"],
                "doc_type": "markdown",
                "updated_at": "2026-04-03T10:00:00+00:00",
            },
        ),
    ]

    citations = VectorStoreService.build_citations(documents)

    assert len(citations) == 1
    assert citations[0].index == 1
    assert citations[0].source_name == "guide.md"
    assert citations[0].section_path == ["Guide", "Streaming"]
    assert citations[0].page == "1"


def test_vector_store_applies_metadata_filter_and_reranks(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-filter-rerank")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        rerank_candidate_multiplier=4,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    markdown_doc = Document(
        page_content="FastAPI streaming response should use text/event-stream.",
        metadata={
            "doc_id": "doc-markdown",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide", "Streaming"],
            "doc_type": "markdown",
            "page": "1",
        },
    )
    txt_doc = Document(
        page_content="FastAPI streaming response should use text/event-stream.",
        metadata={
            "doc_id": "doc-txt",
            "source_name": "notes.txt",
            "source_type": "txt",
            "source_uri_or_path": "notes.txt",
            "section_path": ["Notes"],
            "doc_type": "txt",
            "page": "1",
        },
    )
    weak_doc = Document(
        page_content="Generic notes without the answer keyword.",
        metadata={
            "doc_id": "doc-weak",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide", "Streaming"],
            "doc_type": "markdown",
            "page": "2",
        },
    )

    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(weak_doc, 0.2), (txt_doc, 0.2), (markdown_doc, 0.2)],
    )

    results = store.similarity_search(
        "FastAPI text/event-stream",
        k=2,
        metadata_filter=RetrievalFilter(
            doc_types=[SourceType.MARKDOWN],
            section_path_prefix=["Guide", "Streaming"],
        ),
    )

    assert [document.metadata["doc_id"] for document in results] == ["doc-markdown", "doc-weak"]
