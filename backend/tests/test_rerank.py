from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.retrieve.store import VectorStoreService


def make_case_dir(name: str) -> Path:
    root = Path("backend/.pytest-tmp") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_cross_encoder_mode_falls_back_to_heuristic_when_model_unavailable(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("rerank-cross-encoder-fallback")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        rerank_mode="cross_encoder",
        hybrid_lexical_enabled=False,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    strong_document = Document(
        page_content="FastAPI should return text event stream for SSE responses.",
        metadata={
            "doc_id": "doc-strong",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide", "Streaming"],
            "doc_type": "markdown",
            "page": "1",
            "chunk_id": "chunk-strong",
        },
        id="chunk-strong",
    )
    weak_document = Document(
        page_content="Generic engineering notes.",
        metadata={
            "doc_id": "doc-weak",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide", "Streaming"],
            "doc_type": "markdown",
            "page": "2",
            "chunk_id": "chunk-weak",
        },
        id="chunk-weak",
    )

    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(weak_document, 0.1), (strong_document, 0.1)],
    )
    monkeypatch.setattr(store._cross_encoder_reranker, "_load_model", lambda: None)

    results = store.similarity_search("text event stream", k=2)

    assert [document.metadata["doc_id"] for document in results] == ["doc-strong", "doc-weak"]
    assert results[0].metadata["rerank_score"] >= results[1].metadata["rerank_score"]
