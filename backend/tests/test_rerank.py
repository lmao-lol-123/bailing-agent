from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.retrieve.rerank import HeuristicReranker, RetrievedCandidate
from backend.src.retrieve.store import VectorStoreService


def make_case_dir(name: str) -> Path:
    root = Path("backend/.pytest-tmp") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_cross_encoder_mode_falls_back_to_heuristic_when_model_unavailable(
    monkeypatch, fake_embeddings
) -> None:
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


def test_heuristic_reranker_prefers_regulation_clause_over_cover_page() -> None:
    reranker = HeuristicReranker(Settings())
    cover_candidate = RetrievedCandidate(
        chunk_id="chunk-cover",
        document=Document(
            page_content="建筑基桩检测技术规范",
            metadata={
                "doc_id": "doc-cover",
                "source_name": "spec.pdf",
                "source_block_type": "paragraph",
                "title": "建筑基桩检测技术规范",
            },
        ),
        fusion_score=0.3,
    )
    clause_candidate = RetrievedCandidate(
        chunk_id="chunk-clause",
        document=Document(
            page_content="`1.0.2` 本规范适用于建筑工程基桩的承载力和桩身完整性的检测与评价。",
            metadata={
                "doc_id": "doc-clause",
                "source_name": "spec.pdf",
                "source_block_type": "paragraph",
                "title": "建筑基桩检测技术规范",
                "clause_id": "1.0.2",
                "is_normative_clause": True,
                "is_regulation_anchor": True,
            },
        ),
        fusion_score=0.3,
    )

    ranked = reranker.rerank(
        query="本规范适用于哪些内容的检测与评价？",
        candidates=[cover_candidate, clause_candidate],
        limit=2,
    )

    assert [candidate.chunk_id for candidate in ranked] == ["chunk-clause", "chunk-cover"]


def test_heuristic_reranker_prefers_numeric_table_row_for_regulation_table_question() -> None:
    reranker = HeuristicReranker(Settings())
    caption_candidate = RetrievedCandidate(
        chunk_id="chunk-caption",
        document=Document(
            page_content="图 10.3.2 声测管布置示意图",
            metadata={
                "doc_id": "doc-caption",
                "source_name": "spec.pdf",
                "source_block_type": "caption",
                "title": "建筑基桩检测技术规范",
                "table_id": "3.2.5",
            },
        ),
        fusion_score=0.32,
    )
    row_candidate = RetrievedCandidate(
        chunk_id="chunk-row",
        document=Document(
            page_content="表3.2.5 砂土 7",
            metadata={
                "doc_id": "doc-row",
                "source_name": "spec.pdf",
                "source_block_type": "paragraph",
                "title": "建筑基桩检测技术规范",
                "table_id": "3.2.5",
                "pseudo_table_row_key": "砂土",
                "pseudo_table_row_value": "7",
                "has_numeric_anchor": True,
                "numeric_anchor_terms": ["7"],
                "is_regulation_anchor": True,
            },
        ),
        fusion_score=0.28,
    )

    ranked = reranker.rerank(
        query="表3.2.5中砂土的休止时间是多少天？",
        candidates=[caption_candidate, row_candidate],
        limit=2,
    )

    assert [candidate.chunk_id for candidate in ranked] == ["chunk-row", "chunk-caption"]
