from __future__ import annotations

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.retrieve.rerank import RetrievedCandidate
from backend.src.retrieve.store import VectorStoreService
from backend.tests.test_vector_store import make_case_dir


def _make_candidate(*, chunk_id: str, doc_id: str, block_type: str, fusion_score: float = 0.1) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        document=Document(
            page_content=f"{block_type} content {chunk_id}",
            metadata={
                "doc_id": doc_id,
                "source_name": "guide.md",
                "source_uri_or_path": "guide.md",
                "source_type": "markdown",
                "doc_type": "markdown",
                "page": "1",
                "section_path": ["Guide"],
                "chunk_id": chunk_id,
                "source_block_type": block_type,
            },
            id=chunk_id,
        ),
        retrieval_sources=("dense",),
        dense_rank=1,
        dense_score=fusion_score,
        fusion_score=fusion_score,
    )


def test_structure_route_retries_once_and_adds_keyword_variant(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("retrieval-retry-structure")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        targeted_retry_enabled=True,
        targeted_retry_max_attempts=1,
        rerank_mode="off",
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    paragraph_candidate = _make_candidate(chunk_id="chunk-p", doc_id="doc-p", block_type="paragraph")
    table_candidate = _make_candidate(chunk_id="chunk-t", doc_id="doc-t", block_type="table", fusion_score=0.3)
    calls: list[list[str]] = []

    def fake_run_round(*, query_variants, decision, metadata_filter, targeted_plan):
        calls.append([variant.variant_id for variant in query_variants])
        if len(calls) == 1:
            return [paragraph_candidate]
        if len(calls) == 2:
            return [paragraph_candidate]
        return [table_candidate, paragraph_candidate]

    monkeypatch.setattr(store, "_run_retrieval_round", fake_run_round)
    monkeypatch.setattr(store, "_rerank_candidates", lambda query, candidates, top_k: sorted(candidates, key=lambda item: -(item.fusion_score or 0.0))[:top_k], )
    monkeypatch.setattr(
        store,
        "_replace_child_hits_with_parent_content",
        lambda *, reranked_documents, top_k: [document for document, _ in reranked_documents[:top_k]],
    )

    results = store.similarity_search("Explain Figure 2 latency tradeoff", k=2)

    assert {document.metadata["doc_id"] for document in results} == {"doc-t", "doc-p"}
    assert len(calls) == 3
    assert calls[0] == ["original", "keyword_focused", "clarified"]
    assert calls[1] == ["original", "keyword_focused", "clarified"]
    assert "retry_keyword" in calls[2]


def test_precision_route_does_not_retry(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("retrieval-retry-precision")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        targeted_retry_enabled=True,
        targeted_retry_max_attempts=1,
        rerank_mode="off",
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    candidate = _make_candidate(chunk_id="chunk-a", doc_id="doc-a", block_type="paragraph")
    calls = {"count": 0}

    def fake_run_round(*, query_variants, decision, metadata_filter, targeted_plan):
        calls["count"] += 1
        return [candidate]

    monkeypatch.setattr(store, "_run_retrieval_round", fake_run_round)
    monkeypatch.setattr(store, "_rerank_candidates", lambda query, candidates, top_k: sorted(candidates, key=lambda item: -(item.fusion_score or 0.0))[:top_k], )
    monkeypatch.setattr(
        store,
        "_replace_child_hits_with_parent_content",
        lambda *, reranked_documents, top_k: [document for document, _ in reranked_documents[:top_k]],
    )

    results = store.similarity_search("StreamingResponse class path backend/src/api/main.py", k=1)

    assert len(results) == 1
    assert calls["count"] == 1


