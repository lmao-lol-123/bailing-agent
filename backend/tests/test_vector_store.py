from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import IngestedChunk, RetrievalFilter, SourceType
from backend.src.ingest.parent_store import JsonParentStore, ParentRecord
from backend.src.retrieve.index_manager import IndexManager
from backend.src.retrieve.rerank import RetrievedCandidate
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
    doc_type: SourceType = SourceType.MARKDOWN,
    source_name: str = "guide.md",
    source_uri_or_path: str = "guide.md",
    page: str = "1",
    section_path: list[str] | None = None,
    parent_chunk_id: str | None = None,
) -> IngestedChunk:
    resolved_section_path = section_path or ["Guide", "Streaming"]
    metadata = {
        "doc_id": doc_id,
        "source_name": source_name,
        "source_type": doc_type.value,
        "source_uri_or_path": source_uri_or_path,
        "page_or_section": page,
        "title": "Guide",
        "section_path": resolved_section_path,
        "page": page,
        "doc_type": doc_type.value,
        "updated_at": "2026-04-03T10:00:00+00:00",
        "chunk_id": chunk_id,
        "chunk_level": "child",
        "source_block_id": f"block-{chunk_id}",
        "source_block_type": "paragraph",
        "child_content": child_content,
    }
    if parent_chunk_id:
        metadata["parent_chunk_id"] = parent_chunk_id
        metadata["parent_store_ref"] = f"{doc_id}:{parent_chunk_id}"
        metadata["parent_content_hash"] = f"hash-{parent_chunk_id}"
        metadata["parent_source_block_ids"] = [f"block-{chunk_id}"]
    return IngestedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_name=source_name,
        source_uri_or_path=source_uri_or_path,
        page_or_section=page,
        page=page,
        title="Guide",
        section_path=resolved_section_path,
        doc_type=doc_type,
        content=content,
        metadata=metadata,
    )


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
            make_chunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                content="FastAPI streams answers using SSE.",
                child_content="FastAPI streams answers using SSE.",
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
        hybrid_lexical_enabled=False,
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
            "chunk_id": "chunk-markdown",
        },
        id="chunk-markdown",
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
            "chunk_id": "chunk-txt",
        },
        id="chunk-txt",
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
            "chunk_id": "chunk-weak",
        },
        id="chunk-weak",
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


def test_vector_store_replaces_child_hit_with_parent_content_and_deduplicates(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-parent-child")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        rerank_candidate_multiplier=4,
        hybrid_lexical_enabled=False,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    child_one = Document(
        page_content="child content one FastAPI",
        metadata={
            "doc_id": "doc-parent",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide"],
            "doc_type": "markdown",
            "page": "1",
            "chunk_id": "child-1",
            "chunk_level": "child",
            "parent_chunk_id": "parent-1",
            "parent_content": "full parent content with FastAPI and SSE details",
            "child_content": "child content one FastAPI",
        },
        id="child-1",
    )
    child_two = Document(
        page_content="child content two SSE FastAPI text event stream",
        metadata={
            "doc_id": "doc-parent",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide"],
            "doc_type": "markdown",
            "page": "1",
            "chunk_id": "child-2",
            "chunk_level": "child",
            "parent_chunk_id": "parent-1",
            "parent_content": "full parent content with FastAPI and SSE details",
            "child_content": "child content two SSE FastAPI text event stream",
        },
        id="child-2",
    )

    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(child_one, 0.4), (child_two, 0.1)],
    )

    results = store.similarity_search("FastAPI SSE text event stream", k=2)

    assert len(results) == 1
    assert results[0].page_content == "full parent content with FastAPI and SSE details"
    assert results[0].metadata["matched_child_ids"] == ["child-1", "child-2"]
    assert results[0].metadata["matched_child_count"] == 2
    assert results[0].metadata["child_content"] == "child content two SSE FastAPI text event stream"
    assert results[0].metadata["parent_content_source"] == "metadata"
    assert results[0].metadata["parent_hydrated"] is True


def test_vector_store_hydrates_parent_from_parent_store(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-parent-store")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        hybrid_lexical_enabled=False,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    parent_store = JsonParentStore(settings.processed_directory)
    parent_store.save_records(
        "doc-parent",
        [
            ParentRecord(
                parent_chunk_id="parent-1",
                doc_id="doc-parent",
                source_block_ids=["block-1"],
                parent_content="hydrated parent content from store",
                parent_wordpiece_count=6,
                parent_content_hash="hash-parent-1",
                section_path=["Guide"],
            )
        ],
    )

    child_document = Document(
        page_content="retrieval text for child",
        metadata={
            "doc_id": "doc-parent",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide"],
            "doc_type": "markdown",
            "page": "1",
            "chunk_id": "child-1",
            "chunk_level": "child",
            "parent_chunk_id": "parent-1",
            "parent_store_ref": "doc-parent:parent-1",
            "child_content": "child fallback text",
        },
        id="child-1",
    )

    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(child_document, 0.1)],
    )

    results = store.similarity_search("hydrated parent", k=1)

    assert len(results) == 1
    assert results[0].page_content == "hydrated parent content from store"
    assert results[0].metadata["parent_content_source"] == "store"
    assert results[0].metadata["parent_hydrated"] is True


def test_vector_store_falls_back_to_child_content_when_parent_store_missing(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-parent-store-missing")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        hybrid_lexical_enabled=False,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    child_document = Document(
        page_content="retrieval text for child",
        metadata={
            "doc_id": "doc-parent",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide"],
            "doc_type": "markdown",
            "page": "1",
            "chunk_id": "child-1",
            "chunk_level": "child",
            "parent_chunk_id": "parent-1",
            "parent_store_ref": "doc-parent:parent-1",
            "child_content": "child fallback text",
        },
        id="child-1",
    )

    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(child_document, 0.1)],
    )

    results = store.similarity_search("hydrated parent", k=1)

    assert len(results) == 1
    assert results[0].page_content == "child fallback text"
    assert results[0].metadata["parent_content_source"] == "child_fallback"
    assert results[0].metadata["parent_hydrated"] is False


def test_vector_store_hybrid_retrieval_merges_dense_and_lexical_sources(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-hybrid")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
        hybrid_dense_k=5,
        hybrid_lexical_k=5,
        hybrid_fusion_top_k=5,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    index_manager = IndexManager(settings=settings, embeddings=fake_embeddings, vector_store=store)
    store.set_index_manager(index_manager)

    chunk = make_chunk(
        doc_id="doc-hybrid",
        chunk_id="chunk-hybrid",
        content="FastAPI text event stream SSE guide",
        child_content="FastAPI text event stream SSE guide",
    )
    index_manager.index_chunks([chunk])

    dense_document = Document(page_content=chunk.content, metadata=dict(chunk.metadata), id=chunk.chunk_id)
    lexical_candidate = RetrievedCandidate(
        chunk_id=chunk.chunk_id,
        document=Document(page_content=chunk.content, metadata=dict(chunk.metadata), id=chunk.chunk_id),
        retrieval_sources=("lexical",),
        lexical_rank=1,
        lexical_score=1.0,
        fusion_score=1.0,
    )
    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(dense_document, 0.1)],
    )
    monkeypatch.setattr(
        store._lexical_retrieval,
        "search",
        lambda query, k, metadata_filter=None, record_matcher=None: [lexical_candidate],
    )

    results = store.similarity_search("FastAPI text event stream", k=1)

    assert len(results) == 1
    assert results[0].metadata["doc_id"] == "doc-hybrid"
    assert results[0].metadata["retrieval_sources"] == ["dense", "lexical"]


def test_vector_store_hybrid_retrieval_uses_lexical_candidates_with_metadata_filters(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-hybrid-filter")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
        hybrid_dense_k=5,
        hybrid_lexical_k=5,
        hybrid_fusion_top_k=5,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)
    index_manager = IndexManager(settings=settings, embeddings=fake_embeddings, vector_store=store)
    store.set_index_manager(index_manager)

    markdown_chunk = make_chunk(
        doc_id="doc-markdown",
        chunk_id="chunk-markdown",
        content="FastAPI response media type should be text event stream",
        child_content="FastAPI response media type should be text event stream",
        doc_type=SourceType.MARKDOWN,
    )
    txt_chunk = make_chunk(
        doc_id="doc-txt",
        chunk_id="chunk-txt",
        content="FastAPI response media type should be text event stream",
        child_content="FastAPI response media type should be text event stream",
        doc_type=SourceType.TXT,
        source_name="notes.txt",
        source_uri_or_path="notes.txt",
    )
    index_manager.index_chunks([markdown_chunk])
    index_manager.index_chunks([txt_chunk])

    monkeypatch.setattr(store._vector_store, "similarity_search_with_score", lambda query, k, **kwargs: [])
    markdown_candidate = RetrievedCandidate(
        chunk_id=markdown_chunk.chunk_id,
        document=Document(page_content=markdown_chunk.content, metadata=dict(markdown_chunk.metadata), id=markdown_chunk.chunk_id),
        retrieval_sources=("lexical",),
        lexical_rank=1,
        lexical_score=1.0,
        fusion_score=1.0,
    )
    txt_candidate = RetrievedCandidate(
        chunk_id=txt_chunk.chunk_id,
        document=Document(page_content=txt_chunk.content, metadata=dict(txt_chunk.metadata), id=txt_chunk.chunk_id),
        retrieval_sources=("lexical",),
        lexical_rank=2,
        lexical_score=0.9,
        fusion_score=0.9,
    )

    def fake_lexical_search(query, k, metadata_filter=None, record_matcher=None):
        candidates = [markdown_candidate, txt_candidate]
        return [
            candidate
            for candidate in candidates
            if store._matches_metadata_filter(candidate.document.metadata, metadata_filter)
        ][:k]

    monkeypatch.setattr(store._lexical_retrieval, "search", fake_lexical_search)

    results = store.similarity_search(
        "text event stream",
        k=2,
        metadata_filter=RetrievalFilter(doc_types=[SourceType.MARKDOWN]),
    )

    assert [document.metadata["doc_id"] for document in results] == ["doc-markdown"]
    assert "lexical" in results[0].metadata["retrieval_sources"]
def test_vector_store_applies_precision_route_metadata(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-route-precision")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        hybrid_lexical_enabled=False,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    precision_doc = Document(
        page_content="StreamingResponse is defined in backend/src/api/main.py",
        metadata={
            "doc_id": "doc-precision",
            "source_name": "guide.md",
            "source_type": "markdown",
            "source_uri_or_path": "guide.md",
            "section_path": ["Guide", "Streaming"],
            "doc_type": "markdown",
            "page": "1",
            "chunk_id": "chunk-precision",
        },
        id="chunk-precision",
    )

    monkeypatch.setattr(
        store._vector_store,
        "similarity_search_with_score",
        lambda query, k, **kwargs: [(precision_doc, 0.1)],
    )

    results = store.similarity_search("StreamingResponse class path backend/src/api/main.py", k=1)

    assert len(results) == 1
    assert results[0].metadata["route_name"] == "precision"
    assert results[0].metadata["query_variant_ids"] == ["original"]
    assert results[0].metadata["matched_query_count"] == 1


def test_vector_store_fuses_multi_query_candidates_across_variants(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-query-fusion")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        query_multi_enabled=True,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    candidate_shared = RetrievedCandidate(
        chunk_id="chunk-a",
        document=Document(
            page_content="Figure 2 latency tradeoff explanation",
            metadata={
                "doc_id": "doc-a",
                "source_name": "guide.md",
                "source_uri_or_path": "guide.md",
                "source_type": "markdown",
                "doc_type": "markdown",
                "page": "1",
                "section_path": ["Guide", "Design"],
                "chunk_id": "chunk-a",
                "source_block_type": "caption",
            },
            id="chunk-a",
        ),
        retrieval_sources=("dense",),
        dense_rank=1,
        dense_score=0.9,
        fusion_score=0.9,
    )
    candidate_other = RetrievedCandidate(
        chunk_id="chunk-b",
        document=Document(
            page_content="Generic implementation notes",
            metadata={
                "doc_id": "doc-b",
                "source_name": "notes.md",
                "source_uri_or_path": "notes.md",
                "source_type": "markdown",
                "doc_type": "markdown",
                "page": "2",
                "section_path": ["Notes"],
                "chunk_id": "chunk-b",
                "source_block_type": "paragraph",
            },
            id="chunk-b",
        ),
        retrieval_sources=("lexical",),
        lexical_rank=1,
        lexical_score=0.8,
        fusion_score=0.8,
    )

    def fake_retrieve_query_variant(*, variant, decision, metadata_filter, targeted_plan):
        if variant.variant_id in {"original", "keyword_focused"}:
            return [candidate_shared, candidate_other]
        return [candidate_shared]

    monkeypatch.setattr(store, "_retrieve_query_variant", fake_retrieve_query_variant)
    monkeypatch.setattr(store, "_rerank_candidates", lambda query, candidates, top_k: candidates[:top_k])

    results = store.similarity_search("Explain Figure 2 latency tradeoff", k=2)

    assert [document.metadata["doc_id"] for document in results] == ["doc-a", "doc-b"]
    assert results[0].metadata["route_name"] == "explained_structure"
    assert results[0].metadata["matched_query_count"] >= 2
    assert set(results[0].metadata["query_variant_ids"]) >= {"original", "keyword_focused"}


def test_vector_store_soft_targeting_bias_prefers_target_block(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-soft-targeting")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        hybrid_lexical_enabled=False,
        rerank_mode="off",
        targeted_hard_min_candidates=2,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    table_doc = Document(page_content="Figure 2 table result", metadata={"doc_id": "doc-table", "source_name": "guide-table.md", "source_uri_or_path": "guide-table.md", "source_type": "markdown", "doc_type": "markdown", "page": "1", "section_path": ["Guide"], "chunk_id": "chunk-table", "source_block_type": "table"}, id="chunk-table")
    paragraph_doc = Document(page_content="Figure 2 is discussed here", metadata={"doc_id": "doc-paragraph", "source_name": "guide-paragraph.md", "source_uri_or_path": "guide-paragraph.md", "source_type": "markdown", "doc_type": "markdown", "page": "2", "section_path": ["Guide"], "chunk_id": "chunk-paragraph", "source_block_type": "paragraph"}, id="chunk-paragraph")

    monkeypatch.setattr(store._vector_store, "similarity_search_with_score", lambda query, k, **kwargs: [(paragraph_doc, 0.1), (table_doc, 0.1)])

    results = store.similarity_search("Figure 2 on page 12", k=2)

    assert [document.metadata["doc_id"] for document in results] == ["doc-table", "doc-paragraph"]
    assert results[0].metadata["targeting_mode"] == "soft_fallback"
    assert results[0].metadata["target_block_hit"] is True


def test_vector_store_hard_targeting_keeps_target_blocks(monkeypatch, fake_embeddings) -> None:
    case_dir = make_case_dir("vector-store-hard-targeting")
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        hybrid_lexical_enabled=False,
        rerank_mode="off",
        targeted_hard_min_candidates=2,
        targeted_retry_enabled=False,
    )
    settings.ensure_directories()
    store = VectorStoreService(settings=settings, embeddings=fake_embeddings)

    table_doc_a = Document(page_content="Figure 2 row 1", metadata={"doc_id": "doc-table-a", "source_name": "guide-table-a.md", "source_uri_or_path": "guide-table-a.md", "source_type": "markdown", "doc_type": "markdown", "page": "1", "section_path": ["Guide"], "chunk_id": "chunk-table-a", "source_block_type": "table"}, id="chunk-table-a")
    table_doc_b = Document(page_content="Figure 2 row 2", metadata={"doc_id": "doc-table-b", "source_name": "guide-table-b.md", "source_uri_or_path": "guide-table-b.md", "source_type": "markdown", "doc_type": "markdown", "page": "2", "section_path": ["Guide"], "chunk_id": "chunk-table-b", "source_block_type": "table"}, id="chunk-table-b")
    paragraph_doc = Document(page_content="Figure 2 explanation", metadata={"doc_id": "doc-paragraph", "source_name": "guide-paragraph.md", "source_uri_or_path": "guide-paragraph.md", "source_type": "markdown", "doc_type": "markdown", "page": "3", "section_path": ["Guide"], "chunk_id": "chunk-paragraph", "source_block_type": "paragraph"}, id="chunk-paragraph")

    monkeypatch.setattr(store._vector_store, "similarity_search_with_score", lambda query, k, **kwargs: [(paragraph_doc, 0.1), (table_doc_a, 0.1), (table_doc_b, 0.1)])

    results = store.similarity_search("Figure 2 on page 12", k=3)

    assert [document.metadata["doc_id"] for document in results] == ["doc-table-a", "doc-table-b"]
    assert all(document.metadata["targeting_mode"] == "hard" for document in results)






