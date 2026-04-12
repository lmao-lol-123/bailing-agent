from __future__ import annotations

from backend.src.core.config import Settings
from backend.src.core.models import NormalizedDocument, SourceType
from backend.src.ingest.chunking import StructureAwareChunkingService


def test_short_parent_content_is_stored_once_per_child(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=64, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)
    content = "Short parent block for retrieval."

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-short-parent",
                source_type=SourceType.TXT,
                source_name="notes.txt",
                source_uri_or_path="notes.txt",
                title="Notes",
                content=content,
                metadata={"block_id": "block-short", "block_type": "paragraph", "layout_role": "body"},
            )
        ]
    )

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["chunk_level"] == "child"
    assert metadata["parent_content"] == content
    assert metadata["child_content"] == content
    assert metadata["returns_parent_content"] is True
    assert metadata["child_index"] == 0
    assert metadata["child_count"] == 1
    assert metadata["parent_block_id"] == "block-short"
    assert metadata["source_block_type"] == "paragraph"


def test_parent_assembler_adds_caption_relation_without_using_same_page_near(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-figure-parent",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                page="2",
                title="Paper",
                section_path=["Paper", "Architecture"],
                content="Image block\nAlt: request flow",
                metadata={
                    "block_id": "image-1",
                    "block_type": "image",
                    "layout_role": "body",
                    "block_order": 1,
                    "graph_edges": [
                        {"type": "has_caption", "target_block_id": "caption-1", "weight": 0.98},
                        {"type": "same_page_near", "target_block_id": "near-1", "weight": 0.74},
                    ],
                },
            ),
            NormalizedDocument(
                doc_id="doc-figure-parent",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                page="2",
                title="Paper",
                section_path=["Paper", "Architecture"],
                content="Figure 1. Request flow between API and retriever.",
                metadata={"block_id": "caption-1", "block_type": "caption", "layout_role": "caption", "block_order": 2},
            ),
            NormalizedDocument(
                doc_id="doc-figure-parent",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                page="2",
                title="Paper",
                section_path=["Paper", "Architecture"],
                content="Nearby but unrelated body text.",
                metadata={"block_id": "near-1", "block_type": "paragraph", "layout_role": "body", "block_order": 3},
            ),
        ]
    )

    image_chunk = next(chunk for chunk in chunks if chunk.metadata["source_block_id"] == "image-1")
    assert image_chunk.metadata["parent_assembly_strategy"] == "relation_limited"
    assert image_chunk.metadata["assembled_parent"] is True
    assert image_chunk.metadata["parent_source_block_ids"] == ["image-1", "caption-1"]
    assert image_chunk.metadata["parent_relation_types"] == ["has_caption"]
    assert "Figure 1. Request flow" in image_chunk.metadata["parent_content"]
    assert "Nearby but unrelated" not in image_chunk.metadata["parent_content"]


def test_parent_assembler_references_related_table_and_skips_noise(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=128, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-table-ref",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                page="4",
                title="Paper",
                section_path=["Paper", "Results"],
                content="Table 2 shows latency improvements.",
                metadata={
                    "block_id": "para-1",
                    "block_type": "paragraph",
                    "layout_role": "body",
                    "block_order": 1,
                    "graph_edges": [
                        {"type": "references", "target_block_id": "table-2", "weight": 0.9},
                        {"type": "references", "target_block_id": "header-1", "weight": 0.9},
                    ],
                },
            ),
            NormalizedDocument(
                doc_id="doc-table-ref",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                page="4",
                title="Paper",
                section_path=["Paper", "Results"],
                content="Table columns: Metric, Value\nRow 1: Metric=Latency; Value=20ms",
                metadata={"block_id": "table-2", "block_type": "table", "layout_role": "body", "block_order": 2},
            ),
            NormalizedDocument(
                doc_id="doc-table-ref",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                page="4",
                title="Paper",
                section_path=["Paper", "Results"],
                content="Repeated header",
                metadata={"block_id": "header-1", "block_type": "paragraph", "layout_role": "header", "block_order": 0},
            ),
        ]
    )

    paragraph_chunk = next(chunk for chunk in chunks if chunk.metadata["source_block_id"] == "para-1")
    assert paragraph_chunk.metadata["parent_source_block_ids"] == ["para-1", "table-2"]
    assert "Metric=Latency" in paragraph_chunk.metadata["parent_content"]
    assert "Repeated header" not in paragraph_chunk.metadata["parent_content"]
