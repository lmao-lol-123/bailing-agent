from __future__ import annotations

from backend.src.core.config import Settings
from backend.src.core.models import NormalizedDocument, SourceType
from backend.src.ingest.chunking import StructureAwareChunkingService


def test_chunking_preserves_metadata_from_structured_block(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=64, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-1",
                source_type=SourceType.MARKDOWN,
                source_name="guide.md",
                source_uri_or_path="guide.md",
                page_or_section="1",
                page="1",
                title="Guide",
                section_path=["Guide", "Install"],
                doc_type=SourceType.MARKDOWN,
                updated_at="2026-04-03T10:00:00+00:00",
                content="Install dependencies with pip.",
                metadata={"block_id": "block-install", "block_type": "paragraph", "layout_role": "body"},
            )
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == "Guide"
    assert chunks[0].metadata["section_path"] == ["Guide", "Install"]
    assert chunks[0].metadata["doc_type"] == "markdown"
    assert chunks[0].metadata["page"] == "1"
    assert chunks[0].metadata["updated_at"] == "2026-04-03T10:00:00+00:00"
    assert chunks[0].metadata["block_type"] == "paragraph"
    assert chunks[0].metadata["source_block_id"] == "block-install"

def test_chunking_splits_oversized_paragraphs_with_token_budget(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=24, chunk_overlap_word_pieces=4)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)
    long_paragraph = " ".join(f"token{i}" for i in range(80))

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-2",
                source_type=SourceType.TXT,
                source_name="notes.txt",
                source_uri_or_path="notes.txt",
                content=long_paragraph,
            )
        ]
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["chunk_wordpiece_count"] <= 24 for chunk in chunks)


def test_chunking_keeps_prelinearized_tables_lists_and_images(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-3",
                source_type=SourceType.MARKDOWN,
                source_name="deployment.md",
                source_uri_or_path="deployment.md",
                title="Deployment Guide",
                section_path=["Deployment Guide", "Runtime"],
                content="Table columns: Setting, Value\nRow 1: Setting=PORT; Value=8000",
                metadata={"block_id": "table-runtime", "block_type": "table", "layout_role": "body", "table_headers": ["Setting", "Value"], "table_row_count": 1},
            ),
            NormalizedDocument(
                doc_id="doc-3",
                source_type=SourceType.MARKDOWN,
                source_name="deployment.md",
                source_uri_or_path="deployment.md",
                title="Deployment Guide",
                section_path=["Deployment Guide", "Runtime"],
                content="Image block\nAlt: Architecture diagram\nSource: images/arch.png\nCaption: Request flow between API and retriever.",
                metadata={"block_id": "image-runtime", "block_type": "image", "layout_role": "body", "image_source": "images/arch.png", "image_alt_text": "Architecture diagram"},
            ),
            NormalizedDocument(
                doc_id="doc-3",
                source_type=SourceType.MARKDOWN,
                source_name="deployment.md",
                source_uri_or_path="deployment.md",
                title="Deployment Guide",
                section_path=["Deployment Guide", "Runtime"],
                content="Level 1: Install dependencies\nLevel 2: Activate venv",
                metadata={"block_id": "list-runtime", "block_type": "list", "layout_role": "body", "list_item_count": 2, "list_max_level": 2},
            ),
        ]
    )

    table_chunk = next(chunk for chunk in chunks if chunk.metadata["block_type"] == "table")
    image_chunk = next(chunk for chunk in chunks if chunk.metadata["block_type"] == "image")
    list_chunk = next(chunk for chunk in chunks if chunk.metadata["block_type"] == "list")

    assert "Table columns: Setting, Value" in table_chunk.content
    assert table_chunk.metadata["table_headers"] == ["Setting", "Value"]
    assert image_chunk.metadata["image_source"] == "images/arch.png"
    assert "Image block" in image_chunk.content
    assert "Level 2: Activate venv" in list_chunk.content
    assert list_chunk.metadata["list_max_level"] == 2

def test_chunking_uses_existing_section_path_without_heading_reparse(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-4",
                source_type=SourceType.MARKDOWN,
                source_name="ops.md",
                source_uri_or_path="ops.md",
                title="Ops",
                section_path=["Ops", "Deploy", "Rollback"],
                content="Rollback safely.",
                metadata={"block_id": "block-rollback", "block_type": "paragraph", "layout_role": "body"},
            )
        ]
    )

    assert chunks[0].metadata["section_path"] == ["Ops", "Deploy", "Rollback"]

def test_chunking_consumes_structured_block_metadata_without_reparsing(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-structured",
                source_type=SourceType.PDF,
                source_name="manual.pdf",
                source_uri_or_path="manual.pdf",
                page="3",
                title="Manual",
                section_path=["Manual", "Runtime"],
                content="Table columns: setting, value\nRow 1: PORT=8000",
                metadata={
                    "block_id": "block-table-1",
                    "block_type": "table",
                    "layout_role": "body",
                    "bbox": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                    "parser_source": "pymupdf4llm",
                    "graph_edges": [{"type": "has_caption", "target_block_id": "caption-1", "weight": 0.98}],
                    "graph_neighbors": ["caption-1"],
                },
            )
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Table columns: setting, value\nRow 1: PORT=8000"
    assert chunks[0].metadata["source_block_id"] == "block-table-1"
    assert chunks[0].metadata["source_block_type"] == "table"
    assert chunks[0].metadata["bbox"] == {"x0": 1, "y0": 2, "x1": 3, "y1": 4}
    assert chunks[0].metadata["graph_neighbors"] == ["caption-1"]


def test_chunking_skips_page_markers_and_noise_blocks(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-noise",
                source_type=SourceType.PDF,
                source_name="manual.pdf",
                source_uri_or_path="manual.pdf",
                content="1",
                metadata={"block_id": "page-1", "block_type": "page_marker", "layout_role": "page_marker"},
            ),
            NormalizedDocument(
                doc_id="doc-noise",
                source_type=SourceType.PDF,
                source_name="manual.pdf",
                source_uri_or_path="manual.pdf",
                content="Company Confidential",
                metadata={"block_id": "header-1", "block_type": "paragraph", "layout_role": "header"},
            ),
            NormalizedDocument(
                doc_id="doc-noise",
                source_type=SourceType.PDF,
                source_name="manual.pdf",
                source_uri_or_path="manual.pdf",
                content="Useful body text",
                metadata={"block_id": "body-1", "block_type": "paragraph", "layout_role": "body"},
            ),
            NormalizedDocument(
                doc_id="doc-noise",
                source_type=SourceType.PDF,
                source_name="manual.pdf",
                source_uri_or_path="manual.pdf",
                content="Footer",
                metadata={"block_id": "footer-1", "block_type": "paragraph", "excluded_from_body": True},
            ),
        ]
    )

    assert [chunk.metadata["source_block_id"] for chunk in chunks] == ["body-1"]


def test_chunking_parent_child_metadata_and_long_parent_storage(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=24, chunk_overlap_word_pieces=4)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)
    long_text = " ".join(f"token{i}" for i in range(80))

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-long",
                source_type=SourceType.TXT,
                source_name="notes.txt",
                source_uri_or_path="notes.txt",
                title="Notes",
                content=long_text,
                metadata={"block_id": "block-long", "block_type": "paragraph", "layout_role": "body"},
            )
        ]
    )

    assert len(chunks) > 1
    assert len({chunk.metadata["parent_chunk_id"] for chunk in chunks}) == 1
    assert all(chunk.metadata["chunk_level"] == "child" for chunk in chunks)
    assert all(chunk.metadata["parent_block_id"] == "block-long" for chunk in chunks)
    assert all(chunk.metadata["source_block_id"] == "block-long" for chunk in chunks)
    assert all(chunk.metadata["child_count"] == len(chunks) for chunk in chunks)
    assert [chunk.metadata["child_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["chunk_wordpiece_count"] <= 24 for chunk in chunks)
    assert all("parent_content" not in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["parent_content_hash"] for chunk in chunks)
    assert all(chunk.metadata["parent_wordpiece_count"] == 80 for chunk in chunks)
    assert chunks[0].metadata["parent_content_preview"].startswith("token0")


def test_chunking_fallbacks_missing_structural_fields(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-fallback",
                source_type=SourceType.TXT,
                source_name="notes.txt",
                source_uri_or_path="notes.txt",
                title="Notes",
                page="2",
                content="Fallback block text",
                metadata={},
            )
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["source_block_type"] == "paragraph"
    assert chunks[0].metadata["layout_role"] == "body"
    assert chunks[0].section_path == ["Notes"]
    assert chunks[0].metadata["source_block_id"].startswith("doc-fallback:2:")
    assert chunks[0].metadata["graph_edges"] == []
    assert chunks[0].metadata["graph_neighbors"] == []



def test_table_child_split_repeats_header_and_records_row_range(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=14, chunk_overlap_word_pieces=2)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)
    table_text = "Table columns: Metric, Value\n" + "\n".join(
        f"Row {index}: Metric=M{index}; Value={index}ms" for index in range(1, 5)
    )

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-table-split",
                source_type=SourceType.PDF,
                source_name="manual.pdf",
                source_uri_or_path="manual.pdf",
                title="Manual",
                section_path=["Manual", "Tables"],
                content=table_text,
                metadata={"block_id": "table-split", "block_type": "table", "layout_role": "body"},
            )
        ]
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["child_split_strategy"] == "table_rows" for chunk in chunks)
    assert all(chunk.metadata["table_header_repeated"] is True for chunk in chunks)
    assert all(chunk.metadata["child_content"].startswith("Table columns: Metric, Value") for chunk in chunks)
    assert chunks[0].metadata["table_row_start"] == 1
    assert chunks[-1].metadata["table_row_end"] == 4


def test_list_child_split_keeps_child_items_with_parent_item(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=16, chunk_overlap_word_pieces=2)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)
    list_text = "\n".join(
        [
            "Level 1: Install dependencies",
            "Level 2: Create venv",
            "Level 2: Install requirements",
            "Level 1: Run service",
            "Level 2: Start API",
        ]
    )

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-list-split",
                source_type=SourceType.MARKDOWN,
                source_name="guide.md",
                source_uri_or_path="guide.md",
                title="Guide",
                section_path=["Guide", "Setup"],
                content=list_text,
                metadata={"block_id": "list-split", "block_type": "list", "layout_role": "body"},
            )
        ]
    )

    assert len(chunks) == 2
    assert all(chunk.metadata["child_split_strategy"] == "list_items" for chunk in chunks)
    assert "Level 2: Install requirements" in chunks[0].metadata["child_content"]
    assert chunks[0].metadata["list_item_start"] == 1
    assert chunks[1].metadata["list_item_start"] == 2


def test_code_child_split_uses_function_units(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=20, chunk_overlap_word_pieces=2)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)
    code_text = "\n\n".join(
        [
            "def load_data():\n    return read_csv()",
            "def clean_data():\n    return normalize_rows()",
            "class Runner:\n    def run(self):\n        return clean_data()",
        ]
    )

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-code-split",
                source_type=SourceType.MARKDOWN,
                source_name="guide.md",
                source_uri_or_path="guide.md",
                title="Guide",
                section_path=["Guide", "Code"],
                content=code_text,
                metadata={"block_id": "code-split", "block_type": "code", "layout_role": "body"},
            )
        ]
    )

    assert len(chunks) == 3
    assert all(chunk.metadata["child_split_strategy"] == "code_units" for chunk in chunks)
    assert chunks[0].metadata["code_unit_start"] == 1
    assert chunks[-1].metadata["code_unit_end"] == 3
    assert all(chunk.metadata["chunk_wordpiece_count"] <= 20 for chunk in chunks)


def test_enhanced_retrieval_text_for_table_image_and_formula_keeps_original_child_content(fake_embeddings) -> None:
    settings = Settings(chunk_max_word_pieces=96, chunk_overlap_word_pieces=8)
    service = StructureAwareChunkingService(settings=settings, embeddings=fake_embeddings)

    chunks = service.chunk_documents(
        [
            NormalizedDocument(
                doc_id="doc-retrieval-text",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                title="Paper",
                section_path=["Paper", "Results"],
                content="Table columns: Metric, Value\nRow 1: Metric=Latency; Value=20ms",
                metadata={
                    "block_id": "table-rt",
                    "block_type": "table",
                    "layout_role": "body",
                    "caption_text": "Table 1. Latency metrics.",
                    "table_headers": ["Metric", "Value"],
                    "table_row_count": 1,
                },
            ),
            NormalizedDocument(
                doc_id="doc-retrieval-text",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                title="Paper",
                section_path=["Paper", "Architecture"],
                content="Image block\nSource: arch.png",
                metadata={
                    "block_id": "image-rt",
                    "block_type": "image",
                    "layout_role": "body",
                    "caption_text": "Figure 1. API request flow.",
                    "image_alt_text": "Architecture diagram",
                    "image_title": "Runtime architecture",
                },
            ),
            NormalizedDocument(
                doc_id="doc-retrieval-text",
                source_type=SourceType.PDF,
                source_name="paper.pdf",
                source_uri_or_path="paper.pdf",
                title="Paper",
                section_path=["Paper", "Math"],
                content="E = mc^2",
                metadata={
                    "block_id": "formula-rt",
                    "block_type": "formula",
                    "layout_role": "body",
                    "caption_text": "Equation 1. Energy mass equivalence.",
                    "formula_linearized_text": "E equals m c squared",
                    "formula_symbols": ["=", "^"],
                },
            ),
        ]
    )

    table_chunk = next(chunk for chunk in chunks if chunk.metadata["source_block_id"] == "table-rt")
    image_chunk = next(chunk for chunk in chunks if chunk.metadata["source_block_id"] == "image-rt")
    formula_chunk = next(chunk for chunk in chunks if chunk.metadata["source_block_id"] == "formula-rt")

    assert table_chunk.content != table_chunk.metadata["child_content"]
    assert "Table 1. Latency metrics." in table_chunk.content
    assert "Headers: Metric, Value" in table_chunk.content
    assert table_chunk.metadata["original_child_content"] == table_chunk.metadata["child_content"]
    assert image_chunk.content != image_chunk.metadata["child_content"]
    assert "Architecture diagram" in image_chunk.content
    assert "Figure 1. API request flow." in image_chunk.content
    assert formula_chunk.content != formula_chunk.metadata["child_content"]
    assert "E equals m c squared" in formula_chunk.content
    assert "Symbols: =, ^" in formula_chunk.content
    assert all(chunk.metadata["content_role"] == "retrieval_text" for chunk in [table_chunk, image_chunk, formula_chunk])
