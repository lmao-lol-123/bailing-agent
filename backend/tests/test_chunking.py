from __future__ import annotations

from backend.src.core.config import Settings
from backend.src.core.models import NormalizedDocument, SourceType
from backend.src.ingest.chunking import StructureAwareChunkingService


def test_chunking_preserves_heading_structure_and_metadata(fake_embeddings) -> None:
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
                doc_type=SourceType.MARKDOWN,
                updated_at="2026-04-03T10:00:00+00:00",
                content=(
                    "# Guide\n\n"
                    "Overview paragraph.\n\n"
                    "## Install\n\n"
                    "- Step one\n"
                    "- Step two\n\n"
                    "```python\n"
                    "print('hello')\n"
                    "```\n"
                ),
            )
        ]
    )

    assert len(chunks) >= 3
    assert chunks[0].metadata["title"] == "Guide"
    assert chunks[0].metadata["section_path"] == ["Guide"]
    assert chunks[0].metadata["doc_type"] == "markdown"
    assert chunks[0].metadata["page"] == "1"
    assert chunks[0].metadata["updated_at"] == "2026-04-03T10:00:00+00:00"
    assert any(chunk.metadata["section_path"] == ["Guide", "Install"] for chunk in chunks)
    assert any(chunk.metadata["block_type"] == "code" for chunk in chunks)


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


def test_chunking_linearizes_tables_lists_and_images(fake_embeddings) -> None:
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
                section_path=["Deployment Guide"],
                content=(
                    "# Deployment Guide\n\n"
                    "## Runtime\n\n"
                    "| Setting | Value |\n"
                    "| --- | --- |\n"
                    "| PORT | 8000 |\n"
                    "| HOST | 0.0.0.0 |\n\n"
                    "![Architecture diagram](images/arch.png \"System Architecture\")\n"
                    "Figure: Request flow between API and retriever.\n\n"
                    "- Install dependencies\n"
                    "  - Activate venv\n"
                    "- Run server\n"
                ),
            )
        ]
    )

    table_chunk = next(chunk for chunk in chunks if chunk.metadata["block_type"] == "table")
    image_chunk = next(chunk for chunk in chunks if chunk.metadata["block_type"] == "image")
    list_chunk = next(chunk for chunk in chunks if chunk.metadata["block_type"] == "list")

    assert "Table columns: Setting, Value" in table_chunk.content
    assert "Row 1: Setting=PORT; Value=8000" in table_chunk.content
    assert table_chunk.metadata["table_headers"] == ["Setting", "Value"]
    assert table_chunk.metadata["table_row_count"] == 2

    assert image_chunk.metadata["image_source"] == "images/arch.png"
    assert image_chunk.metadata["image_alt_text"] == "Architecture diagram"
    assert image_chunk.metadata["image_title"] == "System Architecture"
    assert image_chunk.metadata["image_caption"] == "Request flow between API and retriever."
    assert "Image block" in image_chunk.content
    assert "Caption: Request flow between API and retriever." in image_chunk.content

    assert "Level 1: Install dependencies" in list_chunk.content
    assert "Level 2: Activate venv" in list_chunk.content
    assert list_chunk.metadata["list_item_count"] == 3
    assert list_chunk.metadata["list_max_level"] == 2


def test_chunking_tracks_nested_heading_hierarchy(fake_embeddings) -> None:
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
                section_path=["Ops"],
                content=(
                    "# Ops\n\n"
                    "## Deploy\n\n"
                    "Intro.\n\n"
                    "### Rollback\n\n"
                    "Rollback safely.\n"
                ),
            )
        ]
    )

    assert any(chunk.metadata["section_path"] == ["Ops", "Deploy"] for chunk in chunks)
    assert any(chunk.metadata["section_path"] == ["Ops", "Deploy", "Rollback"] for chunk in chunks)
