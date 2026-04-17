from __future__ import annotations

from backend.src.ingest.cleaning import ParsedPage, StructuredContentCleaner


def test_cleaner_repairs_hyphenation_and_preserves_structure() -> None:
    cleaner = StructuredContentCleaner()
    cleaned = cleaner.clean_ingest_text(
        "# Guide\r\n\r\nengi-\n"
        "neering docs\r\n\r\n"
        "- item  one\r\n"
        "- item   two\r\n\r\n"
        "| A | B |\r\n"
        "| --- | --- |\r\n"
        "| 1 | 2 |\r\n"
    )

    assert "engineering docs" in cleaned
    assert "# Guide" in cleaned
    assert "- item one" in cleaned
    assert "| A | B |" in cleaned
    assert "\n\n" in cleaned


def test_cleaner_builds_page_markers_and_links_figures_tables() -> None:
    cleaner = StructuredContentCleaner()
    pages = [
        ParsedPage(
            page_number=1,
            page_label="1",
            text=(
                "1\n"
                "# Guide\n\n"
                "Figure 1. Request flow\n"
                '![Architecture](images/arch.png "System Architecture")\n\n'
                "As shown in Figure 1 and Table 2, the pipeline is stable.\n\n"
                "Table 2. Runtime config\n"
                "| Metric | Value |\n"
                "| --- | --- |\n"
                "| top_k | 4 |\n"
            ),
            metadata={"source_page_count": 1},
        )
    ]

    blocks = cleaner.build_blocks(pages=pages, title="Guide", include_page_markers=True)

    page_marker = next(block for block in blocks if block.block_type == "page_marker")
    image_block = next(block for block in blocks if block.block_type == "image")
    table_block = next(block for block in blocks if block.block_type == "table")
    paragraph_block = next(block for block in blocks if block.block_type == "paragraph")

    assert page_marker.text == "1"
    assert image_block.metadata["caption_type"] == "figure"
    assert image_block.metadata["caption_text"] == "Figure 1. Request flow"
    assert table_block.metadata["caption_type"] == "table"
    assert table_block.metadata["caption_text"] == "Table 2. Runtime config"
    assert table_block.metadata["table_headers"] == ["Metric", "Value"]
    assert "Row 1: Metric=top_k; Value=4" in table_block.text
    assert paragraph_block.metadata["references_figures"] == ["1"]
    assert paragraph_block.metadata["references_tables"] == ["2"]
    assert image_block.block_id in paragraph_block.metadata["related_block_ids"]
    assert table_block.block_id in paragraph_block.metadata["related_block_ids"]
