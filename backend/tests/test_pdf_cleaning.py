from __future__ import annotations

from backend.src.core.models import SourceType
from backend.src.ingest.cleaning import ParsedLayoutItem, ParsedPage, StructuredContentCleaner


def test_pdf_layout_items_use_bbox_reading_order_and_noise_roles() -> None:
    cleaner = StructuredContentCleaner()
    pages = [
        ParsedPage(
            page_number=1,
            page_label="1",
            text="1\nAcme Internal\nLeft intro\nRight intro\nBody detail\nConfidential",
            width=800,
            height=1000,
            layout_items=[
                ParsedLayoutItem(item_type="paragraph", text="Body detail", bbox={"x0": 90, "y0": 210, "x1": 340, "y1": 250}, order=4),
                ParsedLayoutItem(item_type="paragraph", text="Acme Internal", bbox={"x0": 120, "y0": 18, "x1": 300, "y1": 40}, order=9),
                ParsedLayoutItem(item_type="paragraph", text="Right intro", bbox={"x0": 430, "y0": 122, "x1": 650, "y1": 160}, order=1),
                ParsedLayoutItem(item_type="paragraph", text="Confidential", bbox={"x0": 320, "y0": 940, "x1": 470, "y1": 970}, order=2),
                ParsedLayoutItem(item_type="paragraph", text="Left intro", bbox={"x0": 90, "y0": 120, "x1": 320, "y1": 160}, order=7),
            ],
            metadata={"source_page_count": 2},
            parser_source="pymupdf4llm",
        ),
        ParsedPage(
            page_number=2,
            page_label="2",
            text="2\nAcme Internal\nContinuation\nConfidential",
            width=800,
            height=1000,
            layout_items=[
                ParsedLayoutItem(item_type="paragraph", text="Acme Internal", bbox={"x0": 120, "y0": 18, "x1": 300, "y1": 40}, order=3),
                ParsedLayoutItem(item_type="paragraph", text="Continuation", bbox={"x0": 90, "y0": 120, "x1": 320, "y1": 170}, order=4),
                ParsedLayoutItem(item_type="paragraph", text="Confidential", bbox={"x0": 320, "y0": 940, "x1": 470, "y1": 970}, order=1),
            ],
            metadata={"source_page_count": 2},
            parser_source="pymupdf4llm",
        ),
    ]

    document = cleaner.build_document_from_pages(
        doc_id="doc-pdf-layout",
        source_type=SourceType.PDF,
        title="Guide",
        pages=pages,
        include_page_markers=True,
    )

    body_texts = [
        block.text
        for block in document.blocks
        if block.page_number == 1 and block.block_type == "paragraph" and block.layout_role == "body"
    ]
    header_block = next(block for block in document.blocks if block.text == "Acme Internal")
    footer_block = next(block for block in document.blocks if block.text == "Confidential")

    assert body_texts == ["Left intro", "Right intro", "Body detail"]
    assert any(block.block_type == "page_marker" and block.text == "1" for block in document.blocks)
    assert header_block.layout_role == "header"
    assert header_block.metadata["excluded_from_body"] is True
    assert footer_block.layout_role == "footer"
    assert footer_block.metadata["excluded_from_body"] is True
