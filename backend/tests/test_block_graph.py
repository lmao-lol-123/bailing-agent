from __future__ import annotations

from backend.src.core.models import SourceType
from backend.src.ingest.cleaning import ParsedLayoutItem, ParsedPage, StructuredContentCleaner


def test_block_graph_links_formula_caption_references_and_neighbors() -> None:
    cleaner = StructuredContentCleaner()
    page = ParsedPage(
        page_number=1,
        page_label="1",
        text="1\n# Guide\nEquation 1. Energy formula\n$$ E = mc^2 $$\nEquation 1 explains the energy model.",
        width=800,
        height=1000,
        layout_items=[
            ParsedLayoutItem(
                item_type="heading",
                text="# Guide",
                bbox={"x0": 80, "y0": 60, "x1": 260, "y1": 90},
                order=0,
            ),
            ParsedLayoutItem(
                item_type="caption",
                text="Equation 1. Energy formula",
                bbox={"x0": 90, "y0": 120, "x1": 320, "y1": 150},
                order=2,
            ),
            ParsedLayoutItem(
                item_type="formula",
                text="$$ E = mc^2 $$",
                bbox={"x0": 90, "y0": 160, "x1": 360, "y1": 220},
                order=1,
            ),
            ParsedLayoutItem(
                item_type="paragraph",
                text="Equation 1 explains the energy model.",
                bbox={"x0": 90, "y0": 240, "x1": 520, "y1": 290},
                order=3,
            ),
        ],
        metadata={"source_page_count": 1},
        parser_source="pymupdf4llm",
    )

    document = cleaner.build_document_from_pages(
        doc_id="doc-graph",
        source_type=SourceType.PDF,
        title="Guide",
        pages=[page],
        include_page_markers=True,
    )

    formula_block = next(block for block in document.blocks if block.block_type == "formula")
    paragraph_block = next(block for block in document.blocks if block.block_type == "paragraph")
    caption_block = next(block for block in document.blocks if block.block_type == "caption")

    assert formula_block.metadata["caption_text"] == "Equation 1. Energy formula"
    assert paragraph_block.metadata["references_formulas"] == ["1"]
    assert any(
        edge["type"] == "formula_explains" and edge["target_block_id"] == formula_block.block_id
        for edge in paragraph_block.metadata["graph_edges"]
    )
    assert any(
        edge["type"] == "caption_of" and edge["target_block_id"] == formula_block.block_id
        for edge in caption_block.metadata["graph_edges"]
    )
    assert formula_block.block_id in paragraph_block.metadata["graph_neighbors"]


def test_non_pdf_structuring_covers_html_csv_and_json() -> None:
    cleaner = StructuredContentCleaner()

    html_document, _ = cleaner.build_document_from_non_pdf(
        doc_id="doc-html",
        source_type=SourceType.WEB,
        title="Web Guide",
        text="<html><body><h1>Guide</h1><p>Figure 1 shows the flow.</p><img src='arch.png' alt='Architecture'/><figcaption>Figure 1. Request flow</figcaption><math>E = mc^2</math></body></html>",
    )
    csv_document, _ = cleaner.build_document_from_non_pdf(
        doc_id="doc-csv",
        source_type=SourceType.CSV,
        title="Metrics",
        text="metric,value\ntop_k,4\n",
    )
    json_document, _ = cleaner.build_document_from_non_pdf(
        doc_id="doc-json",
        source_type=SourceType.JSON,
        title="Config",
        text='[{"name":"top_k","value":4}]',
    )

    assert {block.block_type for block in html_document.blocks} >= {
        "section_header",
        "paragraph",
        "image",
        "caption",
        "formula",
    }
    assert any(
        block.block_type == "table" and "row 1: metric=top_k; value=4" in block.text.lower()
        for block in csv_document.blocks
    )
    assert any(
        block.block_type == "table" and block.metadata.get("json_table") is True
        for block in json_document.blocks
    )
