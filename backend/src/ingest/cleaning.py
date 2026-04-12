from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.src.core.models import SourceType

_LINE_SPACES_RE = re.compile(r" {2,}")
_HYPHENATED_LINEBREAK_RE = re.compile(r"(?<=\w)-\n(?=\w)")
_HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S+")
_LIST_ITEM_CAPTURE_PATTERN = re.compile(r"^(?P<indent>\s*)(?:[-*+]|\d+[.)])\s+(?P<text>\S.*)$")
_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW_PATTERN = re.compile(r"^\|.*\|$")
_TABLE_ALIGNMENT_PATTERN = re.compile(r"^:?-{3,}:?$")
_PAGE_MARKER_PATTERN = re.compile(r"^(?:\d+|page\s+\d+|\d+\s*/\s*\d+)$", re.IGNORECASE)
_MD_IMAGE_PATTERN = re.compile(
    r'^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+[\"\'](?P<title>.*?)[\"\'])?\)$'
)
_HTML_IMG_PATTERN = re.compile(r"^<img\s+(?P<attrs>[^>]+?)\s*/?>$", re.IGNORECASE)
_HTML_ATTR_PATTERN = re.compile(r'(?P<name>[a-zA-Z_:][a-zA-Z0-9:._-]*)\s*=\s*[\"\'](?P<value>.*?)[\"\']')
_CAPTION_PATTERN = re.compile(
    r"^(?:(?:caption:\s*)?(?P<kind>figure|fig\.|table|image|equation|formula|eq\.|图|图片|表|公式)\s*(?P<number>\d+)?)\s*[:.\-]\s*(?P<caption>.+)$",
    re.IGNORECASE,
)
_REFERENCE_PATTERN = re.compile(
    r"(?P<kind>figure|fig\.|table|equation|formula|eq\.|图|表|公式)\s*(?P<number>\d+)",
    re.IGNORECASE,
)
_FORMULA_BLOCK_START_PATTERN = re.compile(r"^(\$\$|\\\[)")
_FORMULA_BLOCK_END_PATTERN = re.compile(r"^(\$\$|\\\])")
_FORMULA_SYMBOL_PATTERN = re.compile(r"(\\sum|\\int|\\frac|\\sqrt|<=|>=|==|!=|=|\^|_|\+|-)")
_WORD_PATTERN = re.compile(r"[A-Za-z\u4e00-\u9fff]+")
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "code", "ul", "ol", "table", "img", "figure", "figcaption", "math"}
_RELATION_PRIORITY = {
    "section_parent": 0.95,
    "section_child": 0.95,
    "caption_of": 0.98,
    "has_caption": 0.98,
    "references": 0.9,
    "referenced_by": 0.9,
    "formula_explains": 0.88,
    "table_row_context": 0.86,
    "continued_from": 0.8,
    "same_page_near": 0.74,
    "next_block": 0.72,
    "prev_block": 0.72,
}


@dataclass
class ParsedLayoutItem:
    item_type: str | None
    text: str
    bbox: dict[str, float] | None = None
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_block_id: str | None = None
    parser_source: str | None = None


@dataclass
class StructuredPage:
    page_number: int
    page_label: str | None
    text: str
    width: float | None = None
    height: float | None = None
    layout_items: list[ParsedLayoutItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_source: str | None = None


ParsedPage = StructuredPage


@dataclass
class BlockRelation:
    source_block_id: str
    target_block_id: str
    relation_type: str
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredBlock:
    block_id: str
    block_type: str
    text: str
    normalized_text: str
    page_number: int | None
    page_label: str | None
    order: int
    bbox: dict[str, float] | None = None
    section_path: list[str] = field(default_factory=list)
    layout_role: str = "body"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredDocument:
    doc_id: str
    source_type: SourceType
    title: str | None
    pages: list[StructuredPage]
    blocks: list[StructuredBlock]
    relations: list[BlockRelation]
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredContentCleaner:
    def clean_ingest_text(self, value: str) -> str:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        normalized = _HYPHENATED_LINEBREAK_RE.sub("", normalized)
        normalized = normalized.replace("\u00a0", " ").replace("\t", "    ")

        cleaned_lines: list[str] = []
        blank_run = 0
        in_code_block = False

        for raw_line in normalized.split("\n"):
            line = raw_line.rstrip()
            stripped = line.strip()

            if _FENCE_PATTERN.match(stripped):
                in_code_block = not in_code_block
                cleaned_lines.append(stripped)
                blank_run = 0
                continue

            if not stripped:
                blank_run += 1
                if blank_run <= 2:
                    cleaned_lines.append("")
                continue

            blank_run = 0
            if in_code_block or _TABLE_ROW_PATTERN.match(stripped):
                cleaned_lines.append(line)
                continue

            leading = re.match(r"^\s*", line).group(0)
            content = line[len(leading) :]
            content = _LINE_SPACES_RE.sub(" ", content).strip()
            cleaned_lines.append(f"{leading}{content}".rstrip())

        return "\n".join(cleaned_lines).strip()

    def clean_non_pdf_text(self, *, text: str, source_type: SourceType) -> tuple[str, list[str]]:
        cleaned_text = text
        applied_rules = ["normalize_newlines", "trim_line_whitespace", "collapse_multi_spaces", "preserve_structure"]

        if source_type is SourceType.JSON:
            normalized_json = self._normalize_json_text(text)
            if normalized_json is not None:
                cleaned_text = normalized_json
                applied_rules.append("json_key_value_normalization")

        cleaned_text = self.clean_ingest_text(cleaned_text)
        if _HYPHENATED_LINEBREAK_RE.search(text.replace("\r\n", "\n").replace("\r", "\n")):
            applied_rules.append("repair_hyphenated_linebreaks")
        return cleaned_text, applied_rules

    def build_blocks(self, *, pages: list[ParsedPage], title: str | None, include_page_markers: bool = True) -> list[StructuredBlock]:
        document = self.build_document_from_pages(
            doc_id=str(uuid4()),
            source_type=SourceType.PDF,
            title=title,
            pages=pages,
            include_page_markers=include_page_markers,
        )
        return document.blocks

    def build_document_from_pages(
        self,
        *,
        doc_id: str,
        source_type: SourceType,
        title: str | None,
        pages: list[StructuredPage],
        include_page_markers: bool = True,
    ) -> StructuredDocument:
        heading_stack = [title] if title else []
        blocks: list[StructuredBlock] = []
        global_order = 0

        for page in pages:
            page_text = self.clean_ingest_text(page.text)
            page.text = page_text
            marker_text, marker_detected = self._resolve_page_marker(page)

            if include_page_markers and marker_text:
                blocks.append(
                    StructuredBlock(
                        block_id=str(uuid4()),
                        block_type="page_marker",
                        text=marker_text,
                        normalized_text=self._normalize_relation_text(marker_text),
                        page_number=page.page_number,
                        page_label=page.page_label,
                        order=global_order,
                        bbox=None,
                        section_path=list(heading_stack),
                        layout_role="page_marker",
                        metadata={
                            "page_boxes": page.metadata.get("page_boxes", []),
                            "page_tables": page.metadata.get("tables", []),
                            "page_images": page.metadata.get("images", []),
                            "page_marker_detected": marker_detected,
                            "source_page_count": page.metadata.get("source_page_count"),
                            "parser_source": page.parser_source or page.metadata.get("parser_source"),
                            "parser_block_id": None,
                            "bbox": None,
                            "layout_role": "page_marker",
                        },
                    )
                )
                global_order += 1

            page_blocks: list[StructuredBlock]
            if page.layout_items:
                page_blocks, heading_stack, global_order = self._build_blocks_from_layout_items(
                    page=page,
                    heading_stack=heading_stack,
                    root_title=title,
                    starting_order=global_order,
                )
            else:
                page_blocks, heading_stack, global_order = self._build_blocks_from_lines(
                    page=page,
                    heading_stack=heading_stack,
                    root_title=title,
                    starting_order=global_order,
                    strip_page_marker=include_page_markers and marker_detected,
                    marker_text=marker_text,
                )
            blocks.extend(page_blocks)

        document = StructuredDocument(
            doc_id=doc_id,
            source_type=source_type,
            title=title,
            pages=pages,
            blocks=blocks,
            relations=[],
            metadata={},
        )
        self._classify_layout_roles(document)
        self._build_relationship_graph(document)
        return document
    def build_document_from_non_pdf(
        self,
        *,
        doc_id: str,
        source_type: SourceType,
        title: str | None,
        text: str,
        page_number: int = 1,
        page_label: str | None = None,
    ) -> tuple[StructuredDocument, list[str]]:
        cleaned_text, cleaning_rules = self.clean_non_pdf_text(text=text, source_type=source_type)
        page_label = page_label or str(page_number)

        if source_type is SourceType.CSV:
            blocks = self._build_csv_blocks(doc_id=doc_id, title=title, text=text, page_number=page_number, page_label=page_label)
            cleaning_rules.append("csv_table_modeling")
            document = StructuredDocument(
                doc_id=doc_id,
                source_type=source_type,
                title=title,
                pages=[StructuredPage(page_number=page_number, page_label=page_label, text=cleaned_text, parser_source="csv")],
                blocks=blocks,
                relations=[],
                metadata={},
            )
            self._build_relationship_graph(document)
            return document, cleaning_rules

        if source_type is SourceType.JSON:
            document = self._build_json_document(
                doc_id=doc_id,
                title=title,
                text=text,
                page_number=page_number,
                page_label=page_label,
            )
            cleaning_rules.append("json_structured_modeling")
            return document, cleaning_rules

        if source_type in {SourceType.WEB, SourceType.MARKDOWN, SourceType.WORD}:
            html_document = self._try_build_html_like_document(
                doc_id=doc_id,
                source_type=source_type,
                title=title,
                raw_text=text,
                page_number=page_number,
                page_label=page_label,
            )
            if html_document is not None:
                cleaning_rules.append("html_dom_structuring")
                return html_document, cleaning_rules

        page = StructuredPage(page_number=page_number, page_label=page_label, text=cleaned_text, parser_source="text")
        document = self.build_document_from_pages(
            doc_id=doc_id,
            source_type=source_type,
            title=title,
            pages=[page],
            include_page_markers=False,
        )
        return document, cleaning_rules

    def _build_blocks_from_layout_items(
        self,
        *,
        page: StructuredPage,
        heading_stack: list[str],
        root_title: str | None,
        starting_order: int,
    ) -> tuple[list[StructuredBlock], list[str], int]:
        blocks: list[StructuredBlock] = []
        order = starting_order
        sorted_items = self._sort_layout_items(page)

        for item in sorted_items:
            text = self.clean_ingest_text(item.text)
            if not text:
                continue

            item_type = self._normalize_item_type(item.item_type, text=text)
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "bbox": item.bbox,
                    "page_width": page.width,
                    "page_height": page.height,
                    "parser_block_id": item.parser_block_id,
                    "parser_source": item.parser_source or page.parser_source,
                }
            )

            if item_type == "section_header":
                level = int(metadata.get("heading_level") or self._detect_heading_level(text) or 1)
                heading_stack = self._update_heading_stack(
                    heading_stack=heading_stack,
                    level=level,
                    title=self._strip_heading_marker(text),
                    root_title=root_title,
                )
                blocks.append(
                    self._new_block(
                        block_type="section_header",
                        text=self._strip_heading_marker(text),
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="body",
                        extra_metadata=metadata | {"heading_level": level},
                    )
                )
                order += 1
                continue

            if item_type == "caption":
                blocks.append(
                    self._new_block(
                        block_type="caption",
                        text=text.strip(),
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="caption",
                        extra_metadata=metadata | self._extract_caption_metadata(text),
                    )
                )
                order += 1
                continue

            if item_type == "image":
                image_metadata = metadata | (self._extract_image_metadata(text.strip()) or {})
                image_text = self._render_image_text(image_metadata) if image_metadata else text.strip()
                blocks.append(
                    self._new_block(
                        block_type="image",
                        text=image_text,
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="body",
                        extra_metadata=image_metadata,
                    )
                )
                order += 1
                continue

            if item_type == "table":
                table_text, table_metadata = self._linearize_any_table(text=text, metadata=metadata)
                blocks.append(
                    self._new_block(
                        block_type="table",
                        text=table_text,
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="body",
                        extra_metadata=metadata | table_metadata,
                    )
                )
                order += 1
                continue

            if item_type == "formula" or self._looks_like_formula_block(text):
                blocks.append(
                    self._new_block(
                        block_type="formula",
                        text=text.strip(),
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="body",
                        extra_metadata=metadata | self._build_formula_metadata(text=text, source=item_type or "parser"),
                    )
                )
                order += 1
                continue

            if item_type == "list":
                blocks.append(
                    self._new_block(
                        block_type="list",
                        text=text.strip(),
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="body",
                        extra_metadata=metadata,
                    )
                )
                order += 1
                continue

            if item_type == "code":
                blocks.append(
                    self._new_block(
                        block_type="code",
                        text=text.strip(),
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        bbox=item.bbox,
                        layout_role="body",
                        extra_metadata=metadata,
                    )
                )
                order += 1
                continue

            blocks.append(
                self._new_block(
                    block_type="paragraph",
                    text=text.strip(),
                    page=page,
                    order=order,
                    section_path=heading_stack,
                    bbox=item.bbox,
                    layout_role="body",
                    extra_metadata=metadata,
                )
            )
            order += 1

        return blocks, heading_stack, order

    def _build_blocks_from_lines(
        self,
        *,
        page: StructuredPage,
        heading_stack: list[str],
        root_title: str | None,
        starting_order: int,
        strip_page_marker: bool,
        marker_text: str | None,
    ) -> tuple[list[StructuredBlock], list[str], int]:
        page_lines = page.text.splitlines()
        if strip_page_marker and marker_text:
            page_lines = self._remove_first_matching_line(page_lines, marker_text)

        blocks: list[StructuredBlock] = []
        order = starting_order
        index = 0

        while index < len(page_lines):
            line = page_lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue

            heading_match = _HEADING_PATTERN.match(stripped)
            if heading_match:
                heading_stack = self._update_heading_stack(
                    heading_stack=heading_stack,
                    level=len(heading_match.group("level")),
                    title=heading_match.group("title").strip(),
                    root_title=root_title,
                )
                blocks.append(
                    self._new_block(
                        block_type="section_header",
                        text=heading_match.group("title").strip(),
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        extra_metadata={"heading_level": len(heading_match.group("level"))},
                    )
                )
                order += 1
                index += 1
                continue

            if _FENCE_PATTERN.match(stripped):
                block_lines = [line]
                fence = stripped[:3]
                index += 1
                while index < len(page_lines):
                    block_lines.append(page_lines[index])
                    if page_lines[index].strip().startswith(fence):
                        index += 1
                        break
                    index += 1
                blocks.append(self._new_block(block_type="code", text="\n".join(block_lines).strip(), page=page, order=order, section_path=heading_stack))
                order += 1
                continue

            if self._is_formula_start(stripped):
                formula_text, index = self._consume_formula_block(page_lines, index)
                blocks.append(
                    self._new_block(
                        block_type="formula",
                        text=formula_text,
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        extra_metadata=self._build_formula_metadata(text=formula_text, source="markdown"),
                    )
                )
                order += 1
                continue

            if self._is_caption_line(stripped):
                blocks.append(
                    self._new_block(
                        block_type="caption",
                        text=stripped,
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        layout_role="caption",
                        extra_metadata=self._extract_caption_metadata(stripped),
                    )
                )
                order += 1
                index += 1
                continue

            image_metadata = self._extract_image_metadata(stripped)
            if image_metadata is not None:
                image_text = self._render_image_text(image_metadata)
                blocks.append(self._new_block(block_type="image", text=image_text, page=page, order=order, section_path=heading_stack, extra_metadata=image_metadata))
                order += 1
                index += 1
                continue

            if self._is_table_row(stripped):
                block_lines = [line]
                index += 1
                while index < len(page_lines) and self._is_table_row(page_lines[index].strip()):
                    block_lines.append(page_lines[index])
                    index += 1
                table_text, table_metadata = self._linearize_table_block("\n".join(block_lines).strip())
                blocks.append(
                    self._new_block(
                        block_type="table",
                        text=table_text,
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        extra_metadata=table_metadata | {"raw_table_text": "\n".join(block_lines).strip()},
                    )
                )
                order += 1
                continue

            if _LIST_ITEM_PATTERN.match(stripped):
                block_lines = [line]
                index += 1
                while index < len(page_lines):
                    candidate = page_lines[index].strip()
                    if not candidate:
                        break
                    if _HEADING_PATTERN.match(candidate) or _FENCE_PATTERN.match(candidate) or self._is_table_row(candidate) or self._extract_image_metadata(candidate) is not None or self._is_caption_line(candidate) or self._is_formula_start(candidate):
                        break
                    if _LIST_ITEM_PATTERN.match(candidate) or page_lines[index].startswith((" ", "\t")):
                        block_lines.append(page_lines[index])
                        index += 1
                        continue
                    break
                blocks.append(self._new_block(block_type="list", text="\n".join(block_lines).strip(), page=page, order=order, section_path=heading_stack))
                order += 1
                continue

            if self._looks_like_formula_block(stripped):
                blocks.append(
                    self._new_block(
                        block_type="formula",
                        text=stripped,
                        page=page,
                        order=order,
                        section_path=heading_stack,
                        extra_metadata=self._build_formula_metadata(text=stripped, source="heuristic"),
                    )
                )
                order += 1
                index += 1
                continue

            block_lines = [line]
            index += 1
            while index < len(page_lines):
                candidate = page_lines[index].strip()
                if not candidate:
                    break
                if _HEADING_PATTERN.match(candidate) or _FENCE_PATTERN.match(candidate) or self._is_table_row(candidate) or _LIST_ITEM_PATTERN.match(candidate) or self._extract_image_metadata(candidate) is not None or self._is_caption_line(candidate) or self._is_formula_start(candidate):
                    break
                if self._looks_like_formula_block(candidate):
                    break
                block_lines.append(page_lines[index])
                index += 1
            blocks.append(self._new_block(block_type="paragraph", text="\n".join(block_lines).strip(), page=page, order=order, section_path=heading_stack))
            order += 1

        return blocks, heading_stack, order

    def _build_csv_blocks(self, *, doc_id: str, title: str | None, text: str, page_number: int, page_label: str) -> list[StructuredBlock]:
        rows = list(csv.reader(text.splitlines()))
        if not rows:
            return []
        headers = rows[0]
        data_rows = rows[1:]
        table_text = self._linearize_rows(headers=headers, rows=data_rows)
        return [
            StructuredBlock(
                block_id=str(uuid4()),
                block_type="table",
                text=table_text,
                normalized_text=self._normalize_relation_text(table_text),
                page_number=page_number,
                page_label=page_label,
                order=0,
                section_path=[title] if title else [],
                layout_role="body",
                metadata={
                    "table_headers": headers,
                    "table_row_count": len(data_rows),
                    "raw_table_text": text.strip(),
                    "source_page_count": 1,
                    "bbox": None,
                    "parser_block_id": f"{doc_id}-csv-table-0",
                    "parser_source": "csv",
                    "related_block_ids": [],
                    "references_figures": [],
                    "references_tables": [],
                },
            )
        ]

    def _build_json_document(self, *, doc_id: str, title: str | None, text: str, page_number: int, page_label: str) -> StructuredDocument:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text

        blocks: list[StructuredBlock] = []
        order = 0
        if isinstance(payload, list) and self._is_homogeneous_object_list(payload):
            headers = list(payload[0].keys()) if payload else []
            rows = [[self._stringify_json_scalar(item.get(header)) for header in headers] for item in payload]
            table_text = self._linearize_rows(headers=headers, rows=rows)
            blocks.append(
                StructuredBlock(
                    block_id=str(uuid4()),
                    block_type="table",
                    text=table_text,
                    normalized_text=self._normalize_relation_text(table_text),
                    page_number=page_number,
                    page_label=page_label,
                    order=order,
                    section_path=[title] if title else [],
                    layout_role="body",
                    metadata={
                        "table_headers": headers,
                        "table_row_count": len(rows),
                        "json_table": True,
                        "bbox": None,
                        "parser_block_id": f"{doc_id}-json-table-0",
                        "parser_source": "json",
                        "related_block_ids": [],
                        "references_figures": [],
                        "references_tables": [],
                    },
                )
            )
        else:
            blocks.extend(self._flatten_json_to_blocks(payload=payload, page_number=page_number, page_label=page_label, section_path=[title] if title else [], starting_order=order))

        document = StructuredDocument(
            doc_id=doc_id,
            source_type=SourceType.JSON,
            title=title,
            pages=[StructuredPage(page_number=page_number, page_label=page_label, text=self.clean_ingest_text(text), parser_source="json")],
            blocks=blocks,
            relations=[],
            metadata={},
        )
        self._build_relationship_graph(document)
        return document
    def _flatten_json_to_blocks(
        self,
        *,
        payload: Any,
        page_number: int,
        page_label: str,
        section_path: list[str],
        starting_order: int,
        path_prefix: str = "",
    ) -> list[StructuredBlock]:
        blocks: list[StructuredBlock] = []
        order = starting_order

        if isinstance(payload, dict):
            for key, value in payload.items():
                next_path = f"{path_prefix}.{key}" if path_prefix else str(key)
                if isinstance(value, dict):
                    next_section = [*section_path, str(key)]
                    blocks.append(
                        StructuredBlock(
                            block_id=str(uuid4()),
                            block_type="section_header",
                            text=str(key),
                            normalized_text=self._normalize_relation_text(str(key)),
                            page_number=page_number,
                            page_label=page_label,
                            order=order,
                            section_path=next_section,
                            layout_role="body",
                            metadata={"heading_level": len(next_section), "bbox": None, "parser_block_id": next_path, "parser_source": "json", "related_block_ids": [], "references_figures": [], "references_tables": []},
                        )
                    )
                    order += 1
                    child_blocks = self._flatten_json_to_blocks(payload=value, page_number=page_number, page_label=page_label, section_path=next_section, starting_order=order, path_prefix=next_path)
                    blocks.extend(child_blocks)
                    order = max((block.order for block in child_blocks), default=order - 1) + 1
                    continue
                if isinstance(value, list) and self._is_homogeneous_object_list(value):
                    headers = list(value[0].keys()) if value else []
                    rows = [[self._stringify_json_scalar(item.get(header)) for header in headers] for item in value]
                    table_text = self._linearize_rows(headers=headers, rows=rows)
                    blocks.append(
                        StructuredBlock(
                            block_id=str(uuid4()),
                            block_type="table",
                            text=table_text,
                            normalized_text=self._normalize_relation_text(table_text),
                            page_number=page_number,
                            page_label=page_label,
                            order=order,
                            section_path=[*section_path, str(key)],
                            layout_role="body",
                            metadata={"table_headers": headers, "table_row_count": len(rows), "json_table": True, "bbox": None, "parser_block_id": next_path, "parser_source": "json", "related_block_ids": [], "references_figures": [], "references_tables": []},
                        )
                    )
                    order += 1
                    continue
                if isinstance(value, list):
                    lines = [f"- {self._stringify_json_scalar(item)}" for item in value]
                    if lines:
                        text_value = "\n".join(lines)
                        blocks.append(
                            StructuredBlock(
                                block_id=str(uuid4()),
                                block_type="list",
                                text=text_value,
                                normalized_text=self._normalize_relation_text(text_value),
                                page_number=page_number,
                                page_label=page_label,
                                order=order,
                                section_path=[*section_path, str(key)],
                                layout_role="body",
                                metadata={"bbox": None, "parser_block_id": next_path, "parser_source": "json", "related_block_ids": [], "references_figures": [], "references_tables": []},
                            )
                        )
                        order += 1
                    continue
                text_value = f"{next_path}={self._stringify_json_scalar(value)}"
                blocks.append(
                    StructuredBlock(
                        block_id=str(uuid4()),
                        block_type="paragraph",
                        text=text_value,
                        normalized_text=self._normalize_relation_text(text_value),
                        page_number=page_number,
                        page_label=page_label,
                        order=order,
                        section_path=section_path,
                        layout_role="body",
                        metadata={"bbox": None, "parser_block_id": next_path, "parser_source": "json", "related_block_ids": [], "references_figures": [], "references_tables": []},
                    )
                )
                order += 1
            return blocks

        text_value = self._stringify_json_scalar(payload)
        blocks.append(
            StructuredBlock(
                block_id=str(uuid4()),
                block_type="paragraph",
                text=text_value,
                normalized_text=self._normalize_relation_text(text_value),
                page_number=page_number,
                page_label=page_label,
                order=starting_order,
                section_path=section_path,
                layout_role="body",
                metadata={"bbox": None, "parser_block_id": path_prefix or "root", "parser_source": "json", "related_block_ids": [], "references_figures": [], "references_tables": []},
            )
        )
        return blocks

    def _try_build_html_like_document(self, *, doc_id: str, source_type: SourceType, title: str | None, raw_text: str, page_number: int, page_label: str) -> StructuredDocument | None:
        if "<" not in raw_text or ">" not in raw_text:
            return None
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        soup = BeautifulSoup(raw_text, "html.parser")
        body = soup.body or soup
        elements = [element for element in body.descendants if getattr(element, "name", None) in _BLOCK_TAGS]
        if not elements:
            return None

        page = StructuredPage(page_number=page_number, page_label=page_label, text=self.clean_ingest_text(body.get_text("\n")), parser_source="html")
        blocks: list[StructuredBlock] = []
        heading_stack = [title] if title else []
        order = 0
        seen_nodes: set[int] = set()

        for element in elements:
            if id(element) in seen_nodes:
                continue
            seen_nodes.add(id(element))
            tag_name = element.name.lower()
            text_value = self.clean_ingest_text(element.get_text("\n"))
            if not text_value and tag_name != "img":
                continue

            if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
                heading_stack = self._update_heading_stack(heading_stack=heading_stack, level=int(tag_name[1]), title=text_value, root_title=title)
                blocks.append(StructuredBlock(block_id=str(uuid4()), block_type="section_header", text=text_value, normalized_text=self._normalize_relation_text(text_value), page_number=page_number, page_label=page_label, order=order, section_path=list(heading_stack), layout_role="body", metadata={"heading_level": int(tag_name[1]), "bbox": None, "parser_block_id": None, "parser_source": "html", "related_block_ids": [], "references_figures": [], "references_tables": []}))
                order += 1
                continue

            if tag_name in {"pre", "code"}:
                blocks.append(self._new_non_pdf_block("code", text_value, page_number, page_label, order, heading_stack, "html"))
                order += 1
                continue

            if tag_name in {"ul", "ol"}:
                list_lines = [f"- {self.clean_ingest_text(item.get_text(' '))}" for item in element.find_all("li", recursive=False)]
                list_text = "\n".join(line for line in list_lines if line.strip())
                if list_text:
                    blocks.append(self._new_non_pdf_block("list", list_text, page_number, page_label, order, heading_stack, "html"))
                    order += 1
                continue

            if tag_name == "table":
                table_text, table_metadata = self._linearize_html_table(element)
                blocks.append(self._new_non_pdf_block("table", table_text, page_number, page_label, order, heading_stack, "html", extra_metadata=table_metadata))
                order += 1
                continue

            if tag_name == "img":
                image_metadata = {"image_alt_text": element.get("alt"), "image_source": element.get("src"), "image_title": element.get("title")}
                blocks.append(self._new_non_pdf_block("image", self._render_image_text({key: value for key, value in image_metadata.items() if value}), page_number, page_label, order, heading_stack, "html", extra_metadata={key: value for key, value in image_metadata.items() if value}))
                order += 1
                continue

            if tag_name == "figcaption":
                blocks.append(self._new_non_pdf_block("caption", text_value, page_number, page_label, order, heading_stack, "html", extra_metadata=self._extract_caption_metadata(text_value)))
                order += 1
                continue

            if tag_name == "math" or self._looks_like_formula_block(text_value):
                blocks.append(self._new_non_pdf_block("formula", text_value, page_number, page_label, order, heading_stack, "html", extra_metadata=self._build_formula_metadata(text=text_value, source="html")))
                order += 1
                continue

            if self._is_caption_line(text_value):
                blocks.append(self._new_non_pdf_block("caption", text_value, page_number, page_label, order, heading_stack, "html", extra_metadata=self._extract_caption_metadata(text_value)))
                order += 1
                continue

            blocks.append(self._new_non_pdf_block("paragraph", text_value, page_number, page_label, order, heading_stack, "html"))
            order += 1

        if not blocks:
            return None

        document = StructuredDocument(doc_id=doc_id, source_type=source_type, title=title, pages=[page], blocks=blocks, relations=[], metadata={})
        self._build_relationship_graph(document)
        return document

    def _new_non_pdf_block(self, block_type: str, text: str, page_number: int, page_label: str, order: int, section_path: list[str], parser_source: str, extra_metadata: dict[str, Any] | None = None) -> StructuredBlock:
        metadata = {"bbox": None, "parser_block_id": None, "parser_source": parser_source, "related_block_ids": [], "references_figures": [], "references_tables": []}
        if extra_metadata:
            metadata.update(extra_metadata)
        layout_role = "caption" if block_type == "caption" else "body"
        return StructuredBlock(block_id=str(uuid4()), block_type=block_type, text=text, normalized_text=self._normalize_relation_text(text), page_number=page_number, page_label=page_label, order=order, bbox=None, section_path=list(section_path), layout_role=layout_role, metadata=metadata)

    def _new_block(self, *, block_type: str, text: str, page: StructuredPage, order: int, section_path: list[str], bbox: dict[str, float] | None = None, layout_role: str = "body", extra_metadata: dict[str, Any] | None = None) -> StructuredBlock:
        metadata = {"page_boxes": page.metadata.get("page_boxes", []), "source_page_count": page.metadata.get("source_page_count"), "page_label": page.page_label, "related_block_ids": [], "references_figures": [], "references_tables": [], "bbox": bbox, "parser_block_id": None, "parser_source": page.parser_source or page.metadata.get("parser_source"), "page_width": page.width, "page_height": page.height}
        if extra_metadata:
            metadata.update(extra_metadata)
        return StructuredBlock(block_id=str(uuid4()), block_type=block_type, text=text, normalized_text=self._normalize_relation_text(text), page_number=page.page_number, page_label=page.page_label, order=order, bbox=bbox, section_path=list(section_path), layout_role=layout_role, metadata=metadata)
    def _build_relationship_graph(self, document: StructuredDocument) -> None:
        relations: list[BlockRelation] = []
        blocks = sorted(document.blocks, key=lambda item: (item.page_number or 0, item.order))
        page_groups: dict[int | None, list[StructuredBlock]] = {}
        for block in blocks:
            page_groups.setdefault(block.page_number, []).append(block)

        for page_blocks in page_groups.values():
            ordered = [block for block in sorted(page_blocks, key=lambda item: item.order) if block.block_type != "page_marker"]
            for previous, current in zip(ordered, ordered[1:]):
                relations.append(self._relation(previous.block_id, current.block_id, "next_block"))
                relations.append(self._relation(current.block_id, previous.block_id, "prev_block"))

        headers_by_path: dict[tuple[str, ...], StructuredBlock] = {}
        for block in blocks:
            path_key = tuple(block.section_path)
            if block.block_type == "section_header":
                headers_by_path[path_key] = block
                parent_key = tuple(block.section_path[:-1])
                if parent_key and parent_key in headers_by_path:
                    parent = headers_by_path[parent_key]
                    relations.append(self._relation(block.block_id, parent.block_id, "section_parent"))
                    relations.append(self._relation(parent.block_id, block.block_id, "section_child"))
                continue
            if path_key and path_key in headers_by_path:
                parent = headers_by_path[path_key]
                relations.append(self._relation(block.block_id, parent.block_id, "section_parent"))
                relations.append(self._relation(parent.block_id, block.block_id, "section_child"))

        caption_targets: dict[tuple[str, str], StructuredBlock] = {}
        typed_targets: dict[str, dict[str, StructuredBlock]] = {"figure": {}, "table": {}, "formula": {}}
        for block in blocks:
            identifier = self._extract_reference_identifier(block)
            if identifier is None:
                continue
            ref_kind, ref_number = identifier
            if ((block.block_type == "image" and ref_kind == "figure") or (block.block_type == "table" and ref_kind == "table") or (block.block_type == "formula" and ref_kind == "formula")):
                typed_targets[ref_kind][ref_number] = block
                caption_targets[(ref_kind, ref_number)] = block

        captions = [block for block in blocks if block.block_type == "caption"]
        for caption in captions:
            caption_identifier = self._extract_reference_identifier(caption)
            best_target: StructuredBlock | None = caption_targets.get(caption_identifier) if caption_identifier is not None else None
            if best_target is None:
                best_target = self._find_best_caption_target(caption, blocks)
            if best_target is None:
                continue
            relations.append(self._relation(caption.block_id, best_target.block_id, "caption_of"))
            relations.append(self._relation(best_target.block_id, caption.block_id, "has_caption"))
            if best_target.block_type == "table":
                relations.append(self._relation(caption.block_id, best_target.block_id, "table_row_context", weight=0.86))
            best_target.metadata["caption_text"] = caption.text
            best_target.metadata["caption_type"] = caption.metadata.get("caption_type")
            best_target.metadata["caption_source_page"] = caption.page_number
            if caption_identifier is not None:
                typed_targets.setdefault(caption_identifier[0], {})[caption_identifier[1]] = best_target

        for block in blocks:
            if block.block_type not in {"paragraph", "list", "code", "caption"}:
                continue
            figure_refs: list[str] = []
            table_refs: list[str] = []
            formula_refs: list[str] = []
            for ref_kind, ref_number in self._extract_references(block.text):
                target = typed_targets.get(ref_kind, {}).get(ref_number)
                if target is None:
                    continue
                relations.append(self._relation(block.block_id, target.block_id, "references"))
                relations.append(self._relation(target.block_id, block.block_id, "referenced_by"))
                if ref_kind == "formula":
                    relations.append(self._relation(block.block_id, target.block_id, "formula_explains", weight=0.88))
                    formula_refs.append(ref_number)
                elif ref_kind == "figure":
                    figure_refs.append(ref_number)
                else:
                    table_refs.append(ref_number)
            if figure_refs:
                block.metadata["references_figures"] = sorted(set(figure_refs), key=int)
            if table_refs:
                block.metadata["references_tables"] = sorted(set(table_refs), key=int)
            if formula_refs:
                block.metadata["references_formulas"] = sorted(set(formula_refs), key=int)

        for page_number, page_blocks in page_groups.items():
            page = next((item for item in document.pages if item.page_number == page_number), None)
            if page is None or not page.width or not page.height:
                continue
            diagonal = math.sqrt(page.width**2 + page.height**2)
            max_distance = diagonal * 0.18
            blocks_with_bbox = [block for block in page_blocks if block.bbox]
            for block in blocks_with_bbox:
                distances: list[tuple[float, StructuredBlock]] = []
                for candidate in blocks_with_bbox:
                    if candidate.block_id == block.block_id:
                        continue
                    distance = self._bbox_center_distance(block.bbox, candidate.bbox)
                    if distance <= max_distance:
                        distances.append((distance, candidate))
                for distance, candidate in sorted(distances, key=lambda item: item[0])[:4]:
                    weight = max(0.3, 1.0 - distance / max(diagonal, 1.0))
                    relations.append(self._relation(block.block_id, candidate.block_id, "same_page_near", weight=round(weight, 4)))

        continued_candidates = [block for block in blocks if block.block_type in {"paragraph", "list", "table", "formula"}]
        for previous, current in zip(continued_candidates, continued_candidates[1:]):
            if previous.page_number is None or current.page_number is None or current.page_number != previous.page_number + 1:
                continue
            if previous.block_type != current.block_type or previous.section_path != current.section_path:
                continue
            if not self._looks_like_continuation(previous.text, current.text):
                continue
            relations.append(self._relation(current.block_id, previous.block_id, "continued_from", weight=0.8))

        deduped_relations = self._dedupe_relations(relations)
        document.relations = deduped_relations
        outgoing_by_source: dict[str, list[BlockRelation]] = {}
        for relation in deduped_relations:
            outgoing_by_source.setdefault(relation.source_block_id, []).append(relation)

        for block in blocks:
            edges = outgoing_by_source.get(block.block_id, [])
            graph_edges = [{"type": edge.relation_type, "target_block_id": edge.target_block_id, "weight": edge.weight} for edge in edges]
            graph_neighbors = list(dict.fromkeys(edge.target_block_id for edge in edges))
            block.metadata["graph_edges"] = graph_edges
            block.metadata["graph_neighbors"] = graph_neighbors
            block.metadata["related_block_ids"] = list(dict.fromkeys([*block.metadata.get("related_block_ids", []), *graph_neighbors]))
            block.metadata["layout_role"] = block.layout_role
            block.metadata["bbox"] = block.bbox
            block.metadata["block_order"] = block.order
            block.metadata["parser_block_id"] = block.metadata.get("parser_block_id")
            block.metadata["parser_source"] = block.metadata.get("parser_source")

    def _classify_layout_roles(self, document: StructuredDocument) -> None:
        text_frequency: dict[str, int] = {}
        for block in document.blocks:
            normalized = self._normalize_relation_text(block.text)
            if normalized and len(normalized) <= 80:
                text_frequency[normalized] = text_frequency.get(normalized, 0) + 1

        for block in document.blocks:
            if block.block_type == "page_marker":
                block.layout_role = "page_marker"
                block.metadata["excluded_from_body"] = False
                continue
            if block.block_type == "caption":
                block.layout_role = "caption"
                block.metadata["excluded_from_body"] = False
                continue

            bbox = block.bbox
            page_height = block.metadata.get("page_height")
            normalized = self._normalize_relation_text(block.text)
            repeated = text_frequency.get(normalized, 0) > 1
            has_reference_signal = bool(self._extract_references(block.text))
            is_short = len(normalized) <= 80 if normalized else False

            if bbox and page_height:
                y0 = float(bbox.get("y0", 0.0))
                y1 = float(bbox.get("y1", 0.0))
                top_ratio = y0 / max(float(page_height), 1.0)
                bottom_ratio = y1 / max(float(page_height), 1.0)
                if top_ratio <= 0.12 and repeated and is_short and not has_reference_signal:
                    block.layout_role = "header"
                    block.metadata["excluded_from_body"] = True
                    continue
                if bottom_ratio >= 0.88 and repeated and is_short and not has_reference_signal:
                    block.layout_role = "footer"
                    block.metadata["excluded_from_body"] = True
                    continue
                if bottom_ratio >= 0.9 and is_short:
                    block.layout_role = "footnote"
                    block.metadata["excluded_from_body"] = False
                    continue

            block.layout_role = "body"
            block.metadata["excluded_from_body"] = False

    def _sort_layout_items(self, page: StructuredPage) -> list[ParsedLayoutItem]:
        if not page.layout_items:
            return []
        items_with_bbox = [item for item in page.layout_items if item.bbox]
        if not items_with_bbox:
            return sorted(page.layout_items, key=lambda item: item.order)

        page_height = float(page.height or 0.0)
        tolerance = min(18.0, max(6.0, page_height * 0.008)) if page_height else 10.0
        sorted_items = sorted(page.layout_items, key=lambda item: (float((item.bbox or {}).get("y0", item.order)), float((item.bbox or {}).get("x0", 0.0)), item.order))
        bands: list[list[ParsedLayoutItem]] = []
        for item in sorted_items:
            if not item.bbox:
                bands.append([item])
                continue
            y0 = float(item.bbox.get("y0", 0.0))
            placed = False
            for band in bands:
                anchor = next((candidate for candidate in band if candidate.bbox), None)
                anchor_y = float((anchor.bbox or {}).get("y0", 0.0)) if anchor else None
                if anchor_y is not None and abs(y0 - anchor_y) <= tolerance:
                    band.append(item)
                    placed = True
                    break
            if not placed:
                bands.append([item])

        ordered: list[ParsedLayoutItem] = []
        for band in bands:
            ordered.extend(sorted(band, key=lambda item: (float((item.bbox or {}).get("x0", 0.0)), item.order)))
        return ordered
    def _relation(self, source_block_id: str, target_block_id: str, relation_type: str, weight: float | None = None) -> BlockRelation:
        return BlockRelation(source_block_id=source_block_id, target_block_id=target_block_id, relation_type=relation_type, weight=weight if weight is not None else _RELATION_PRIORITY[relation_type])

    def _dedupe_relations(self, relations: list[BlockRelation]) -> list[BlockRelation]:
        unique: dict[tuple[str, str, str], BlockRelation] = {}
        for relation in relations:
            key = (relation.source_block_id, relation.target_block_id, relation.relation_type)
            existing = unique.get(key)
            if existing is None or relation.weight > existing.weight:
                unique[key] = relation
        return list(unique.values())

    def _normalize_item_type(self, item_type: str | None, *, text: str) -> str:
        if not item_type:
            if self._is_caption_line(text):
                return "caption"
            if self._extract_image_metadata(text.strip()) is not None:
                return "image"
            if self._is_table_row(text.strip()):
                return "table"
            if self._looks_like_formula_block(text):
                return "formula"
            return "paragraph"

        normalized = item_type.lower().strip().replace("-", "_").replace(" ", "_")
        mapping = {"title": "section_header", "heading": "section_header", "header": "section_header", "text": "paragraph", "para": "paragraph", "paragraph": "paragraph", "table": "table", "image": "image", "figure": "image", "caption": "caption", "code": "code", "formula": "formula", "equation": "formula", "list": "list", "bullet": "list"}
        return mapping.get(normalized, normalized)

    def _resolve_page_marker(self, page: StructuredPage) -> tuple[str | None, bool]:
        page_lines = page.text.splitlines()
        page_non_empty_lines = [line for line in page_lines if line.strip()]
        if page_non_empty_lines and self._is_page_marker_line(page_non_empty_lines[0]):
            return page_non_empty_lines[0].strip(), True
        if page.page_label:
            return str(page.page_label), False
        if page.page_number is not None:
            return str(page.page_number), False
        return None, False

    def _remove_first_matching_line(self, lines: list[str], target: str) -> list[str]:
        removed = False
        remaining: list[str] = []
        for line in lines:
            if not removed and line.strip() == target:
                removed = True
                continue
            remaining.append(line)
        return remaining

    def _update_heading_stack(self, *, heading_stack: list[str], level: int, title: str, root_title: str | None) -> list[str]:
        if level <= 1:
            return [title]
        if not heading_stack:
            return [root_title, title] if root_title and root_title != title else [title]
        parent_depth = min(level - 1, len(heading_stack))
        updated_stack = [*heading_stack[:parent_depth], title]
        return [item for item in updated_stack if item]

    def _consume_formula_block(self, lines: list[str], start_index: int) -> tuple[str, int]:
        block_lines = [lines[start_index].strip()]
        index = start_index + 1
        start_marker = block_lines[0][:2]
        while index < len(lines):
            block_lines.append(lines[index].strip())
            if lines[index].strip().startswith(start_marker) or _FORMULA_BLOCK_END_PATTERN.match(lines[index].strip()):
                index += 1
                break
            index += 1
        return "\n".join(line for line in block_lines if line).strip(), index

    def _is_formula_start(self, line: str) -> bool:
        return _FORMULA_BLOCK_START_PATTERN.match(line) is not None

    def _looks_like_formula_block(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.startswith("$$") or stripped.startswith("\\[") or stripped.startswith("\\("):
            return True
        formula_hits = len(_FORMULA_SYMBOL_PATTERN.findall(stripped))
        word_hits = len(_WORD_PATTERN.findall(stripped))
        return formula_hits >= 2 and word_hits <= max(2, formula_hits // 2)

    def _build_formula_metadata(self, *, text: str, source: str) -> dict[str, Any]:
        symbols = sorted(set(match.group(0) for match in _FORMULA_SYMBOL_PATTERN.finditer(text)))
        return {"formula_format": "display_math" if "\n" in text or text.strip().startswith(("$$", "\\[")) else "inline_like", "formula_source": source, "formula_symbols": symbols, "formula_linearized_text": self._normalize_relation_text(text)}

    def _find_best_caption_target(self, caption: StructuredBlock, blocks: list[StructuredBlock]) -> StructuredBlock | None:
        expected_type = caption.metadata.get("caption_type")
        candidates = [block for block in blocks if block.page_number == caption.page_number and block.block_type in {"image", "table", "formula"}]
        best_candidate: StructuredBlock | None = None
        best_distance = 10**9
        for candidate in candidates:
            if expected_type and candidate.block_type == "image" and expected_type != "figure":
                continue
            if expected_type and candidate.block_type == "table" and expected_type != "table":
                continue
            if expected_type and candidate.block_type == "formula" and expected_type != "formula":
                continue
            distance = abs(candidate.order - caption.order)
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate
        return best_candidate

    def _extract_reference_identifier(self, block: StructuredBlock) -> tuple[str, str] | None:
        source_text = block.metadata.get("caption_text") or block.text
        match = _REFERENCE_PATTERN.search(source_text)
        if not match:
            if block.block_type == "formula" and block.metadata.get("caption_number"):
                return "formula", str(block.metadata["caption_number"])
            return None
        return self._normalize_reference_kind(match.group("kind")), match.group("number")

    def _extract_references(self, text: str) -> list[tuple[str, str]]:
        return [(self._normalize_reference_kind(match.group("kind")), match.group("number")) for match in _REFERENCE_PATTERN.finditer(text)]

    def _normalize_reference_kind(self, raw_kind: str) -> str:
        lowered = raw_kind.lower()
        if lowered.startswith("fig") or raw_kind in {"图", "图片"}:
            return "figure"
        if lowered.startswith("table") or raw_kind == "表":
            return "table"
        return "formula"

    def _normalize_json_text(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        flattened_lines: list[str] = []
        self._flatten_json(payload, prefix="", output=flattened_lines)
        return "\n".join(flattened_lines)

    def _flatten_json(self, payload: Any, *, prefix: str, output: list[str]) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                self._flatten_json(value, prefix=next_prefix, output=output)
            return
        if isinstance(payload, list):
            for index, value in enumerate(payload):
                next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                self._flatten_json(value, prefix=next_prefix, output=output)
            return
        output.append(f"{prefix}={payload}")

    def _is_homogeneous_object_list(self, payload: Any) -> bool:
        return bool(payload) and isinstance(payload, list) and all(isinstance(item, dict) for item in payload)

    def _stringify_json_scalar(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return "" if value is None else str(value)

    def _linearize_any_table(self, *, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if metadata.get("rows") and isinstance(metadata["rows"], list):
            rows = metadata["rows"]
            headers = metadata.get("headers") or []
            if rows and isinstance(rows[0], dict):
                headers = headers or list(rows[0].keys())
                data_rows = [[self._stringify_json_scalar(row.get(header)) for header in headers] for row in rows]
                table_text = self._linearize_rows(headers=headers, rows=data_rows)
                return table_text, {"table_headers": headers, "table_row_count": len(data_rows), "raw_table_text": text}
        if self._is_table_row(text.splitlines()[0].strip()) if text.splitlines() else False:
            return self._linearize_table_block(text)
        return text, {"table_headers": metadata.get("headers", []), "table_row_count": len(metadata.get("rows", [])), "raw_table_text": text}

    def _linearize_table_block(self, text: str) -> tuple[str, dict[str, Any]]:
        rows = [self._split_table_row(line) for line in text.splitlines() if self._is_table_row(line.strip())]
        if not rows:
            return text, {}
        headers = rows[0]
        data_rows = rows[1:]
        if data_rows and all(_TABLE_ALIGNMENT_PATTERN.match(cell.strip()) for cell in data_rows[0]):
            data_rows = data_rows[1:]
        else:
            headers = [f"column_{index + 1}" for index in range(len(headers))]
            data_rows = rows
        metadata = {"table_headers": headers, "table_row_count": len(data_rows)}
        return self._linearize_rows(headers=headers, rows=data_rows), metadata

    def _linearize_rows(self, *, headers: list[str], rows: list[list[str]]) -> str:
        linearized_lines = [f"Table columns: {', '.join(headers)}"] if headers else ["Table"]
        for row_index, row in enumerate(rows, start=1):
            padded_row = row + [""] * (len(headers) - len(row))
            assignments = [f"{header}={value.strip()}" for header, value in zip(headers, padded_row, strict=False) if header.strip() and value.strip()]
            if assignments:
                linearized_lines.append(f"Row {row_index}: {'; '.join(assignments)}")
        return "\n".join(linearized_lines)

    def _linearize_html_table(self, element: Any) -> tuple[str, dict[str, Any]]:
        headers = [self.clean_ingest_text(cell.get_text(" ")) for cell in element.find_all("th")]
        rows: list[list[str]] = []
        for row in element.find_all("tr"):
            cells = [self.clean_ingest_text(cell.get_text(" ")) for cell in row.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if headers and rows and rows[0] == headers:
            rows = rows[1:]
        elif not headers and rows:
            headers = [f"column_{index + 1}" for index in range(len(rows[0]))]
        return self._linearize_rows(headers=headers, rows=rows), {"table_headers": headers, "table_row_count": len(rows)}

    def _render_image_text(self, metadata: dict[str, Any]) -> str:
        image_lines = ["Image block"]
        if metadata.get("image_alt_text"):
            image_lines.append(f"Alt: {metadata['image_alt_text']}")
        if metadata.get("image_source"):
            image_lines.append(f"Source: {metadata['image_source']}")
        if metadata.get("image_title"):
            image_lines.append(f"Title: {metadata['image_title']}")
        if metadata.get("caption_text"):
            image_lines.append(f"Caption: {metadata['caption_text']}")
        return "\n".join(image_lines)

    def _extract_image_metadata(self, line: str) -> dict[str, Any] | None:
        markdown_match = _MD_IMAGE_PATTERN.match(line)
        if markdown_match:
            metadata = {"image_alt_text": markdown_match.group("alt").strip() or None, "image_source": markdown_match.group("src").strip(), "image_title": markdown_match.group("title").strip() if markdown_match.group("title") else None}
            return {key: value for key, value in metadata.items() if value}
        html_match = _HTML_IMG_PATTERN.match(line)
        if html_match:
            attrs = {match.group("name").lower(): match.group("value").strip() for match in _HTML_ATTR_PATTERN.finditer(html_match.group("attrs"))}
            metadata = {"image_alt_text": attrs.get("alt"), "image_source": attrs.get("src"), "image_title": attrs.get("title")}
            filtered = {key: value for key, value in metadata.items() if value}
            return filtered or None
        return None

    def _extract_caption_metadata(self, line: str) -> dict[str, Any]:
        match = _CAPTION_PATTERN.match(line.strip())
        if not match:
            return {}
        raw_kind = (match.group("kind") or "").lower()
        if raw_kind.startswith("table") or raw_kind == "表":
            caption_type = "table"
        elif raw_kind.startswith("equ") or raw_kind.startswith("formula") or raw_kind == "公式":
            caption_type = "formula"
        else:
            caption_type = "figure"
        metadata = {"caption_text": line.strip(), "caption_type": caption_type}
        if match.group("number"):
            metadata["caption_number"] = match.group("number")
        return metadata

    def _is_caption_line(self, line: str) -> bool:
        return _CAPTION_PATTERN.match(line) is not None

    def _is_page_marker_line(self, line: str) -> bool:
        return _PAGE_MARKER_PATTERN.match(line.strip()) is not None

    def _normalize_relation_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    def _detect_heading_level(self, text: str) -> int | None:
        match = _HEADING_PATTERN.match(text.strip())
        return len(match.group("level")) if match else None

    def _strip_heading_marker(self, text: str) -> str:
        match = _HEADING_PATTERN.match(text.strip())
        return match.group("title").strip() if match else text.strip()

    def _bbox_center_distance(self, left: dict[str, float] | None, right: dict[str, float] | None) -> float:
        if not left or not right:
            return 10**9
        left_center = ((float(left.get("x0", 0.0)) + float(left.get("x1", 0.0))) / 2.0, (float(left.get("y0", 0.0)) + float(left.get("y1", 0.0))) / 2.0)
        right_center = ((float(right.get("x0", 0.0)) + float(right.get("x1", 0.0))) / 2.0, (float(right.get("y0", 0.0)) + float(right.get("y1", 0.0))) / 2.0)
        return math.sqrt((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2)

    def _looks_like_continuation(self, previous_text: str, current_text: str) -> bool:
        previous = previous_text.strip()
        current = current_text.strip()
        if not previous or not current:
            return False
        if previous.endswith((":", ",", "(", "[")):
            return True
        return current[:1].islower()

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return _TABLE_ROW_PATTERN.match(line.strip()) is not None

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [cell.strip() for cell in stripped.split("|")]



