from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from backend.src.core.models import SourceType

_LINE_SPACES_RE = re.compile(r" {2,}")
_HYPHENATED_LINEBREAK_RE = re.compile(r"(?<=\w)-\n(?=\w)")
_HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S+")
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
    r"^(?:(?:caption:\s*)?(?P<kind>figure|fig\.|table|image|图|图片|表)\s*(?P<number>\d+)?)\s*[:.\-]?\s*(?P<caption>.+)$",
    re.IGNORECASE,
)
_REFERENCE_PATTERN = re.compile(r"(?P<kind>figure|fig\.|table)\s+(?P<number>\d+)", re.IGNORECASE)


@dataclass
class ParsedPage:
    page_number: int
    page_label: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredBlock:
    block_id: str
    block_type: str
    text: str
    page_number: int | None
    page_label: str | None
    order: int
    section_path: list[str] = field(default_factory=list)
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

    def build_blocks(
        self,
        *,
        pages: list[ParsedPage],
        title: str | None,
        include_page_markers: bool = True,
    ) -> list[StructuredBlock]:
        heading_stack = [title] if title else []
        blocks: list[StructuredBlock] = []
        global_order = 0

        for page in pages:
            page_text = self.clean_ingest_text(page.text)
            page_lines = page_text.splitlines()
            page_non_empty_lines = [line for line in page_lines if line.strip()]
            page_marker_consumed = False

            if include_page_markers:
                if page_non_empty_lines and self._is_page_marker_line(page_non_empty_lines[0]):
                    marker_text = page_non_empty_lines[0].strip()
                    page_lines = self._remove_first_matching_line(page_lines, marker_text)
                    page_marker_consumed = True
                else:
                    marker_text = str(page.page_label or page.page_number)

                blocks.append(
                    StructuredBlock(
                        block_id=str(uuid4()),
                        block_type="page_marker",
                        text=marker_text,
                        page_number=page.page_number,
                        page_label=page.page_label,
                        order=global_order,
                        section_path=list(heading_stack),
                        metadata={
                            "page_boxes": page.metadata.get("page_boxes", []),
                            "page_tables": page.metadata.get("tables", []),
                            "page_images": page.metadata.get("images", []),
                            "page_marker_detected": page_marker_consumed,
                            "source_page_count": page.metadata.get("source_page_count"),
                        },
                    )
                )
                global_order += 1

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
                        root_title=title,
                    )
                    blocks.append(
                        StructuredBlock(
                            block_id=str(uuid4()),
                            block_type="section_header",
                            text=heading_match.group("title").strip(),
                            page_number=page.page_number,
                            page_label=page.page_label,
                            order=global_order,
                            section_path=list(heading_stack),
                            metadata={"heading_level": len(heading_match.group("level"))},
                        )
                    )
                    global_order += 1
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
                    blocks.append(
                        self._make_block(
                            block_type="code",
                            text="\n".join(block_lines).strip(),
                            page=page,
                            order=global_order,
                            section_path=heading_stack,
                        )
                    )
                    global_order += 1
                    continue

                if self._is_caption_line(stripped):
                    blocks.append(
                        self._make_block(
                            block_type="caption",
                            text=stripped,
                            page=page,
                            order=global_order,
                            section_path=heading_stack,
                            extra_metadata=self._extract_caption_metadata(stripped),
                        )
                    )
                    global_order += 1
                    index += 1
                    continue

                image_metadata = self._extract_image_metadata(stripped)
                if image_metadata is not None:
                    image_text = self._render_image_text(image_metadata)
                    blocks.append(
                        self._make_block(
                            block_type="image",
                            text=image_text,
                            page=page,
                            order=global_order,
                            section_path=heading_stack,
                            extra_metadata=image_metadata,
                        )
                    )
                    global_order += 1
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
                        self._make_block(
                            block_type="table",
                            text=table_text,
                            page=page,
                            order=global_order,
                            section_path=heading_stack,
                            extra_metadata=table_metadata | {"raw_table_text": "\n".join(block_lines).strip()},
                        )
                    )
                    global_order += 1
                    continue

                if _LIST_ITEM_PATTERN.match(stripped):
                    block_lines = [line]
                    index += 1
                    while index < len(page_lines):
                        candidate = page_lines[index].strip()
                        if not candidate:
                            break
                        if (
                            _HEADING_PATTERN.match(candidate)
                            or _FENCE_PATTERN.match(candidate)
                            or self._is_table_row(candidate)
                            or self._extract_image_metadata(candidate) is not None
                            or self._is_caption_line(candidate)
                        ):
                            break
                        if _LIST_ITEM_PATTERN.match(candidate) or page_lines[index].startswith((" ", "\t")):
                            block_lines.append(page_lines[index])
                            index += 1
                            continue
                        break
                    blocks.append(
                        self._make_block(
                            block_type="list",
                            text="\n".join(block_lines).strip(),
                            page=page,
                            order=global_order,
                            section_path=heading_stack,
                        )
                    )
                    global_order += 1
                    continue

                block_lines = [line]
                index += 1
                while index < len(page_lines):
                    candidate = page_lines[index].strip()
                    if not candidate:
                        break
                    if (
                        _HEADING_PATTERN.match(candidate)
                        or _FENCE_PATTERN.match(candidate)
                        or self._is_table_row(candidate)
                        or _LIST_ITEM_PATTERN.match(candidate)
                        or self._extract_image_metadata(candidate) is not None
                        or self._is_caption_line(candidate)
                    ):
                        break
                    block_lines.append(page_lines[index])
                    index += 1
                blocks.append(
                    self._make_block(
                        block_type="paragraph",
                        text="\n".join(block_lines).strip(),
                        page=page,
                        order=global_order,
                        section_path=heading_stack,
                    )
                )
                global_order += 1

        self._enhance_relationships(blocks)
        return blocks

    def _make_block(
        self,
        *,
        block_type: str,
        text: str,
        page: ParsedPage,
        order: int,
        section_path: list[str],
        extra_metadata: dict[str, Any] | None = None,
    ) -> StructuredBlock:
        metadata = {
            "page_boxes": page.metadata.get("page_boxes", []),
            "source_page_count": page.metadata.get("source_page_count"),
            "page_label": page.page_label,
            "related_block_ids": [],
            "references_figures": [],
            "references_tables": [],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return StructuredBlock(
            block_id=str(uuid4()),
            block_type=block_type,
            text=text,
            page_number=page.page_number,
            page_label=page.page_label,
            order=order,
            section_path=list(section_path),
            metadata=metadata,
        )

    def _enhance_relationships(self, blocks: list[StructuredBlock]) -> None:
        captions = [block for block in blocks if block.block_type == "caption"]
        for block in blocks:
            if block.block_type not in {"image", "table"}:
                continue
            caption = self._find_best_caption(block, captions)
            if caption is None:
                continue
            block.metadata["caption_text"] = caption.text
            block.metadata["caption_type"] = caption.metadata.get("caption_type")
            block.metadata["caption_source_page"] = caption.page_number
            self._link_blocks(block, caption)

        figure_targets: dict[str, StructuredBlock] = {}
        table_targets: dict[str, StructuredBlock] = {}
        for block in blocks:
            identifier = self._extract_reference_identifier(block)
            if identifier is None:
                continue
            ref_kind, ref_number = identifier
            if block.block_type == "image" and ref_kind == "figure":
                figure_targets[ref_number] = block
            if block.block_type == "table" and ref_kind == "table":
                table_targets[ref_number] = block

        for block in blocks:
            if block.block_type not in {"paragraph", "list", "code", "caption"}:
                continue
            figure_refs: list[str] = []
            table_refs: list[str] = []
            for ref_kind, ref_number in self._extract_references(block.text):
                if ref_kind == "figure":
                    figure_refs.append(ref_number)
                    target = figure_targets.get(ref_number)
                else:
                    table_refs.append(ref_number)
                    target = table_targets.get(ref_number)
                if target is not None:
                    self._link_blocks(block, target)
            if figure_refs:
                block.metadata["references_figures"] = sorted(set(figure_refs), key=int)
            if table_refs:
                block.metadata["references_tables"] = sorted(set(table_refs), key=int)

    def _find_best_caption(self, block: StructuredBlock, captions: list[StructuredBlock]) -> StructuredBlock | None:
        expected_type = "figure" if block.block_type == "image" else "table"
        best_caption: StructuredBlock | None = None
        best_distance = 10**9
        for caption in captions:
            if caption.page_number != block.page_number:
                continue
            caption_type = caption.metadata.get("caption_type")
            if caption_type and caption_type != expected_type:
                continue
            distance = abs(caption.order - block.order)
            if distance < best_distance:
                best_distance = distance
                best_caption = caption
        return best_caption

    def _extract_reference_identifier(self, block: StructuredBlock) -> tuple[str, str] | None:
        source_text = block.metadata.get("caption_text") or block.text
        match = _REFERENCE_PATTERN.search(source_text)
        if not match:
            return None
        ref_kind = "figure" if match.group("kind").lower().startswith("fig") else "table"
        return ref_kind, match.group("number")

    def _extract_references(self, text: str) -> list[tuple[str, str]]:
        references: list[tuple[str, str]] = []
        for match in _REFERENCE_PATTERN.finditer(text):
            ref_kind = "figure" if match.group("kind").lower().startswith("fig") else "table"
            references.append((ref_kind, match.group("number")))
        return references

    def _link_blocks(self, left: StructuredBlock, right: StructuredBlock) -> None:
        left_ids = set(left.metadata.get("related_block_ids", []))
        right_ids = set(right.metadata.get("related_block_ids", []))
        if right.block_id not in left_ids:
            left.metadata.setdefault("related_block_ids", []).append(right.block_id)
        if left.block_id not in right_ids:
            right.metadata.setdefault("related_block_ids", []).append(left.block_id)

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

        linearized_lines = [f"Table columns: {', '.join(headers)}"]
        for row_index, row in enumerate(data_rows, start=1):
            padded_row = row + [""] * (len(headers) - len(row))
            assignments = [
                f"{header}={value.strip()}"
                for header, value in zip(headers, padded_row, strict=False)
                if header.strip() and value.strip()
            ]
            if assignments:
                linearized_lines.append(f"Row {row_index}: {'; '.join(assignments)}")

        metadata = {
            "table_headers": headers,
            "table_row_count": len(data_rows),
        }
        return "\n".join(linearized_lines), metadata

    def _render_image_text(self, metadata: dict[str, Any]) -> str:
        image_lines = ["Image block"]
        if metadata.get("image_alt_text"):
            image_lines.append(f"Alt: {metadata['image_alt_text']}")
        if metadata.get("image_source"):
            image_lines.append(f"Source: {metadata['image_source']}")
        if metadata.get("image_title"):
            image_lines.append(f"Title: {metadata['image_title']}")
        return "\n".join(image_lines)

    def _extract_image_metadata(self, line: str) -> dict[str, Any] | None:
        markdown_match = _MD_IMAGE_PATTERN.match(line)
        if markdown_match:
            metadata = {
                "image_alt_text": markdown_match.group("alt").strip() or None,
                "image_source": markdown_match.group("src").strip(),
                "image_title": markdown_match.group("title").strip() if markdown_match.group("title") else None,
            }
            return {key: value for key, value in metadata.items() if value}

        html_match = _HTML_IMG_PATTERN.match(line)
        if html_match:
            attrs = {
                match.group("name").lower(): match.group("value").strip()
                for match in _HTML_ATTR_PATTERN.finditer(html_match.group("attrs"))
            }
            metadata = {
                "image_alt_text": attrs.get("alt"),
                "image_source": attrs.get("src"),
                "image_title": attrs.get("title"),
            }
            filtered = {key: value for key, value in metadata.items() if value}
            return filtered or None

        return None

    def _extract_caption_metadata(self, line: str) -> dict[str, Any]:
        match = _CAPTION_PATTERN.match(line.strip())
        if not match:
            return {}
        raw_kind = (match.group("kind") or "").lower()
        caption_type = "table" if raw_kind.startswith("table") or raw_kind == "表" else "figure"
        metadata = {
            "caption_text": line.strip(),
            "caption_type": caption_type,
        }
        if match.group("number"):
            metadata["caption_number"] = match.group("number")
        return metadata

    def _is_caption_line(self, line: str) -> bool:
        return _CAPTION_PATTERN.match(line) is not None

    def _is_page_marker_line(self, line: str) -> bool:
        return _PAGE_MARKER_PATTERN.match(line.strip()) is not None

    def _remove_first_matching_line(self, lines: list[str], target: str) -> list[str]:
        removed = False
        remaining: list[str] = []
        for line in lines:
            if not removed and line.strip() == target:
                removed = True
                continue
            remaining.append(line)
        return remaining

    def _update_heading_stack(
        self,
        *,
        heading_stack: list[str],
        level: int,
        title: str,
        root_title: str | None,
    ) -> list[str]:
        if level <= 1:
            return [title]
        if not heading_stack:
            return [root_title, title] if root_title and root_title != title else [title]
        parent_depth = min(level - 1, len(heading_stack))
        updated_stack = [*heading_stack[:parent_depth], title]
        return [item for item in updated_stack if item]

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return _TABLE_ROW_PATTERN.match(line.strip()) is not None

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [cell.strip() for cell in stripped.split("|")]

