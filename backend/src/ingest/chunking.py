from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.src.core.config import Settings
from backend.src.core.models import IngestedChunk, NormalizedDocument

_HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S+")
_LIST_ITEM_CAPTURE_PATTERN = re.compile(r"^(?P<indent>\s*)(?:[-*+]|\d+[.)])\s+(?P<text>\S.*)$")
_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_TABLE_ALIGNMENT_PATTERN = re.compile(r"^:?-{3,}:?$")
_MD_IMAGE_PATTERN = re.compile(
    r'^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+[\"\'](?P<title>.*?)[\"\'])?\)$'
)
_HTML_IMG_PATTERN = re.compile(r"^<img\s+(?P<attrs>[^>]+?)\s*/?>$", re.IGNORECASE)
_HTML_ATTR_PATTERN = re.compile(r'(?P<name>[a-zA-Z_:][a-zA-Z0-9:._-]*)\s*=\s*[\"\'](?P<value>.*?)[\"\']')
_CAPTION_PATTERN = re.compile(r"^(?:figure|fig\.|image|caption|图示|图片|图)\s*[:.\-]?\s*(?P<caption>.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class _StructuredBlock:
    text: str
    section_path: list[str]
    block_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunkingService:
    def __init__(self, settings: Settings, embeddings: object) -> None:
        self._settings = settings
        self._embeddings = embeddings
        model_max_length = int(getattr(embeddings, "max_seq_length", settings.chunk_max_word_pieces) or settings.chunk_max_word_pieces)
        self._chunk_max_word_pieces = min(settings.chunk_max_word_pieces, model_max_length)
        self._chunk_overlap_word_pieces = min(
            settings.chunk_overlap_word_pieces,
            max(0, self._chunk_max_word_pieces // 4),
        )
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_max_word_pieces,
            chunk_overlap=self._chunk_overlap_word_pieces,
            length_function=self._count_word_pieces,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_documents(self, documents: Iterable[NormalizedDocument]) -> list[IngestedChunk]:
        chunks: list[IngestedChunk] = []
        for document in documents:
            if not document.content.strip():
                continue
            for block in self._split_structured_blocks(document):
                for chunk_text in self._split_oversized_block(block):
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    chunks.append(self._build_chunk(document=document, block=block, chunk_text=chunk_text))
        return chunks

    def _build_chunk(
        self,
        *,
        document: NormalizedDocument,
        block: _StructuredBlock,
        chunk_text: str,
    ) -> IngestedChunk:
        metadata = {
            **document.metadata,
            **block.metadata,
            "doc_id": document.doc_id,
            "source_name": document.source_name,
            "source_type": document.source_type.value,
            "source_uri_or_path": document.source_uri_or_path,
            "page_or_section": document.page_or_section,
            "page": document.page,
            "title": document.title,
            "section_path": block.section_path,
            "section_depth": len(block.section_path),
            "doc_type": (document.doc_type or document.source_type).value,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "block_type": block.block_type,
            "chunk_wordpiece_count": self._count_word_pieces(chunk_text),
        }
        return IngestedChunk(
            chunk_id=str(uuid4()),
            doc_id=document.doc_id,
            source_name=document.source_name,
            source_uri_or_path=document.source_uri_or_path,
            page_or_section=document.page_or_section,
            page=document.page,
            title=document.title,
            section_path=block.section_path,
            doc_type=document.doc_type or document.source_type,
            updated_at=document.updated_at,
            content=chunk_text,
            metadata=metadata,
        )

    def _split_structured_blocks(self, document: NormalizedDocument) -> list[_StructuredBlock]:
        lines = document.content.splitlines()
        blocks: list[_StructuredBlock] = []
        heading_stack = list(document.section_path)
        index = 0

        while index < len(lines):
            line = lines[index]
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
                    document=document,
                )
                index += 1
                continue

            if _FENCE_PATTERN.match(stripped):
                fence = _FENCE_PATTERN.match(stripped).group(1)
                block_lines = [line]
                index += 1
                while index < len(lines):
                    block_lines.append(lines[index])
                    if lines[index].strip().startswith(fence):
                        index += 1
                        break
                    index += 1
                blocks.append(self._make_block(block_lines, heading_stack, document, "code"))
                continue

            image_metadata = self._parse_image_line(stripped)
            if image_metadata is not None:
                block_lines = [line]
                index += 1
                if index < len(lines):
                    caption_match = _CAPTION_PATTERN.match(lines[index].strip())
                    if caption_match:
                        image_metadata["image_caption"] = caption_match.group("caption").strip()
                        block_lines.append(lines[index])
                        index += 1
                blocks.append(self._make_block(block_lines, heading_stack, document, "image", image_metadata))
                continue

            if self._is_table_row(stripped):
                block_lines = [line]
                index += 1
                while index < len(lines) and self._is_table_row(lines[index].strip()):
                    block_lines.append(lines[index])
                    index += 1
                blocks.append(self._make_block(block_lines, heading_stack, document, "table"))
                continue

            if _LIST_ITEM_PATTERN.match(stripped):
                block_lines = [line]
                index += 1
                while index < len(lines):
                    candidate = lines[index].strip()
                    if not candidate:
                        index += 1
                        break
                    if (
                        _HEADING_PATTERN.match(candidate)
                        or _FENCE_PATTERN.match(candidate)
                        or self._is_table_row(candidate)
                        or self._parse_image_line(candidate) is not None
                    ):
                        break
                    if _LIST_ITEM_PATTERN.match(candidate) or lines[index].startswith((" ", "\t")):
                        block_lines.append(lines[index])
                        index += 1
                        continue
                    break
                blocks.append(self._make_block(block_lines, heading_stack, document, "list"))
                continue

            block_lines = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate:
                    index += 1
                    break
                if (
                    _HEADING_PATTERN.match(candidate)
                    or _FENCE_PATTERN.match(candidate)
                    or self._is_table_row(candidate)
                    or _LIST_ITEM_PATTERN.match(candidate)
                    or self._parse_image_line(candidate) is not None
                ):
                    break
                block_lines.append(lines[index])
                index += 1
            blocks.append(self._make_block(block_lines, heading_stack, document, "paragraph"))

        if not blocks:
            blocks.append(
                _StructuredBlock(
                    text=document.content,
                    section_path=list(document.section_path) or [document.title or document.source_name],
                    block_type="paragraph",
                )
            )
        return blocks

    def _make_block(
        self,
        block_lines: list[str],
        heading_stack: list[str],
        document: NormalizedDocument,
        block_type: str,
        block_metadata: dict[str, Any] | None = None,
    ) -> _StructuredBlock:
        section_path = list(heading_stack) or list(document.section_path) or [document.title or document.source_name]
        raw_text = "\n".join(block_lines).strip()
        normalized_text, normalized_metadata = self._normalize_block(
            block_type=block_type,
            text=raw_text,
            metadata=block_metadata or {},
        )
        return _StructuredBlock(
            text=normalized_text,
            section_path=section_path,
            block_type=block_type,
            metadata=normalized_metadata,
        )

    def _normalize_block(
        self,
        *,
        block_type: str,
        text: str,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if block_type == "table":
            return self._linearize_table_block(text)
        if block_type == "list":
            return self._linearize_list_block(text)
        if block_type == "image":
            return self._linearize_image_block(text, metadata)
        return text, dict(metadata)

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

    def _linearize_list_block(self, text: str) -> tuple[str, dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            match = _LIST_ITEM_CAPTURE_PATTERN.match(line)
            if match:
                indent = match.group("indent").replace("\t", "    ")
                level = max(1, (len(indent) // 2) + 1)
                items.append({"level": level, "parts": [match.group("text").strip()]})
                continue
            if items:
                items[-1]["parts"].append(line.strip())

        if not items:
            return text, {}

        normalized_lines = [
            f"Level {item['level']}: {' '.join(part for part in item['parts'] if part).strip()}"
            for item in items
        ]
        metadata = {
            "list_item_count": len(items),
            "list_max_level": max(int(item["level"]) for item in items),
        }
        return "\n".join(normalized_lines), metadata

    def _linearize_image_block(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        normalized_metadata = dict(metadata)
        image_lines = ["Image block"]
        if normalized_metadata.get("image_alt_text"):
            image_lines.append(f"Alt: {normalized_metadata['image_alt_text']}")
        if normalized_metadata.get("image_source"):
            image_lines.append(f"Source: {normalized_metadata['image_source']}")
        if normalized_metadata.get("image_title"):
            image_lines.append(f"Title: {normalized_metadata['image_title']}")
        if normalized_metadata.get("image_caption"):
            image_lines.append(f"Caption: {normalized_metadata['image_caption']}")
        if len(image_lines) == 1:
            image_lines.append(text)
        return "\n".join(image_lines), normalized_metadata

    def _split_oversized_block(self, block: _StructuredBlock) -> list[str]:
        if self._count_word_pieces(block.text) <= self._chunk_max_word_pieces:
            return [block.text]

        if block.block_type == "paragraph":
            try:
                semantic_splitter = SemanticChunker(self._embeddings)
                semantic_parts = [
                    split_document.page_content
                    for split_document in semantic_splitter.split_documents([Document(page_content=block.text, metadata={})])
                    if split_document.page_content.strip()
                ]
                if semantic_parts:
                    return self._enforce_budget(semantic_parts)
            except Exception:
                pass

        return self._fallback_splitter.split_text(block.text)

    def _enforce_budget(self, chunk_texts: list[str]) -> list[str]:
        bounded_chunks: list[str] = []
        for chunk_text in chunk_texts:
            if self._count_word_pieces(chunk_text) <= self._chunk_max_word_pieces:
                bounded_chunks.append(chunk_text)
            else:
                bounded_chunks.extend(self._fallback_splitter.split_text(chunk_text))
        return bounded_chunks

    def _count_word_pieces(self, text: str) -> int:
        counter = getattr(self._embeddings, "count_word_pieces", None)
        if callable(counter):
            return max(1, int(counter(text)))
        return max(1, len(_TOKEN_PATTERN.findall(text)))

    def _update_heading_stack(
        self,
        *,
        heading_stack: list[str],
        level: int,
        title: str,
        document: NormalizedDocument,
    ) -> list[str]:
        if level <= 1 or not heading_stack:
            return [title]

        parent_depth = min(level - 1, len(heading_stack))
        updated_stack = [*heading_stack[:parent_depth], title]
        if not updated_stack:
            return [document.title or document.source_name, title]
        return updated_stack

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return line.startswith("|") and line.endswith("|") and line.count("|") >= 2

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [cell.strip() for cell in stripped.split("|")]

    def _parse_image_line(self, line: str) -> dict[str, Any] | None:
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


SemanticChunkingService = StructureAwareChunkingService
