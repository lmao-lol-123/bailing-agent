from __future__ import annotations

import hashlib
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
from backend.src.ingest.parent_store import JsonParentStore, ParentRecord

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_TABLE_ROW_PATTERN = re.compile(r"^\s*Row\s+(\d+)\s*:", re.IGNORECASE)
_LIST_LEVEL_PATTERN = re.compile(r"^\s*Level\s+(\d+)\s*:", re.IGNORECASE)
_CODE_UNIT_PATTERN = re.compile(
    r"^\s*(def\s+\w+|class\s+\w+|function\s+\w+|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|public\s+|private\s+)",
    re.IGNORECASE,
)
_SKIP_LAYOUT_ROLES = {"header", "footer"}
_SKIP_BLOCK_TYPES = {"page_marker"}
_PARENT_RELATION_TYPES = {
    "has_caption",
    "caption_of",
    "references",
    "referenced_by",
    "formula_explains",
    "table_row_context",
    "continued_from",
}
_MAX_RELATED_PARENT_BLOCKS = 3
_PARENT_WORDPIECE_MULTIPLIER = 4


@dataclass(frozen=True)
class _ChunkSourceBlock:
    doc_id: str
    text: str
    section_path: list[str]
    block_type: str
    source_block_id: str
    layout_role: str
    block_order: int
    page: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SourceEntry:
    document: NormalizedDocument
    source_block: _ChunkSourceBlock


@dataclass(frozen=True)
class _ChildSplit:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ParentAssembly:
    parent_content: str
    blocks: list[_ChunkSourceBlock]
    relation_types: list[str]
    strategy: str
    assembled: bool

    @property
    def source_block_ids(self) -> list[str]:
        return [block.source_block_id for block in self.blocks]


class _ChunkAssemblyContext:
    def __init__(self, entries: list[_SourceEntry]) -> None:
        self.entries = entries
        self._blocks_by_key = {(entry.source_block.doc_id, entry.source_block.source_block_id): entry.source_block for entry in entries}

    def get_block(self, *, doc_id: str, block_id: str) -> _ChunkSourceBlock | None:
        return self._blocks_by_key.get((doc_id, block_id))


class StructureAwareChunkingService:
    def __init__(self, settings: Settings, embeddings: object, persist_parent_store: bool = False) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._parent_store = JsonParentStore(settings.processed_directory) if persist_parent_store else None
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
        entries = [
            _SourceEntry(document=document, source_block=self._build_source_block(document=document, fallback_order=fallback_order))
            for fallback_order, document in enumerate(documents)
        ]
        context = _ChunkAssemblyContext(entries)
        chunks: list[IngestedChunk] = []
        parent_records_by_doc: dict[str, dict[str, ParentRecord]] = {}

        for entry in context.entries:
            document = entry.document
            source_block = entry.source_block
            if self._should_skip_source_block(source_block):
                continue

            child_splits = [child_split for child_split in self._split_parent_to_children(source_block) if child_split.text.strip()]
            if not child_splits:
                continue

            assembly = self._assemble_parent_for_block(source_block=source_block, context=context)
            parent_chunk_id = self._build_assembled_parent_chunk_id(document=document, assembly=assembly)
            parent_metadata = self._build_parent_metadata(
                parent_chunk_id=parent_chunk_id,
                source_block=source_block,
                assembly=assembly,
                child_count=len(child_splits),
            )
            parent_records_by_doc.setdefault(document.doc_id, {})[parent_chunk_id] = self._build_parent_record(
                document=document,
                assembly=assembly,
                parent_metadata=parent_metadata,
            )

            for child_index, child_split in enumerate(child_splits):
                chunks.append(
                    self._build_child_chunk(
                        document=document,
                        source_block=source_block,
                        child_split=child_split,
                        assembly=assembly,
                        parent_metadata=parent_metadata,
                        child_index=child_index,
                        child_count=len(child_splits),
                    )
                )

        if self._parent_store is not None:
            for doc_id, records_by_parent_id in parent_records_by_doc.items():
                self._parent_store.save_records(doc_id=doc_id, records=list(records_by_parent_id.values()))
        return chunks

    def _build_source_block(self, *, document: NormalizedDocument, fallback_order: int) -> _ChunkSourceBlock:
        metadata = dict(document.metadata or {})
        block_type = str(metadata.get("block_type") or "paragraph")
        layout_role = str(metadata.get("layout_role") or "body")
        block_order = self._coerce_int(metadata.get("block_order", metadata.get("order", fallback_order)), fallback_order)
        section_path = list(document.section_path) or self._coerce_section_path(metadata.get("section_path"))
        if not section_path:
            section_path = [document.title or document.source_name]

        source_block_id = str(metadata.get("block_id") or self._fallback_block_id(document=document, block_order=block_order))
        normalized_metadata = {
            **metadata,
            "block_type": block_type,
            "layout_role": layout_role,
            "block_order": block_order,
            "block_id": source_block_id,
            "graph_edges": metadata.get("graph_edges", []),
            "graph_neighbors": metadata.get("graph_neighbors", []),
        }
        return _ChunkSourceBlock(
            doc_id=document.doc_id,
            text=document.content.strip(),
            section_path=section_path,
            block_type=block_type,
            source_block_id=source_block_id,
            layout_role=layout_role,
            block_order=block_order,
            page=document.page or document.page_or_section,
            title=document.title,
            metadata=normalized_metadata,
        )

    def _should_skip_source_block(self, source_block: _ChunkSourceBlock) -> bool:
        if not source_block.text.strip():
            return True
        if source_block.block_type in _SKIP_BLOCK_TYPES:
            return True
        if source_block.layout_role in _SKIP_LAYOUT_ROLES:
            return True
        return bool(source_block.metadata.get("excluded_from_body"))

    def _assemble_parent_for_block(self, *, source_block: _ChunkSourceBlock, context: _ChunkAssemblyContext) -> _ParentAssembly:
        related_blocks, relation_types = self._select_related_blocks(source_block=source_block, context=context)
        if not related_blocks:
            return _ParentAssembly(
                parent_content=source_block.text,
                blocks=[source_block],
                relation_types=[],
                strategy="single_block",
                assembled=False,
            )

        ordered_blocks = self._sort_parent_blocks([source_block, *related_blocks])
        return _ParentAssembly(
            parent_content=self._format_parent_content(source_block=source_block, related_blocks=ordered_blocks),
            blocks=ordered_blocks,
            relation_types=relation_types,
            strategy="relation_limited",
            assembled=True,
        )

    def _select_related_blocks(
        self,
        *,
        source_block: _ChunkSourceBlock,
        context: _ChunkAssemblyContext,
    ) -> tuple[list[_ChunkSourceBlock], list[str]]:
        selected_blocks: list[_ChunkSourceBlock] = []
        selected_relation_types: list[str] = []
        seen_block_ids = {source_block.source_block_id}
        parent_limit = self._chunk_max_word_pieces * _PARENT_WORDPIECE_MULTIPLIER

        edges = self._coerce_graph_edges(source_block.metadata.get("graph_edges"))
        edges.sort(key=lambda edge: (-float(edge.get("weight", 0.0)), str(edge.get("type", ""))))
        for edge in edges:
            relation_type = str(edge.get("type") or "")
            target_block_id = str(edge.get("target_block_id") or "")
            if relation_type not in _PARENT_RELATION_TYPES or not target_block_id or target_block_id in seen_block_ids:
                continue
            target_block = context.get_block(doc_id=source_block.doc_id, block_id=target_block_id)
            if target_block is None or self._should_skip_source_block(target_block):
                continue

            trial_blocks = self._sort_parent_blocks([source_block, *selected_blocks, target_block])
            trial_content = self._format_parent_content(source_block=source_block, related_blocks=trial_blocks)
            if self._count_word_pieces(trial_content) > parent_limit:
                continue

            selected_blocks.append(target_block)
            selected_relation_types.append(relation_type)
            seen_block_ids.add(target_block_id)
            if len(selected_blocks) >= _MAX_RELATED_PARENT_BLOCKS:
                break

        return selected_blocks, selected_relation_types

    def _split_parent_to_children(self, source_block: _ChunkSourceBlock) -> list[_ChildSplit]:
        if source_block.block_type == "table":
            table_splits = self._split_table_block(source_block)
            if table_splits:
                return table_splits
        if source_block.block_type == "list":
            list_splits = self._split_list_block(source_block)
            if list_splits:
                return list_splits
        if source_block.block_type == "code":
            code_splits = self._split_code_block(source_block)
            if code_splits:
                return code_splits

        if self._count_word_pieces(source_block.text) <= self._chunk_max_word_pieces:
            return [_ChildSplit(text=source_block.text, metadata={"child_split_strategy": "single_block"})]

        if source_block.block_type == "paragraph":
            try:
                semantic_splitter = SemanticChunker(self._embeddings)
                semantic_parts = [
                    split_document.page_content
                    for split_document in semantic_splitter.split_documents([Document(page_content=source_block.text, metadata={})])
                    if split_document.page_content.strip()
                ]
                if semantic_parts:
                    return [_ChildSplit(text=text, metadata={"child_split_strategy": "semantic"}) for text in self._enforce_budget(semantic_parts)]
            except Exception:
                pass

        return self._split_with_fallback(source_block)

    def _split_table_block(self, source_block: _ChunkSourceBlock) -> list[_ChildSplit]:
        header_line, rows = self._parse_table_lines(source_block.text)
        if not header_line or not rows:
            if self._count_word_pieces(source_block.text) <= self._chunk_max_word_pieces:
                return [_ChildSplit(text=source_block.text, metadata={"child_split_strategy": "single_block"})]
            return []

        splits: list[_ChildSplit] = []
        current_rows: list[tuple[int, str]] = []
        for row_number, row_text in rows:
            candidate_rows = [*current_rows, (row_number, row_text)]
            candidate_text = "\n".join([header_line, *[row for _, row in candidate_rows]])
            if self._count_word_pieces(candidate_text) > self._chunk_max_word_pieces and current_rows:
                splits.append(self._build_table_child_split(header_line=header_line, rows=current_rows))
                current_rows = [(row_number, row_text)]
                continue
            if self._count_word_pieces(candidate_text) > self._chunk_max_word_pieces:
                return self._split_with_fallback(source_block)
            current_rows = candidate_rows

        if current_rows:
            splits.append(self._build_table_child_split(header_line=header_line, rows=current_rows))
        return splits

    def _split_list_block(self, source_block: _ChunkSourceBlock) -> list[_ChildSplit]:
        groups = self._parse_list_lines(source_block.text)
        if not groups:
            if self._count_word_pieces(source_block.text) <= self._chunk_max_word_pieces:
                return [_ChildSplit(text=source_block.text, metadata={"child_split_strategy": "single_block"})]
            return []

        splits: list[_ChildSplit] = []
        current_groups: list[tuple[int, list[str]]] = []
        for item_number, lines in groups:
            candidate_groups = [*current_groups, (item_number, lines)]
            candidate_text = "\n".join(line for _, group_lines in candidate_groups for line in group_lines)
            if self._count_word_pieces(candidate_text) > self._chunk_max_word_pieces and current_groups:
                splits.append(self._build_list_child_split(current_groups))
                current_groups = [(item_number, lines)]
                continue
            if self._count_word_pieces(candidate_text) > self._chunk_max_word_pieces:
                return self._split_with_fallback(source_block)
            current_groups = candidate_groups

        if current_groups:
            splits.append(self._build_list_child_split(current_groups))
        return splits

    def _split_code_block(self, source_block: _ChunkSourceBlock) -> list[_ChildSplit]:
        units = self._parse_code_units(source_block.text)
        if not units:
            if self._count_word_pieces(source_block.text) <= self._chunk_max_word_pieces:
                return [_ChildSplit(text=source_block.text, metadata={"child_split_strategy": "single_block"})]
            return []

        splits: list[_ChildSplit] = []
        for unit_number, unit_text in units:
            if self._count_word_pieces(unit_text) > self._chunk_max_word_pieces:
                return self._split_with_fallback(source_block)
            splits.append(self._build_code_child_split([(unit_number, unit_text)]))
        return splits

    def _split_with_fallback(self, source_block: _ChunkSourceBlock) -> list[_ChildSplit]:
        return [
            _ChildSplit(text=text, metadata={"child_split_strategy": "fallback"})
            for text in self._fallback_splitter.split_text(source_block.text)
            if text.strip()
        ]

    def _build_child_chunk(
        self,
        *,
        document: NormalizedDocument,
        source_block: _ChunkSourceBlock,
        child_split: _ChildSplit,
        assembly: _ParentAssembly,
        parent_metadata: dict[str, Any],
        child_index: int,
        child_count: int,
    ) -> IngestedChunk:
        chunk_id = str(uuid4())
        child_text = child_split.text
        retrieval_text, retrieval_strategy = self._build_retrieval_text(source_block=source_block, child_text=child_text, assembly=assembly)
        metadata = {
            **document.metadata,
            **source_block.metadata,
            **parent_metadata,
            **child_split.metadata,
            "chunk_id": chunk_id,
            "doc_id": document.doc_id,
            "source_name": document.source_name,
            "source_type": document.source_type.value,
            "source_uri_or_path": document.source_uri_or_path,
            "page_or_section": document.page_or_section,
            "page": document.page,
            "title": document.title,
            "section_path": source_block.section_path,
            "section_depth": len(source_block.section_path),
            "doc_type": (document.doc_type or document.source_type).value,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "block_type": source_block.block_type,
            "chunk_level": "child",
            "source_block_id": source_block.source_block_id,
            "source_block_type": source_block.block_type,
            "parent_block_id": source_block.source_block_id,
            "child_index": child_index,
            "child_count": child_count,
            "child_content": child_text,
            "original_child_content": child_text,
            "retrieval_text": retrieval_text,
            "retrieval_text_strategy": retrieval_strategy,
            "content_role": "retrieval_text",
            "returns_parent_content": True,
            "chunk_wordpiece_count": self._count_word_pieces(child_text),
            "retrieval_wordpiece_count": self._count_word_pieces(retrieval_text),
        }
        return IngestedChunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            source_name=document.source_name,
            source_uri_or_path=document.source_uri_or_path,
            page_or_section=document.page_or_section,
            page=document.page,
            title=document.title,
            section_path=source_block.section_path,
            doc_type=document.doc_type or document.source_type,
            updated_at=document.updated_at,
            content=retrieval_text,
            metadata=metadata,
        )

    def _build_parent_metadata(
        self,
        *,
        parent_chunk_id: str,
        source_block: _ChunkSourceBlock,
        assembly: _ParentAssembly,
        child_count: int,
    ) -> dict[str, Any]:
        parent_wordpiece_count = self._count_word_pieces(assembly.parent_content)
        metadata: dict[str, Any] = {
            "parent_chunk_id": parent_chunk_id,
            "parent_block_id": source_block.source_block_id,
            "parent_wordpiece_count": parent_wordpiece_count,
            "parent_content_hash": hashlib.sha256(assembly.parent_content.encode("utf-8")).hexdigest(),
            "parent_content_preview": self._build_parent_preview(assembly.parent_content),
            "parent_assembly_strategy": assembly.strategy,
            "parent_source_block_ids": assembly.source_block_ids,
            "parent_relation_types": assembly.relation_types,
            "parent_block_count": len(assembly.blocks),
            "assembled_parent": assembly.assembled,
            "parent_store_ref": f"{source_block.doc_id}:{parent_chunk_id}",
            "parent_stored": self._parent_store is not None,
        }
        if parent_wordpiece_count <= self._chunk_max_word_pieces * 2:
            metadata["parent_content"] = assembly.parent_content
            metadata["parent_content_available"] = "metadata"
        else:
            metadata["parent_content_available"] = "store" if self._parent_store is not None else "preview_only"
        return metadata

    def _build_parent_record(
        self,
        *,
        document: NormalizedDocument,
        assembly: _ParentAssembly,
        parent_metadata: dict[str, Any],
    ) -> ParentRecord:
        return ParentRecord(
            parent_chunk_id=str(parent_metadata["parent_chunk_id"]),
            doc_id=document.doc_id,
            source_block_ids=assembly.source_block_ids,
            parent_content=assembly.parent_content,
            parent_wordpiece_count=int(parent_metadata["parent_wordpiece_count"]),
            parent_content_hash=str(parent_metadata["parent_content_hash"]),
            section_path=assembly.blocks[0].section_path if assembly.blocks else list(document.section_path),
            page=document.page or document.page_or_section,
            title=document.title,
            metadata={
                "parent_assembly_strategy": parent_metadata["parent_assembly_strategy"],
                "parent_relation_types": parent_metadata["parent_relation_types"],
                "parent_block_count": parent_metadata["parent_block_count"],
            },
        )

    def _build_assembled_parent_chunk_id(self, *, document: NormalizedDocument, assembly: _ParentAssembly) -> str:
        raw_value = f"{document.doc_id}:{':'.join(assembly.source_block_ids)}:parent"
        digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:16]
        return f"parent-{digest}"

    def _fallback_block_id(self, *, document: NormalizedDocument, block_order: int) -> str:
        page = document.page or document.page_or_section or "none"
        return f"{document.doc_id}:{page}:{block_order}"

    def _build_parent_preview(self, text: str, limit: int = 240) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _format_parent_content(self, *, source_block: _ChunkSourceBlock, related_blocks: list[_ChunkSourceBlock]) -> str:
        if len(related_blocks) == 1 and related_blocks[0].source_block_id == source_block.source_block_id:
            return source_block.text
        parts = []
        for block in related_blocks:
            label = block.block_type.replace("_", " ")
            parts.append(f"[{label}]\n{block.text.strip()}")
        return "\n\n".join(parts).strip()

    def _sort_parent_blocks(self, blocks: list[_ChunkSourceBlock]) -> list[_ChunkSourceBlock]:
        return sorted(blocks, key=lambda block: (self._page_sort_value(block.page), block.block_order, block.source_block_id))

    def _build_table_child_split(self, *, header_line: str, rows: list[tuple[int, str]]) -> _ChildSplit:
        row_numbers = [row_number for row_number, _ in rows]
        return _ChildSplit(
            text="\n".join([header_line, *[row_text for _, row_text in rows]]),
            metadata={
                "child_split_strategy": "table_rows",
                "table_header_repeated": True,
                "table_row_start": min(row_numbers),
                "table_row_end": max(row_numbers),
            },
        )

    def _build_list_child_split(self, groups: list[tuple[int, list[str]]]) -> _ChildSplit:
        item_numbers = [item_number for item_number, _ in groups]
        return _ChildSplit(
            text="\n".join(line for _, lines in groups for line in lines),
            metadata={
                "child_split_strategy": "list_items",
                "list_item_start": min(item_numbers),
                "list_item_end": max(item_numbers),
            },
        )

    def _build_code_child_split(self, units: list[tuple[int, str]]) -> _ChildSplit:
        unit_numbers = [unit_number for unit_number, _ in units]
        return _ChildSplit(
            text="\n\n".join(unit_text for _, unit_text in units),
            metadata={
                "child_split_strategy": "code_units",
                "code_unit_start": min(unit_numbers),
                "code_unit_end": max(unit_numbers),
            },
        )

    def _parse_table_lines(self, text: str) -> tuple[str | None, list[tuple[int, str]]]:
        header_line: str | None = None
        rows: list[tuple[int, str]] = []
        for line in [line.strip() for line in text.splitlines() if line.strip()]:
            if header_line is None and line.lower().startswith("table columns:"):
                header_line = line
                continue
            match = _TABLE_ROW_PATTERN.match(line)
            if match:
                rows.append((int(match.group(1)), line))
        return header_line, rows

    def _parse_list_lines(self, text: str) -> list[tuple[int, list[str]]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        groups: list[tuple[int, list[str]]] = []
        current_lines: list[str] = []
        current_item = 0
        saw_level_marker = False
        for line in lines:
            match = _LIST_LEVEL_PATTERN.match(line)
            if match:
                saw_level_marker = True
                level = int(match.group(1))
                if level == 1:
                    if current_lines:
                        groups.append((current_item, current_lines))
                    current_item += 1
                    current_lines = [line]
                    continue
            if not current_lines:
                current_item += 1
            current_lines.append(line)

        if current_lines:
            groups.append((current_item, current_lines))
        if saw_level_marker:
            return groups
        return [(index, [line]) for index, line in enumerate(lines, start=1)]

    def _parse_code_units(self, text: str) -> list[tuple[int, str]]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) > 1:
            return [(index, paragraph) for index, paragraph in enumerate(paragraphs, start=1)]

        units: list[str] = []
        current_lines: list[str] = []
        for line in text.splitlines():
            if _CODE_UNIT_PATTERN.match(line) and current_lines:
                units.append("\n".join(current_lines).strip())
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            units.append("\n".join(current_lines).strip())
        return [(index, unit) for index, unit in enumerate(units, start=1) if unit]

    def _build_retrieval_text(self, *, source_block: _ChunkSourceBlock, child_text: str, assembly: _ParentAssembly) -> tuple[str, str]:
        if source_block.block_type == "table":
            return self._build_table_retrieval_text(source_block=source_block, child_text=child_text)
        if source_block.block_type == "image":
            return self._build_image_retrieval_text(source_block=source_block, child_text=child_text, assembly=assembly)
        if source_block.block_type == "formula":
            return self._build_formula_retrieval_text(source_block=source_block, child_text=child_text, assembly=assembly)
        return child_text, "original"

    def _build_table_retrieval_text(self, *, source_block: _ChunkSourceBlock, child_text: str) -> tuple[str, str]:
        metadata = source_block.metadata
        if not any(metadata.get(key) for key in ("caption_text", "table_headers", "table_row_count")):
            return child_text, "original"
        parts = [
            self._metadata_line("Title", source_block.title),
            self._metadata_line("Section", " > ".join(source_block.section_path)),
            self._metadata_line("Caption", metadata.get("caption_text")),
            self._metadata_line("Headers", metadata.get("table_headers")),
            self._metadata_line("Rows", metadata.get("table_row_count")),
            child_text,
        ]
        return self._bounded_retrieval_text(parts=parts, child_text=child_text, strategy="table_enhanced")

    def _build_image_retrieval_text(
        self,
        *,
        source_block: _ChunkSourceBlock,
        child_text: str,
        assembly: _ParentAssembly,
    ) -> tuple[str, str]:
        metadata = source_block.metadata
        parts = [
            self._metadata_line("Title", source_block.title),
            self._metadata_line("Section", " > ".join(source_block.section_path)),
            self._metadata_line("Image title", metadata.get("image_title")),
            self._metadata_line("Alt", metadata.get("image_alt_text")),
            self._metadata_line("Caption", metadata.get("caption_text")),
            self._metadata_line("Context", self._related_parent_preview(source_block=source_block, assembly=assembly)),
            child_text,
        ]
        return self._bounded_retrieval_text(parts=parts, child_text=child_text, strategy="image_enhanced")

    def _build_formula_retrieval_text(
        self,
        *,
        source_block: _ChunkSourceBlock,
        child_text: str,
        assembly: _ParentAssembly,
    ) -> tuple[str, str]:
        metadata = source_block.metadata
        parts = [
            self._metadata_line("Title", source_block.title),
            self._metadata_line("Section", " > ".join(source_block.section_path)),
            self._metadata_line("Caption", metadata.get("caption_text")),
            self._metadata_line("Formula", metadata.get("formula_linearized_text")),
            self._metadata_line("Symbols", metadata.get("formula_symbols")),
            self._metadata_line("Context", self._related_parent_preview(source_block=source_block, assembly=assembly)),
            child_text,
        ]
        return self._bounded_retrieval_text(parts=parts, child_text=child_text, strategy="formula_enhanced")

    def _bounded_retrieval_text(self, *, parts: list[str | None], child_text: str, strategy: str) -> tuple[str, str]:
        metadata_parts = [part for part in parts if part and part.strip() and part.strip() != child_text.strip()]
        if not metadata_parts:
            return child_text, "original"

        selected_metadata: list[str] = []
        for part in metadata_parts:
            candidate = "\n".join([*selected_metadata, part, child_text])
            if self._count_word_pieces(candidate) <= self._chunk_max_word_pieces:
                selected_metadata.append(part)
        retrieval_text = "\n".join([*selected_metadata, child_text]).strip()
        if not selected_metadata or retrieval_text == child_text:
            return child_text, "original"
        return retrieval_text, strategy

    def _metadata_line(self, label: str, value: Any) -> str | None:
        compact_value = self._compact_metadata_text(value)
        if not compact_value:
            return None
        return f"{label}: {compact_value}"

    def _compact_metadata_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        if len(normalized) <= 180:
            return normalized
        return normalized[:177].rstrip() + "..."

    def _related_parent_preview(self, *, source_block: _ChunkSourceBlock, assembly: _ParentAssembly) -> str:
        related_text = " ".join(block.text for block in assembly.blocks if block.source_block_id != source_block.source_block_id)
        return self._build_parent_preview(related_text, limit=160) if related_text else ""

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

    def _coerce_section_path(self, raw_value: Any) -> list[str]:
        if isinstance(raw_value, list):
            return [str(item) for item in raw_value if str(item).strip()]
        if isinstance(raw_value, str) and raw_value.strip():
            return [item.strip() for item in raw_value.split("/") if item.strip()]
        return []

    def _coerce_int(self, raw_value: Any, default_value: int) -> int:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return default_value

    def _page_sort_value(self, raw_value: str | None) -> tuple[int, str]:
        if raw_value is None:
            return (10**9, "")
        try:
            return (int(str(raw_value)), str(raw_value))
        except ValueError:
            return (10**9, str(raw_value))

    def _coerce_graph_edges(self, raw_value: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_value, list):
            return []
        return [dict(item) for item in raw_value if isinstance(item, dict)]


SemanticChunkingService = StructureAwareChunkingService



