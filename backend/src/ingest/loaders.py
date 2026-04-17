from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from langchain_community.document_loaders import CSVLoader, JSONLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import NormalizedDocument, SourceType
from backend.src.ingest.cleaning import (
    StructuredContentCleaner,
    StructuredDocument,
)
from backend.src.ingest.pdf_parser import MinerUFallbackError, PDFParsingService

_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}-")


class DocumentLoaderRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cleaner = StructuredContentCleaner()
        self._pdf_parser = PDFParsingService(settings)

    def load_file(
        self,
        file_path: Path,
        force_mineru: bool = False,
        *,
        is_sensitive: bool | None = None,
        mineru_mode: str | None = None,
        source_name: str | None = None,
        doc_id_override: str | None = None,
    ) -> tuple[list[NormalizedDocument], bool]:
        suffix = file_path.suffix.lower()
        source_type = self._detect_source_type(file_path)
        resolved_source_name = source_name or file_path.name
        doc_id = doc_id_override or self._build_file_doc_id(file_path=file_path, source_type=source_type)
        updated_at = self._resolve_updated_at(file_path)

        if suffix == ".pdf":
            pages, used_mineru, parser_metadata = self._pdf_parser.parse_pdf(
                file_path=file_path,
                force_mineru=force_mineru,
                is_sensitive=is_sensitive,
                mineru_mode=mineru_mode,
            )
            structured_document = self._cleaner.build_document_from_pages(
                doc_id=doc_id,
                source_type=SourceType.PDF,
                title=Path(resolved_source_name).stem,
                pages=pages,
                include_page_markers=True,
            )
            normalized_documents = self._build_documents_from_structured_document(
                structured_document=structured_document,
                source_name=resolved_source_name,
                source_uri_or_path=str(file_path),
                updated_at=updated_at,
                base_metadata=parser_metadata,
                cleaning_rules_applied=self._build_pdf_cleaning_rules(parser_metadata),
                fallback_used=used_mineru,
                base_title=Path(resolved_source_name).stem,
            )
            return normalized_documents, used_mineru

        loader = self._build_file_loader(file_path)
        documents = self._convert_documents(
            doc_id=doc_id,
            source_type=source_type,
            source_name=resolved_source_name,
            source_uri_or_path=str(file_path),
            updated_at=updated_at,
            documents=loader.load(),
        )
        return documents, False

    def load_url(self, url: str) -> list[NormalizedDocument]:
        doc_id = self._build_url_doc_id(url)
        parsed = urlparse(url)
        source_name = parsed.netloc or "web-page"
        return self._convert_documents(
            doc_id=doc_id,
            source_type=SourceType.WEB,
            source_name=source_name,
            source_uri_or_path=url,
            updated_at=self._resolve_updated_at(None),
            documents=WebBaseLoader(url).load(),
        )

    def _build_file_loader(self, file_path: Path) -> Any:
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            return UnstructuredWordDocumentLoader(str(file_path))
        if suffix == ".md":
            return UnstructuredMarkdownLoader(str(file_path))
        if suffix == ".csv":
            return CSVLoader(str(file_path))
        if suffix == ".json":
            return JSONLoader(file_path=str(file_path), jq_schema=".", text_content=False)
        if suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8")
        raise ValueError(f"Unsupported file type: {suffix}")

    def _convert_documents(
        self,
        *,
        doc_id: str,
        source_type: SourceType,
        source_name: str,
        source_uri_or_path: str,
        updated_at,
        documents: list[Document],
    ) -> list[NormalizedDocument]:
        normalized_documents: list[NormalizedDocument] = []
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata or {})
            raw_text = document.page_content or ""
            title = self._extract_title(
                source_name=source_name, metadata=metadata, content=raw_text
            )
            page_or_section = self._extract_page_or_section(metadata, index)
            page = self._extract_page(metadata, page_or_section)
            block_page = self._coerce_page_number(page, index)
            page_label = str(page) if page is not None else str(block_page)

            structured_document, cleaning_rules = self._cleaner.build_document_from_non_pdf(
                doc_id=doc_id,
                source_type=source_type,
                title=title,
                text=raw_text,
                page_number=block_page,
                page_label=page_label,
            )
            normalized_documents.extend(
                self._build_documents_from_structured_document(
                    structured_document=structured_document,
                    source_name=source_name,
                    source_uri_or_path=source_uri_or_path,
                    updated_at=updated_at,
                    base_metadata=metadata
                    | {
                        "loader": self._detect_loader_name(source_type),
                        "cleaning_path": "non_pdf",
                        "source_page_count": 1,
                    },
                    cleaning_rules_applied=cleaning_rules,
                    fallback_used=False,
                    base_title=title,
                )
            )
        return normalized_documents

    def _build_documents_from_structured_document(
        self,
        *,
        structured_document: StructuredDocument,
        source_name: str,
        source_uri_or_path: str,
        updated_at,
        base_metadata: dict[str, Any],
        cleaning_rules_applied: list[str],
        fallback_used: bool,
        base_title: str | None,
    ) -> list[NormalizedDocument]:
        excluded_block_ids = [
            block.block_id
            for block in structured_document.blocks
            if block.metadata.get("excluded_from_body")
        ]
        documents: list[NormalizedDocument] = []
        for block in structured_document.blocks:
            metadata = {
                **base_metadata,
                **block.metadata,
                "block_id": block.block_id,
                "block_type": block.block_type,
                "order": block.order,
                "block_order": block.metadata.get("block_order", block.order),
                "cleaning_rules_applied": cleaning_rules_applied,
                "removed_page_artifacts": excluded_block_ids,
                "fallback_used": fallback_used,
                "page": str(block.page_number) if block.page_number is not None else None,
                "page_label": block.page_label,
                "related_block_ids": block.metadata.get("related_block_ids", []),
                "references_figures": block.metadata.get("references_figures", []),
                "references_tables": block.metadata.get("references_tables", []),
                "bbox": block.bbox,
                "layout_role": block.layout_role,
                "graph_edges": block.metadata.get("graph_edges", []),
                "graph_neighbors": block.metadata.get("graph_neighbors", []),
                "parser_block_id": block.metadata.get("parser_block_id"),
                "parser_source": block.metadata.get("parser_source"),
            }
            title = block.section_path[0] if block.section_path else base_title
            page_value = str(block.page_number) if block.page_number is not None else None
            page_or_section = (
                block.page_label
                or page_value
                or (block.section_path[-1] if block.section_path else None)
            )
            documents.append(
                NormalizedDocument(
                    doc_id=structured_document.doc_id,
                    source_type=structured_document.source_type,
                    source_name=source_name,
                    source_uri_or_path=source_uri_or_path,
                    page_or_section=page_or_section,
                    page=page_value,
                    title=title,
                    section_path=list(block.section_path),
                    doc_type=structured_document.source_type,
                    updated_at=updated_at,
                    content=block.text,
                    metadata=metadata,
                )
            )
        return documents

    def _extract_title(self, *, source_name: str, metadata: dict[str, Any], content: str) -> str:
        metadata_title = metadata.get("title")
        if metadata_title:
            return str(metadata_title).strip()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        source_path = Path(source_name)
        return source_path.stem if source_path.suffix else source_name

    def _extract_page(self, metadata: dict[str, Any], page_or_section: str | None) -> str | None:
        for key in ("page", "page_number"):
            if metadata.get(key) is not None:
                return str(metadata[key])
        return page_or_section

    def _extract_page_or_section(self, metadata: dict[str, Any], index: int) -> str | None:
        for key in ("page", "page_number", "section", "seq_num"):
            if key in metadata and metadata[key] is not None:
                return str(metadata[key])
        return str(index)

    def _detect_source_type(self, file_path: Path) -> SourceType:
        suffix = file_path.suffix.lower()
        mapping = {
            ".docx": SourceType.WORD,
            ".md": SourceType.MARKDOWN,
            ".csv": SourceType.CSV,
            ".json": SourceType.JSON,
            ".txt": SourceType.TXT,
            ".pdf": SourceType.PDF,
        }
        try:
            return mapping[suffix]
        except KeyError as exc:
            raise ValueError(f"Unsupported file type: {suffix}") from exc

    def _detect_loader_name(self, source_type: SourceType) -> str:
        return {
            SourceType.WORD: "unstructured_word_document_loader",
            SourceType.MARKDOWN: "unstructured_markdown_loader",
            SourceType.CSV: "csv_loader",
            SourceType.JSON: "json_loader",
            SourceType.TXT: "text_loader",
            SourceType.WEB: "web_base_loader",
            SourceType.PDF: "pymupdf4llm",
        }[source_type]

    def _build_pdf_cleaning_rules(self, parser_metadata: dict[str, Any]) -> list[str]:
        rules = [
            "normalize_newlines",
            "trim_line_whitespace",
            "collapse_multi_spaces",
            "preserve_structure",
            "preserve_page_markers",
            "reading_order_from_parser",
            "layout_role_classification",
            "graph_relationship_projection",
        ]
        if parser_metadata.get("cleaning_path") == "ocr_fallback":
            rules.append("ocr_fallback")
        return rules

    def _coerce_page_number(self, page_value: str | None, default_value: int) -> int:
        if page_value is None:
            return default_value
        try:
            return int(page_value)
        except ValueError:
            return default_value

    def _build_file_doc_id(self, *, file_path: Path, source_type: SourceType) -> str:
        normalized_source = self._normalize_file_source_key(file_path)
        return self._hash_doc_id(prefix=source_type.value, value=normalized_source)

    def _build_url_doc_id(self, url: str) -> str:
        parsed = urlparse(url)
        normalized = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/") or "/",
                "",
                parsed.query,
                "",
            )
        )
        return self._hash_doc_id(prefix=SourceType.WEB.value, value=normalized)

    def _hash_doc_id(self, *, prefix: str, value: str) -> str:
        digest = hashlib.sha1(f"{prefix}:{value}".encode("utf-8")).hexdigest()[:16]
        return f"doc-{digest}"

    def _normalize_file_source_key(self, file_path: Path) -> str:
        resolved = file_path.resolve()
        try:
            uploads_root = self._settings.uploads_directory.resolve()
            object_root = (uploads_root / "objects").resolve()
            if object_root in resolved.parents:
                return resolved.relative_to(uploads_root).as_posix().lower()
            if uploads_root in resolved.parents:
                return _UPLOAD_PREFIX_RE.sub("", resolved.name).lower()
        except OSError:
            pass
        return resolved.as_posix().lower()

    def _resolve_updated_at(self, file_path: Path | None):
        if file_path is None:
            from datetime import datetime, timezone

            return datetime.now(timezone.utc)
        from datetime import datetime, timezone

        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)


def dump_normalized_documents(documents: list[NormalizedDocument], output_path: Path) -> None:
    serialized = [_to_json_safe(document.model_dump(mode="python")) for document in documents]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["DocumentLoaderRouter", "MinerUFallbackError", "dump_normalized_documents"]


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)

    # pymupdf geometry objects (e.g., Rect, Point, IRect) are not JSON serializable.
    if all(hasattr(value, attr) for attr in ("x0", "y0", "x1", "y1")):
        try:
            return {
                "x0": float(getattr(value, "x0")),
                "y0": float(getattr(value, "y0")),
                "x1": float(getattr(value, "x1")),
                "y1": float(getattr(value, "y1")),
            }
        except (TypeError, ValueError):
            return str(value)

    if all(hasattr(value, attr) for attr in ("x", "y")):
        try:
            return {"x": float(getattr(value, "x")), "y": float(getattr(value, "y"))}
        except (TypeError, ValueError):
            return str(value)

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except TypeError:
            pass

    return str(value)
