from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from langchain_community.document_loaders import CSVLoader, JSONLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import NormalizedDocument, SourceType
from backend.src.ingest.cleaning import ParsedPage, StructuredBlock, StructuredContentCleaner
from backend.src.ingest.pdf_parser import MinerUFallbackError, PDFParsingService


class DocumentLoaderRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cleaner = StructuredContentCleaner()
        self._pdf_parser = PDFParsingService(settings)

    def load_file(self, file_path: Path, force_mineru: bool = False) -> tuple[list[NormalizedDocument], bool]:
        suffix = file_path.suffix.lower()
        doc_id = str(uuid4())
        updated_at = self._resolve_updated_at(file_path)

        if suffix == ".pdf":
            pages, used_mineru, parser_metadata = self._pdf_parser.parse_pdf(file_path=file_path, force_mineru=force_mineru)
            base_title = file_path.stem
            blocks = self._cleaner.build_blocks(pages=pages, title=base_title, include_page_markers=True)
            normalized_documents = self._build_documents_from_blocks(
                doc_id=doc_id,
                source_type=SourceType.PDF,
                source_name=file_path.name,
                source_uri_or_path=str(file_path),
                updated_at=updated_at,
                blocks=blocks,
                base_metadata=parser_metadata,
                cleaning_rules_applied=self._build_pdf_cleaning_rules(parser_metadata),
                fallback_used=used_mineru,
                base_title=base_title,
            )
            return normalized_documents, used_mineru

        loader = self._build_file_loader(file_path)
        source_type = self._detect_source_type(file_path)
        documents = self._convert_documents(
            doc_id=doc_id,
            source_type=source_type,
            source_name=file_path.name,
            source_uri_or_path=str(file_path),
            updated_at=updated_at,
            documents=loader.load(),
        )
        return documents, False

    def load_url(self, url: str) -> list[NormalizedDocument]:
        doc_id = str(uuid4())
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
            cleaned_text, cleaning_rules = self._cleaner.clean_non_pdf_text(text=raw_text, source_type=source_type)
            if not cleaned_text:
                continue

            page_or_section = self._extract_page_or_section(metadata, index)
            page = self._extract_page(metadata, page_or_section)
            title = self._extract_title(source_name=source_name, metadata=metadata, content=cleaned_text)
            block_page = self._coerce_page_number(page, index)
            blocks = self._cleaner.build_blocks(
                pages=[
                    ParsedPage(
                        page_number=block_page,
                        page_label=str(page) if page is not None else str(block_page),
                        text=cleaned_text,
                        metadata={"source_page_count": 1},
                    )
                ],
                title=title,
                include_page_markers=False,
            )
            if not blocks:
                blocks = [
                    StructuredBlock(
                        block_id=str(uuid4()),
                        block_type="paragraph",
                        text=cleaned_text,
                        page_number=block_page,
                        page_label=str(page) if page is not None else str(block_page),
                        order=0,
                        section_path=[title] if title else [],
                        metadata={"related_block_ids": [], "references_figures": [], "references_tables": []},
                    )
                ]

            normalized_documents.extend(
                self._build_documents_from_blocks(
                    doc_id=doc_id,
                    source_type=source_type,
                    source_name=source_name,
                    source_uri_or_path=source_uri_or_path,
                    updated_at=updated_at,
                    blocks=blocks,
                    base_metadata=metadata | {
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

    def _build_documents_from_blocks(
        self,
        *,
        doc_id: str,
        source_type: SourceType,
        source_name: str,
        source_uri_or_path: str,
        updated_at,
        blocks: list[StructuredBlock],
        base_metadata: dict[str, Any],
        cleaning_rules_applied: list[str],
        fallback_used: bool,
        base_title: str | None,
    ) -> list[NormalizedDocument]:
        documents: list[NormalizedDocument] = []
        for block in blocks:
            metadata = {
                **base_metadata,
                **block.metadata,
                "block_id": block.block_id,
                "block_type": block.block_type,
                "order": block.order,
                "cleaning_rules_applied": cleaning_rules_applied,
                "removed_page_artifacts": [],
                "fallback_used": fallback_used,
                "page": str(block.page_number) if block.page_number is not None else None,
                "page_label": block.page_label,
                "related_block_ids": block.metadata.get("related_block_ids", []),
                "references_figures": block.metadata.get("references_figures", []),
                "references_tables": block.metadata.get("references_tables", []),
            }
            title = block.section_path[0] if block.section_path else base_title
            page_value = str(block.page_number) if block.page_number is not None else None
            page_or_section = block.page_label or page_value or (block.section_path[-1] if block.section_path else None)
            documents.append(
                NormalizedDocument(
                    doc_id=doc_id,
                    source_type=source_type,
                    source_name=source_name,
                    source_uri_or_path=source_uri_or_path,
                    page_or_section=page_or_section,
                    page=page_value,
                    title=title,
                    section_path=list(block.section_path),
                    doc_type=source_type,
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

    def _resolve_updated_at(self, file_path: Path | None):
        if file_path is None:
            from datetime import datetime, timezone

            return datetime.now(timezone.utc)
        from datetime import datetime, timezone

        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)


def dump_normalized_documents(documents: list[NormalizedDocument], output_path: Path) -> None:
    serialized = [document.model_dump(mode="json") for document in documents]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["DocumentLoaderRouter", "MinerUFallbackError", "dump_normalized_documents"]
