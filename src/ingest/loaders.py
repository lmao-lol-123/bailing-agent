from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from langchain_community.document_loaders import CSVLoader, JSONLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

from src.core.config import Settings
from src.core.models import NormalizedDocument, SourceType
from src.core.text import is_probably_garbled, normalize_text


class MinerUFallbackError(RuntimeError):
    """Raised when MinerU parsing is requested but unavailable."""


class DocumentLoaderRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load_file(self, file_path: Path, force_mineru: bool = False) -> tuple[list[NormalizedDocument], bool]:
        suffix = file_path.suffix.lower()
        doc_id = str(uuid4())

        if suffix == ".pdf":
            if force_mineru:
                return self._load_pdf_with_mineru(file_path, doc_id), True

            documents = self._convert_documents(
                doc_id=doc_id,
                source_type=SourceType.PDF,
                source_name=file_path.name,
                source_uri_or_path=str(file_path),
                documents=PyPDFLoader(str(file_path)).load(),
            )
            if self._should_fallback_to_mineru(documents):
                return self._load_pdf_with_mineru(file_path, doc_id), True
            return documents, False

        loader = self._build_file_loader(file_path)
        source_type = self._detect_source_type(file_path)
        documents = self._convert_documents(
            doc_id=doc_id,
            source_type=source_type,
            source_name=file_path.name,
            source_uri_or_path=str(file_path),
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
        documents: list[Document],
    ) -> list[NormalizedDocument]:
        normalized_documents: list[NormalizedDocument] = []
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata or {})
            page_or_section = self._extract_page_or_section(metadata, index)
            content = normalize_text(document.page_content)
            if not content:
                continue
            normalized_documents.append(
                NormalizedDocument(
                    doc_id=doc_id,
                    source_type=source_type,
                    source_name=source_name,
                    source_uri_or_path=source_uri_or_path,
                    page_or_section=page_or_section,
                    content=content,
                    metadata=metadata,
                )
            )
        return normalized_documents

    def _extract_page_or_section(self, metadata: dict[str, Any], index: int) -> str | None:
        for key in ("page", "page_number", "section", "seq_num", "source"):
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

    def _should_fallback_to_mineru(self, documents: list[NormalizedDocument]) -> bool:
        merged_text = " ".join(document.content for document in documents)
        if len(merged_text) < self._settings.pdf_min_text_chars:
            return True
        return is_probably_garbled(merged_text, self._settings.pdf_garbled_char_ratio)

    def _load_pdf_with_mineru(self, file_path: Path, doc_id: str) -> list[NormalizedDocument]:
        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.libs.version import __version__ as _  # noqa: F401
        except ImportError as exc:
            raise MinerUFallbackError(
                "MinerU fallback requested, but MinerU is not installed. "
                "Install MinerU separately for OCR-heavy PDF support."
            ) from exc

        output_dir = self._settings.processed_directory / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = PymuDocDataset(str(file_path))
        infer_result = dataset.classify().analyze()
        markdown_result = infer_result.pipe_txt_mode(FileBasedDataWriter(str(output_dir)), "")
        markdown_content = markdown_result.get_markdown("")

        normalized_markdown = normalize_text(markdown_content)
        if not normalized_markdown:
            raise MinerUFallbackError(f"MinerU could not extract usable text from {file_path.name}.")

        return [
            NormalizedDocument(
                doc_id=doc_id,
                source_type=SourceType.PDF,
                source_name=file_path.name,
                source_uri_or_path=str(file_path),
                page_or_section=None,
                content=normalized_markdown,
                metadata={"loader": "mineru"},
            )
        ]


def dump_normalized_documents(documents: list[NormalizedDocument], output_path: Path) -> None:
    serialized = [document.model_dump(mode="json") for document in documents]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

