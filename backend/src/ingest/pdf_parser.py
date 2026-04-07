from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.src.core.config import Settings
from backend.src.core.text import is_probably_garbled, normalize_text
from backend.src.ingest.cleaning import ParsedPage

_PAGE_SEPARATOR_RE = re.compile(r"\n+--- end of page=(?P<page>\d+) ---\n+", re.IGNORECASE)


class MinerUFallbackError(RuntimeError):
    """Raised when MinerU parsing is requested but unavailable."""


class PDFParsingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def parse_pdf(self, file_path: Path, force_mineru: bool = False) -> tuple[list[ParsedPage], bool, dict[str, Any]]:
        if force_mineru:
            pages = self._load_with_mineru(file_path)
            return pages, True, self._build_parser_metadata(loader="mineru", cleaning_path="ocr_fallback", pages=pages)

        pages = self._load_with_pymupdf4llm(file_path)
        if self._should_fallback_to_mineru(pages):
            pages = self._load_with_mineru(file_path)
            return pages, True, self._build_parser_metadata(loader="mineru", cleaning_path="ocr_fallback", pages=pages)

        return pages, False, self._build_parser_metadata(loader="pymupdf4llm", cleaning_path="text_pdf", pages=pages)

    def _load_with_pymupdf4llm(self, file_path: Path) -> list[ParsedPage]:
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise RuntimeError(
                "pymupdf4llm is required for default PDF parsing. Install pymupdf4llm and pymupdf."
            ) from exc

        result = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True,
            page_separators=False,
            use_ocr=False,
            force_ocr=False,
            force_text=True,
            write_images=False,
            header=True,
            footer=True,
        )
        return self._normalize_pymupdf4llm_output(result, file_path)

    def _normalize_pymupdf4llm_output(self, result: Any, file_path: Path) -> list[ParsedPage]:
        if isinstance(result, list):
            pages: list[ParsedPage] = []
            for index, page in enumerate(result, start=1):
                metadata = dict(page.get("metadata", {})) if isinstance(page, dict) else {}
                page_number = int(metadata.get("page_number", index))
                page_count = int(metadata.get("page_count", len(result)))
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        page_label=str(metadata.get("page_number", page_number)),
                        text=str(page.get("text", "")) if isinstance(page, dict) else str(page),
                        metadata={
                            "source_page_count": page_count,
                            "toc_items": list(page.get("toc_items", [])) if isinstance(page, dict) else [],
                            "page_boxes": list(page.get("page_boxes", [])) if isinstance(page, dict) else [],
                            "tables": list(page.get("tables", [])) if isinstance(page, dict) else [],
                            "images": list(page.get("images", [])) if isinstance(page, dict) else [],
                            "file_path": str(metadata.get("file_path", file_path)),
                        },
                    )
                )
            return pages

        if isinstance(result, str):
            chunks = _PAGE_SEPARATOR_RE.split(result)
            if len(chunks) > 1:
                pages: list[ParsedPage] = []
                initial_page_text = chunks[0].strip()
                if initial_page_text:
                    pages.append(ParsedPage(page_number=1, page_label="1", text=initial_page_text, metadata={}))
                for index in range(1, len(chunks), 2):
                    current_page_number = int(chunks[index]) + 1
                    page_text = chunks[index + 1].strip()
                    pages.append(
                        ParsedPage(
                            page_number=current_page_number,
                            page_label=str(current_page_number),
                            text=page_text,
                            metadata={},
                        )
                    )
                return pages

            normalized = result.strip()
            return [ParsedPage(page_number=1, page_label="1", text=normalized, metadata={})]

        raise RuntimeError(f"Unsupported pymupdf4llm output type: {type(result)!r}")

    def _should_fallback_to_mineru(self, pages: list[ParsedPage]) -> bool:
        merged_text = "\n".join(page.text for page in pages)
        normalized_text = normalize_text(merged_text)
        if len(normalized_text) < self._settings.pdf_min_text_chars:
            return True
        if is_probably_garbled(merged_text, self._settings.pdf_garbled_char_ratio):
            return True
        non_empty_lines = [line for line in merged_text.splitlines() if line.strip()]
        return len(non_empty_lines) < 3

    def _load_with_mineru(self, file_path: Path) -> list[ParsedPage]:
        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.libs.version import __version__ as _  # noqa: F401
        except ImportError as exc:
            raise MinerUFallbackError(
                "MinerU fallback requested, but MinerU is not installed. "
                "Install MinerU separately for OCR-heavy PDF support."
            ) from exc

        output_dir = self._settings.processed_directory / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = PymuDocDataset(str(file_path))
        infer_result = dataset.classify().analyze()
        markdown_result = infer_result.pipe_txt_mode(FileBasedDataWriter(str(output_dir)), "")
        markdown_content = markdown_result.get_markdown("")
        normalized = markdown_content.strip()
        if not normalized:
            raise MinerUFallbackError(f"MinerU could not extract usable text from {file_path.name}.")

        return [
            ParsedPage(
                page_number=1,
                page_label="1",
                text=normalized,
                metadata={"source_page_count": self._get_pdf_page_count(file_path)},
            )
        ]

    def _build_parser_metadata(self, *, loader: str, cleaning_path: str, pages: list[ParsedPage]) -> dict[str, Any]:
        source_page_count = max((int(page.metadata.get("source_page_count", page.page_number)) for page in pages), default=1)
        return {
            "loader": loader,
            "cleaning_path": cleaning_path,
            "source_page_count": source_page_count,
        }

    def _get_pdf_page_count(self, file_path: Path) -> int:
        try:
            import fitz
        except ImportError:
            return 1

        document = fitz.open(str(file_path))
        try:
            return int(document.page_count)
        finally:
            document.close()
