from __future__ import annotations

import json

import re
from pathlib import Path
from typing import Any

from backend.src.core.config import Settings
from backend.src.core.text import is_probably_garbled, normalize_text
from backend.src.ingest.cleaning import ParsedLayoutItem, ParsedPage

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
            raise RuntimeError("pymupdf4llm is required for default PDF parsing. Install pymupdf4llm and pymupdf.") from exc

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
                page_dict = dict(page) if isinstance(page, dict) else {"text": str(page)}
                metadata = dict(page_dict.get("metadata", {})) if isinstance(page_dict.get("metadata", {}), dict) else {}
                page_number = int(metadata.get("page_number", page_dict.get("page_number", index)))
                page_count = int(metadata.get("page_count", len(result)))
                page_boxes = list(page_dict.get("page_boxes", []))
                tables = list(page_dict.get("tables", []))
                images = list(page_dict.get("images", []))
                layout_items = self._extract_layout_items(page_boxes=page_boxes, tables=tables, images=images, parser_source="pymupdf4llm")
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        page_label=str(metadata.get("page_label") or metadata.get("page_number") or page_number),
                        text=str(page_dict.get("text", "")),
                        width=self._coerce_optional_float(page_dict.get("width") or metadata.get("width")),
                        height=self._coerce_optional_float(page_dict.get("height") or metadata.get("height")),
                        layout_items=layout_items,
                        metadata={
                            "source_page_count": page_count,
                            "toc_items": list(page_dict.get("toc_items", [])),
                            "page_boxes": page_boxes,
                            "tables": tables,
                            "images": images,
                            "file_path": str(metadata.get("file_path", file_path)),
                            "parser_source": "pymupdf4llm",
                        },
                        parser_source="pymupdf4llm",
                    )
                )
            return pages

        if isinstance(result, str):
            chunks = _PAGE_SEPARATOR_RE.split(result)
            if len(chunks) > 1:
                pages: list[ParsedPage] = []
                initial_page_text = chunks[0].strip()
                if initial_page_text:
                    pages.append(ParsedPage(page_number=1, page_label="1", text=initial_page_text, parser_source="pymupdf4llm_text_only", metadata={"parser_source": "pymupdf4llm_text_only"}))
                for index in range(1, len(chunks), 2):
                    current_page_number = int(chunks[index]) + 1
                    page_text = chunks[index + 1].strip()
                    pages.append(ParsedPage(page_number=current_page_number, page_label=str(current_page_number), text=page_text, parser_source="pymupdf4llm_text_only", metadata={"parser_source": "pymupdf4llm_text_only"}))
                return pages
            normalized = result.strip()
            return [ParsedPage(page_number=1, page_label="1", text=normalized, parser_source="pymupdf4llm_text_only", metadata={"parser_source": "pymupdf4llm_text_only"})]

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
            raise MinerUFallbackError("MinerU fallback requested, but MinerU is not installed. Install MinerU separately for OCR-heavy PDF support.") from exc

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
                metadata={"source_page_count": self._get_pdf_page_count(file_path), "parser_source": "mineru_text_only"},
                parser_source="mineru_text_only",
            )
        ]

    def _extract_layout_items(self, *, page_boxes: list[Any], tables: list[Any], images: list[Any], parser_source: str) -> list[ParsedLayoutItem]:
        items: list[ParsedLayoutItem] = []
        for index, entry in enumerate(page_boxes):
            item = self._normalize_layout_entry(entry=entry, order=index, parser_source=parser_source)
            if item is not None:
                items.append(item)
        next_order = len(items)
        if not any(item.item_type == "table" for item in items):
            for table in tables:
                item = self._normalize_table_entry(table=table, order=next_order, parser_source=parser_source)
                if item is not None:
                    items.append(item)
                    next_order += 1
        if not any(item.item_type in {"image", "figure"} for item in items):
            for image in images:
                item = self._normalize_image_entry(image=image, order=next_order, parser_source=parser_source)
                if item is not None:
                    items.append(item)
                    next_order += 1
        return items

    def _normalize_layout_entry(self, *, entry: Any, order: int, parser_source: str) -> ParsedLayoutItem | None:
        if isinstance(entry, str):
            text = entry.strip()
            return ParsedLayoutItem(item_type=None, text=text, order=order, parser_source=parser_source) if text else None
        if not isinstance(entry, dict):
            return None
        text = self._extract_entry_text(entry)
        if not text:
            return None
        item_type = entry.get("type") or entry.get("kind") or entry.get("block_type") or entry.get("label")
        return ParsedLayoutItem(
            item_type=str(item_type) if item_type is not None else None,
            text=text,
            bbox=self._normalize_bbox(entry.get("bbox") or entry.get("box") or entry.get("rect")),
            order=order,
            metadata={key: value for key, value in entry.items() if key not in {"text", "lines", "bbox", "box", "rect"}},
            parser_block_id=str(entry.get("id") or entry.get("block_id") or entry.get("number") or f"box-{order}"),
            parser_source=parser_source,
        )

    def _normalize_table_entry(self, *, table: Any, order: int, parser_source: str) -> ParsedLayoutItem | None:
        if not isinstance(table, dict):
            return None
        rows = table.get("rows") or table.get("data") or []
        headers = table.get("headers") or []
        text = self._extract_entry_text(table)
        if not text and rows:
            text = json.dumps(rows, ensure_ascii=False)
        if not text:
            return None
        return ParsedLayoutItem(item_type="table", text=text, bbox=self._normalize_bbox(table.get("bbox") or table.get("box") or table.get("rect")), order=order, metadata={"rows": rows, "headers": headers}, parser_block_id=str(table.get("id") or f"table-{order}"), parser_source=parser_source)

    def _normalize_image_entry(self, *, image: Any, order: int, parser_source: str) -> ParsedLayoutItem | None:
        if not isinstance(image, dict):
            return None
        alt = image.get("alt") or image.get("caption") or image.get("title") or image.get("text")
        src = image.get("src") or image.get("path") or image.get("image_path")
        text = "\n".join(part for part in [f"![{alt}]({src})" if alt or src else None, str(image.get("caption") or "").strip()] if part)
        if not text:
            return None
        return ParsedLayoutItem(item_type="image", text=text, bbox=self._normalize_bbox(image.get("bbox") or image.get("box") or image.get("rect")), order=order, metadata={"image_alt_text": alt, "image_source": src, "image_title": image.get("title")}, parser_block_id=str(image.get("id") or f"image-{order}"), parser_source=parser_source)

    def _extract_entry_text(self, entry: dict[str, Any]) -> str:
        for key in ("text", "content", "markdown", "value"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        lines = entry.get("lines")
        if isinstance(lines, list):
            fragments = []
            for line in lines:
                if isinstance(line, str) and line.strip():
                    fragments.append(line.strip())
                elif isinstance(line, dict):
                    for key in ("text", "content", "value"):
                        if isinstance(line.get(key), str) and line[key].strip():
                            fragments.append(line[key].strip())
                            break
            return "\n".join(fragments).strip()
        return ""

    def _normalize_bbox(self, raw_bbox: Any) -> dict[str, float] | None:
        if isinstance(raw_bbox, dict):
            if all(key in raw_bbox for key in ("x0", "y0", "x1", "y1")):
                return {key: float(raw_bbox[key]) for key in ("x0", "y0", "x1", "y1")}
            if all(key in raw_bbox for key in ("left", "top", "right", "bottom")):
                return {"x0": float(raw_bbox["left"]), "y0": float(raw_bbox["top"]), "x1": float(raw_bbox["right"]), "y1": float(raw_bbox["bottom"])}
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            return {"x0": float(raw_bbox[0]), "y0": float(raw_bbox[1]), "x1": float(raw_bbox[2]), "y1": float(raw_bbox[3])}
        return None

    def _coerce_optional_float(self, value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _build_parser_metadata(self, *, loader: str, cleaning_path: str, pages: list[ParsedPage]) -> dict[str, Any]:
        source_page_count = max((int(page.metadata.get("source_page_count", page.page_number)) for page in pages), default=1)
        parser_source = next((page.parser_source for page in pages if page.parser_source), loader)
        return {"loader": loader, "cleaning_path": cleaning_path, "source_page_count": source_page_count, "parser_source": parser_source}

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
