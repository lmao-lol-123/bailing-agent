from __future__ import annotations

import json
import os
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

    def parse_pdf(
        self,
        file_path: Path,
        force_mineru: bool = False,
        *,
        is_sensitive: bool | None = None,
        mineru_mode: str | None = None,
    ) -> tuple[list[ParsedPage], bool, dict[str, Any]]:
        sensitive = self._is_sensitive_document(file_path=file_path, explicit_value=is_sensitive)
        resolved_mode = self._resolve_mineru_mode(mineru_mode)
        mineru_error: Exception | None = None
        mineru_attempted = False

        if force_mineru:
            pages = self._load_with_mineru(file_path=file_path, mode=resolved_mode)
            return (
                pages,
                True,
                self._build_parser_metadata(
                    loader="mineru", cleaning_path="ocr_fallback", pages=pages
                ),
            )

        if not sensitive:
            mineru_attempted = True
            try:
                pages = self._load_with_mineru(file_path=file_path, mode=resolved_mode)
                return (
                    pages,
                    True,
                    self._build_parser_metadata(
                        loader="mineru", cleaning_path="mineru_api", pages=pages
                    ),
                )
            except Exception as exc:
                mineru_error = exc

        pymupdf_error: Exception | None = None
        pages: list[ParsedPage] | None = None
        try:
            pages = self._load_with_pymupdf4llm(file_path)
        except Exception as exc:
            pymupdf_error = exc

        if pages is None:
            if mineru_error is not None:
                raise RuntimeError(
                    f"Both MinerU and pymupdf4llm parsing failed for {file_path.name}: mineru={mineru_error}; pymupdf={pymupdf_error}"
                ) from pymupdf_error or mineru_error
            if pymupdf_error is not None:
                raise RuntimeError(
                    f"pymupdf4llm parsing failed for {file_path.name}: {pymupdf_error}"
                ) from pymupdf_error
            raise RuntimeError(f"Failed to parse PDF: {file_path.name}")

        if not sensitive and not mineru_attempted and self._should_fallback_to_mineru(pages):
            try:
                pages = self._load_with_mineru(file_path=file_path, mode=resolved_mode)
                return (
                    pages,
                    True,
                    self._build_parser_metadata(
                        loader="mineru", cleaning_path="ocr_fallback", pages=pages
                    ),
                )
            except Exception:
                pass

        if pymupdf_error is not None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"pymupdf4llm parsing failed for {file_path.name}: {pymupdf_error}"
            ) from pymupdf_error
        return (
            pages,
            False,
            self._build_parser_metadata(
                loader="pymupdf4llm", cleaning_path="text_pdf", pages=pages
            ),
        )

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
                page_dict = dict(page) if isinstance(page, dict) else {"text": str(page)}
                metadata = (
                    dict(page_dict.get("metadata", {}))
                    if isinstance(page_dict.get("metadata", {}), dict)
                    else {}
                )
                page_number = int(metadata.get("page_number", page_dict.get("page_number", index)))
                page_count = int(metadata.get("page_count", len(result)))
                page_boxes = list(page_dict.get("page_boxes", []))
                tables = list(page_dict.get("tables", []))
                images = list(page_dict.get("images", []))
                layout_items = self._extract_layout_items(
                    page_boxes=page_boxes, tables=tables, images=images, parser_source="pymupdf4llm"
                )
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        page_label=str(
                            metadata.get("page_label") or metadata.get("page_number") or page_number
                        ),
                        text=str(page_dict.get("text", "")),
                        width=self._coerce_optional_float(
                            page_dict.get("width") or metadata.get("width")
                        ),
                        height=self._coerce_optional_float(
                            page_dict.get("height") or metadata.get("height")
                        ),
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
                    pages.append(
                        ParsedPage(
                            page_number=1,
                            page_label="1",
                            text=initial_page_text,
                            parser_source="pymupdf4llm_text_only",
                            metadata={"parser_source": "pymupdf4llm_text_only"},
                        )
                    )
                for index in range(1, len(chunks), 2):
                    current_page_number = int(chunks[index]) + 1
                    page_text = chunks[index + 1].strip()
                    pages.append(
                        ParsedPage(
                            page_number=current_page_number,
                            page_label=str(current_page_number),
                            text=page_text,
                            parser_source="pymupdf4llm_text_only",
                            metadata={"parser_source": "pymupdf4llm_text_only"},
                        )
                    )
                return pages
            normalized = result.strip()
            return [
                ParsedPage(
                    page_number=1,
                    page_label="1",
                    text=normalized,
                    parser_source="pymupdf4llm_text_only",
                    metadata={"parser_source": "pymupdf4llm_text_only"},
                )
            ]

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

    def _load_with_mineru(self, *, file_path: Path, mode: str) -> list[ParsedPage]:
        try:
            from langchain_mineru import MinerULoader
            from mineru.exceptions import FlashFileTooLargeError, FlashPageLimitError, MinerUError
        except ImportError as exc:
            raise MinerUFallbackError(
                "MinerU API fallback requested, but langchain-mineru/mineru-open-sdk is not installed. Install langchain-mineru."
            ) from exc

        if not self._settings.mineru_enabled:
            raise MinerUFallbackError("MinerU API integration is disabled by settings.")

        requested_mode = self._resolve_mineru_mode(mode)
        token = self._resolve_mineru_token()

        def run_loader(target_mode: str) -> list[ParsedPage]:
            loader = MinerULoader(
                source=str(file_path),
                language=self._settings.mineru_language,
                timeout=self._settings.mineru_timeout_seconds,
                split_pages=self._settings.mineru_split_pages,
                mode=target_mode,
                token=token,
                ocr=target_mode == "precision",
                formula=True,
                table=True,
            )
            return self._normalize_mineru_documents(docs=loader.load(), mode=target_mode)

        try:
            return run_loader(requested_mode)
        except (FlashFileTooLargeError, FlashPageLimitError):
            if (
                requested_mode == "flash"
                and self._settings.mineru_allow_precision_fallback
                and token
            ):
                try:
                    return run_loader("precision")
                except Exception as exc:
                    raise MinerUFallbackError(
                        f"MinerU precision fallback failed for {file_path.name}: {exc}"
                    ) from exc
            raise MinerUFallbackError(
                f"MinerU flash mode limit reached for {file_path.name}; no precision token configured for fallback."
            )
        except MinerUError as exc:
            raise MinerUFallbackError(
                f"MinerU API parsing failed for {file_path.name}: {exc}"
            ) from exc
        except Exception as exc:
            raise MinerUFallbackError(
                f"MinerU API parsing failed for {file_path.name}: {exc}"
            ) from exc

    def _normalize_mineru_documents(self, *, docs: list[Any], mode: str) -> list[ParsedPage]:
        non_empty_docs = [doc for doc in docs if str(getattr(doc, "page_content", "")).strip()]
        if not non_empty_docs:
            raise MinerUFallbackError("MinerU returned no usable markdown content.")

        parser_source = f"mineru_{mode}"
        source_page_count = len(non_empty_docs)
        pages: list[ParsedPage] = []
        for index, doc in enumerate(non_empty_docs, start=1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            page_number = self._coerce_page_number(metadata.get("page"), default=index)
            page_label = str(metadata.get("page_label") or page_number)
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    page_label=page_label,
                    text=str(doc.page_content).strip(),
                    metadata={
                        "source_page_count": source_page_count,
                        "parser_source": parser_source,
                        "mineru_mode": mode,
                        "mineru_loader_metadata": metadata,
                    },
                    parser_source=parser_source,
                )
            )
        return pages

    def _resolve_mineru_mode(self, requested_mode: str | None) -> str:
        mode = (requested_mode or self._settings.mineru_default_mode or "flash").strip().lower()
        return mode if mode in {"flash", "precision"} else "flash"

    def _resolve_mineru_token(self) -> str | None:
        token = (self._settings.mineru_api_token or os.getenv("MINERU_TOKEN") or "").strip()
        return token or None

    def _is_sensitive_document(self, *, file_path: Path, explicit_value: bool | None) -> bool:
        if explicit_value is not None:
            return explicit_value
        name = file_path.name.lower()
        patterns = [
            item.strip().lower()
            for item in self._settings.mineru_sensitive_name_patterns.split(",")
            if item.strip()
        ]
        if any(pattern in name for pattern in patterns):
            return True
        return self._settings.mineru_default_sensitive

    def _extract_layout_items(
        self, *, page_boxes: list[Any], tables: list[Any], images: list[Any], parser_source: str
    ) -> list[ParsedLayoutItem]:
        items: list[ParsedLayoutItem] = []
        for index, entry in enumerate(page_boxes):
            item = self._normalize_layout_entry(
                entry=entry, order=index, parser_source=parser_source
            )
            if item is not None:
                items.append(item)
        next_order = len(items)
        if not any(item.item_type == "table" for item in items):
            for table in tables:
                item = self._normalize_table_entry(
                    table=table, order=next_order, parser_source=parser_source
                )
                if item is not None:
                    items.append(item)
                    next_order += 1
        if not any(item.item_type in {"image", "figure"} for item in items):
            for image in images:
                item = self._normalize_image_entry(
                    image=image, order=next_order, parser_source=parser_source
                )
                if item is not None:
                    items.append(item)
                    next_order += 1
        return items

    def _normalize_layout_entry(
        self, *, entry: Any, order: int, parser_source: str
    ) -> ParsedLayoutItem | None:
        if isinstance(entry, str):
            text = entry.strip()
            return (
                ParsedLayoutItem(
                    item_type=None, text=text, order=order, parser_source=parser_source
                )
                if text
                else None
            )
        if not isinstance(entry, dict):
            return None
        text = self._extract_entry_text(entry)
        if not text:
            return None
        item_type = (
            entry.get("type") or entry.get("kind") or entry.get("block_type") or entry.get("label")
        )
        return ParsedLayoutItem(
            item_type=str(item_type) if item_type is not None else None,
            text=text,
            bbox=self._normalize_bbox(entry.get("bbox") or entry.get("box") or entry.get("rect")),
            order=order,
            metadata={
                key: value
                for key, value in entry.items()
                if key not in {"text", "lines", "bbox", "box", "rect"}
            },
            parser_block_id=str(
                entry.get("id") or entry.get("block_id") or entry.get("number") or f"box-{order}"
            ),
            parser_source=parser_source,
        )

    def _normalize_table_entry(
        self, *, table: Any, order: int, parser_source: str
    ) -> ParsedLayoutItem | None:
        if not isinstance(table, dict):
            return None
        rows = table.get("rows") or table.get("data") or []
        headers = table.get("headers") or []
        text = self._extract_entry_text(table)
        if not text and rows:
            text = json.dumps(rows, ensure_ascii=False)
        if not text:
            return None
        return ParsedLayoutItem(
            item_type="table",
            text=text,
            bbox=self._normalize_bbox(table.get("bbox") or table.get("box") or table.get("rect")),
            order=order,
            metadata={"rows": rows, "headers": headers},
            parser_block_id=str(table.get("id") or f"table-{order}"),
            parser_source=parser_source,
        )

    def _normalize_image_entry(
        self, *, image: Any, order: int, parser_source: str
    ) -> ParsedLayoutItem | None:
        if not isinstance(image, dict):
            return None
        alt = image.get("alt") or image.get("caption") or image.get("title") or image.get("text")
        src = image.get("src") or image.get("path") or image.get("image_path")
        text = "\n".join(
            part
            for part in [
                f"![{alt}]({src})" if alt or src else None,
                str(image.get("caption") or "").strip(),
            ]
            if part
        )
        if not text:
            return None
        return ParsedLayoutItem(
            item_type="image",
            text=text,
            bbox=self._normalize_bbox(image.get("bbox") or image.get("box") or image.get("rect")),
            order=order,
            metadata={
                "image_alt_text": alt,
                "image_source": src,
                "image_title": image.get("title"),
            },
            parser_block_id=str(image.get("id") or f"image-{order}"),
            parser_source=parser_source,
        )

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
                return {
                    "x0": float(raw_bbox["left"]),
                    "y0": float(raw_bbox["top"]),
                    "x1": float(raw_bbox["right"]),
                    "y1": float(raw_bbox["bottom"]),
                }
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            return {
                "x0": float(raw_bbox[0]),
                "y0": float(raw_bbox[1]),
                "x1": float(raw_bbox[2]),
                "y1": float(raw_bbox[3]),
            }
        return None

    def _coerce_optional_float(self, value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _coerce_page_number(self, value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_parser_metadata(
        self, *, loader: str, cleaning_path: str, pages: list[ParsedPage]
    ) -> dict[str, Any]:
        source_page_count = max(
            (int(page.metadata.get("source_page_count", page.page_number)) for page in pages),
            default=1,
        )
        parser_source = next((page.parser_source for page in pages if page.parser_source), loader)
        metadata = {
            "loader": loader,
            "cleaning_path": cleaning_path,
            "source_page_count": source_page_count,
            "parser_source": parser_source,
        }
        if loader == "mineru":
            metadata["mineru_mode"] = next(
                (
                    str(page.metadata.get("mineru_mode"))
                    for page in pages
                    if page.metadata.get("mineru_mode")
                ),
                None,
            )
        return metadata
