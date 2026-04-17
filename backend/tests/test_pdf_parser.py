from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from backend.src.core.config import Settings
from backend.src.ingest.cleaning import ParsedPage
from backend.src.ingest.pdf_parser import MinerUFallbackError, PDFParsingService


def make_case_dir(name: str) -> Path:
    root = Path("backend/.pytest-tmp") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_service(case_dir: Path) -> PDFParsingService:
    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
    )
    settings.ensure_directories()
    return PDFParsingService(settings)


def test_parse_pdf_uses_mineru_first_for_non_sensitive(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-parser-mineru-first")
    pdf_path = case_dir / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = make_service(case_dir)

    mineru_pages = [
        ParsedPage(
            page_number=1,
            page_label="1",
            text="OCR text from MinerU",
            parser_source="mineru_flash",
            metadata={"source_page_count": 1, "mineru_mode": "flash"},
        )
    ]
    monkeypatch.setattr(parser, "_load_with_mineru", lambda *, file_path, mode: mineru_pages)
    monkeypatch.setattr(
        parser,
        "_load_with_pymupdf4llm",
        lambda file_path: (_ for _ in ()).throw(
            AssertionError("pymupdf4llm should not run for non-sensitive first-pass success")
        ),
    )

    pages, used_mineru, metadata = parser.parse_pdf(pdf_path, is_sensitive=False)

    assert pages == mineru_pages
    assert used_mineru is True
    assert metadata["loader"] == "mineru"
    assert metadata["cleaning_path"] == "mineru_api"


def test_parse_pdf_skips_mineru_for_sensitive_documents(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-parser-sensitive")
    pdf_path = case_dir / "confidential-report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = make_service(case_dir)

    pymupdf_pages = [
        ParsedPage(
            page_number=1,
            page_label="1",
            text="This is long enough text from pymupdf4llm parser to pass quality check.",
            parser_source="pymupdf4llm",
            metadata={"source_page_count": 1},
        )
    ]
    monkeypatch.setattr(
        parser,
        "_load_with_mineru",
        lambda *, file_path, mode: (_ for _ in ()).throw(
            AssertionError("MinerU should not run for sensitive documents")
        ),
    )
    monkeypatch.setattr(parser, "_load_with_pymupdf4llm", lambda file_path: pymupdf_pages)

    pages, used_mineru, metadata = parser.parse_pdf(pdf_path)

    assert pages == pymupdf_pages
    assert used_mineru is False
    assert metadata["loader"] == "pymupdf4llm"


def test_parse_pdf_falls_back_to_pymupdf_when_mineru_fails(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-parser-fallback")
    pdf_path = case_dir / "general.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = make_service(case_dir)

    pymupdf_pages = [
        ParsedPage(
            page_number=1,
            page_label="1",
            text="This is long enough text from pymupdf4llm parser to pass quality check.",
            parser_source="pymupdf4llm",
            metadata={"source_page_count": 1},
        )
    ]
    monkeypatch.setattr(
        parser,
        "_load_with_mineru",
        lambda *, file_path, mode: (_ for _ in ()).throw(
            MinerUFallbackError("network unavailable")
        ),
    )
    monkeypatch.setattr(parser, "_load_with_pymupdf4llm", lambda file_path: pymupdf_pages)

    pages, used_mineru, metadata = parser.parse_pdf(pdf_path, is_sensitive=False)

    assert pages == pymupdf_pages
    assert used_mineru is False
    assert metadata["loader"] == "pymupdf4llm"
    assert metadata["cleaning_path"] == "text_pdf"


def test_parse_pdf_force_mineru_raises_on_mineru_failure(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-parser-force")
    pdf_path = case_dir / "force.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = make_service(case_dir)

    monkeypatch.setattr(
        parser,
        "_load_with_mineru",
        lambda *, file_path, mode: (_ for _ in ()).throw(MinerUFallbackError("mineru unavailable")),
    )

    with pytest.raises(MinerUFallbackError):
        parser.parse_pdf(pdf_path, force_mineru=True)
