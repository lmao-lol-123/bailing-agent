from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from backend.src.core.config import Settings
from backend.src.core.models import SourceType
from backend.src.ingest.cleaning import ParsedPage
from backend.src.ingest.loaders import DocumentLoaderRouter


def make_case_dir(name: str) -> Path:
    root = Path("backend/.pytest-tmp") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_txt_file_uses_text_loader() -> None:
    case_dir = make_case_dir("txt-loader")
    text_path = case_dir / "notes.txt"
    text_path.write_text("hello engineering rag", encoding="utf-8")

    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
    )
    settings.ensure_directories()
    router = DocumentLoaderRouter(settings)

    documents, used_mineru = router.load_file(text_path)

    assert used_mineru is False
    assert len(documents) == 1
    assert documents[0].source_type == SourceType.TXT
    assert documents[0].title == "notes"
    assert documents[0].doc_type == SourceType.TXT
    assert documents[0].page == "1"
    assert documents[0].updated_at is not None
    assert documents[0].content == "hello engineering rag"
    assert documents[0].metadata["loader"] == "text_loader"
    assert documents[0].metadata["cleaning_path"] == "non_pdf"


def test_pdf_uses_pymupdf4llm_by_default_and_preserves_page_marker(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-default")
    pdf_path = case_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
    )
    settings.ensure_directories()
    router = DocumentLoaderRouter(settings)

    monkeypatch.setattr(
        router._pdf_parser,
        "parse_pdf",
        lambda file_path, force_mineru=False: (
            [
                ParsedPage(
                    page_number=1,
                    page_label="1",
                    text="# Guide\n\n1\n\nParagraph on page one.",
                    metadata={"source_page_count": 2, "page_boxes": []},
                ),
                ParsedPage(
                    page_number=2,
                    page_label="2",
                    text="2\n\nFigure 1. Request flow\n![Alt](fig.png)",
                    metadata={"source_page_count": 2, "page_boxes": []},
                ),
            ],
            False,
            {"loader": "pymupdf4llm", "cleaning_path": "text_pdf", "source_page_count": 2},
        ),
    )

    documents, used_mineru = router.load_file(pdf_path)

    assert used_mineru is False
    assert any(document.metadata["block_type"] == "page_marker" for document in documents)
    assert any(document.metadata["loader"] == "pymupdf4llm" for document in documents)
    assert all(document.metadata["cleaning_path"] == "text_pdf" for document in documents)
    assert any(document.page == "2" for document in documents)


def test_pdf_falls_back_to_mineru_when_requested(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-fallback")
    pdf_path = case_dir / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
    )
    settings.ensure_directories()
    router = DocumentLoaderRouter(settings)

    monkeypatch.setattr(
        router._pdf_parser,
        "parse_pdf",
        lambda file_path, force_mineru=False: (
            [
                ParsedPage(
                    page_number=1,
                    page_label="1",
                    text="1\n\n# Scan Guide\n\nParagraph from OCR.",
                    metadata={"source_page_count": 1, "page_boxes": []},
                )
            ],
            True,
            {"loader": "mineru", "cleaning_path": "ocr_fallback", "source_page_count": 1},
        ),
    )

    documents, used_mineru = router.load_file(pdf_path, force_mineru=True)

    assert used_mineru is True
    assert all(document.metadata["loader"] == "mineru" for document in documents)
    assert all(document.metadata["cleaning_path"] == "ocr_fallback" for document in documents)


def test_loader_generates_stable_doc_id_for_same_file() -> None:
    case_dir = make_case_dir("stable-doc-id")
    text_path = case_dir / "notes.txt"
    text_path.write_text("hello engineering rag", encoding="utf-8")

    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        faiss_index_directory=case_dir / "faiss",
        index_state_directory=case_dir / "index",
    )
    settings.ensure_directories()
    router = DocumentLoaderRouter(settings)

    first_documents, _ = router.load_file(text_path)
    second_documents, _ = router.load_file(text_path)

    assert first_documents[0].doc_id == second_documents[0].doc_id
    assert first_documents[0].doc_id.startswith("doc-")
