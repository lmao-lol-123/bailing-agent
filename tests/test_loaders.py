from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from src.core.config import Settings
from src.core.models import NormalizedDocument, SourceType
from src.ingest.loaders import DocumentLoaderRouter


def make_case_dir(name: str) -> Path:
    root = Path("test_runtime") / f"{name}-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_txt_file_uses_text_loader() -> None:
    case_dir = make_case_dir("txt-loader")
    text_path = case_dir / "notes.txt"
    text_path.write_text("hello engineering rag", encoding="utf-8")

    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        chroma_persist_directory=case_dir / "chroma",
    )
    settings.ensure_directories()
    router = DocumentLoaderRouter(settings)

    documents, used_mineru = router.load_file(text_path)

    assert used_mineru is False
    assert len(documents) == 1
    assert documents[0].source_type == SourceType.TXT
    assert documents[0].content == "hello engineering rag"


def test_pdf_falls_back_to_mineru_when_extraction_quality_is_low(monkeypatch) -> None:
    case_dir = make_case_dir("pdf-fallback")
    pdf_path = case_dir / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    settings = Settings(
        uploads_directory=case_dir / "uploads",
        processed_directory=case_dir / "processed",
        chroma_persist_directory=case_dir / "chroma",
        pdf_min_text_chars=50,
    )
    settings.ensure_directories()
    router = DocumentLoaderRouter(settings)

    monkeypatch.setattr(
        "src.ingest.loaders.PyPDFLoader.load",
        lambda self: [Document(page_content="bad text", metadata={})],
    )
    monkeypatch.setattr(
        router,
        "_load_pdf_with_mineru",
        lambda file_path, doc_id: [
            NormalizedDocument(
                doc_id=doc_id,
                source_type=SourceType.PDF,
                source_name=file_path.name,
                source_uri_or_path=str(file_path),
                content="ocr content from mineru",
                metadata={"loader": "mineru"},
            )
        ],
    )

    documents, used_mineru = router.load_file(pdf_path)

    assert used_mineru is True
    assert documents[0].metadata["loader"] == "mineru"
