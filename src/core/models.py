from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    WEB = "web"
    PDF = "pdf"
    WORD = "word"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    TXT = "txt"


class NormalizedDocument(BaseModel):
    doc_id: str
    source_type: SourceType
    source_name: str
    source_uri_or_path: str
    page_or_section: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    doc_id: str
    source_name: str
    source_uri_or_path: str
    page_or_section: str | None = None
    snippet: str


class IngestedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_name: str
    source_uri_or_path: str
    page_or_section: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    source_name: str
    source_type: SourceType
    source_uri_or_path: str
    doc_id: str
    documents_loaded: int
    chunks_indexed: int
    rebuild_triggered: bool = True
    used_mineru: bool = False


class AskResult(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool = True


class DocumentSummary(BaseModel):
    doc_id: str
    source_name: str
    source_type: SourceType
    source_uri_or_path: str
    chunk_count: int
    page_or_sections: list[str] = Field(default_factory=list)

