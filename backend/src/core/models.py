from __future__ import annotations

from datetime import datetime
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


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class NormalizedDocument(BaseModel):
    doc_id: str
    source_type: SourceType
    source_name: str
    source_uri_or_path: str
    page_or_section: str | None = None
    page: str | None = None
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    doc_type: SourceType | None = None
    updated_at: datetime | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    doc_id: str
    source_name: str
    source_uri_or_path: str
    page_or_section: str | None = None
    page: str | None = None
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    doc_type: SourceType | None = None
    updated_at: datetime | None = None
    snippet: str


class ChatMessage(BaseModel):
    session_id: str
    role: ChatRole
    content: str
    grounded: bool | None = None
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime | None = None


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    updated_at: datetime | None = None


class IngestedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_name: str
    source_uri_or_path: str
    page_or_section: str | None = None
    page: str | None = None
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    doc_type: SourceType | None = None
    updated_at: datetime | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalFilter(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    doc_types: list[SourceType] = Field(default_factory=list)
    pages: list[str] = Field(default_factory=list)
    section_path_prefix: list[str] = Field(default_factory=list)
    title_contains: str | None = None
    updated_after: datetime | None = None
    updated_before: datetime | None = None


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
