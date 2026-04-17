from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

from backend.src.core.models import (
    ChatMessage,
    ChatSessionSummary,
    DocumentSummary,
    IngestResult,
    RetrievalFilter,
    SessionFileSummary,
)


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestURLRequest(BaseModel):
    url: HttpUrl


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, min_length=1)
    metadata_filter: RetrievalFilter | None = None


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=32)


class IngestManyResponse(BaseModel):
    results: list[IngestResult]


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class SessionFileListResponse(BaseModel):
    session_id: str
    files: list[SessionFileSummary]
