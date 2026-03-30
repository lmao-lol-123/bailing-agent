from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

from src.core.models import DocumentSummary, IngestResult


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestURLRequest(BaseModel):
    url: HttpUrl


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestManyResponse(BaseModel):
    results: list[IngestResult]


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]

