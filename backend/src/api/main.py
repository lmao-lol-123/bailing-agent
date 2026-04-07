from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.src.api.schemas import (
    AskRequest,
    ChatHistoryResponse,
    ChatSessionListResponse,
    DocumentListResponse,
    HealthResponse,
    IngestManyResponse,
    IngestURLRequest,
    SessionRenameRequest,
)
from backend.src.core.dependencies import AppContainer, get_container


app = FastAPI(title="Engineering RAG Assistant")
STATIC_DIRECTORY = Path(__file__).resolve().parents[3] / "frontend"
SESSION_COOKIE_NAME = "rag_session_id"
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


def format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ingest/files", response_model=IngestManyResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    container: AppContainer = Depends(get_container),
) -> IngestManyResponse:
    results = []
    for upload in files:
        payload = await upload.read()
        saved_path = container.ingestion_service.save_upload(upload.filename, payload)
        results.append(container.ingestion_service.ingest_saved_file(saved_path))
    return IngestManyResponse(results=results)


@app.post("/ingest/url", response_model=IngestManyResponse)
def ingest_url(
    request: IngestURLRequest,
    container: AppContainer = Depends(get_container),
) -> IngestManyResponse:
    result = container.ingestion_service.ingest_url(str(request.url))
    return IngestManyResponse(results=[result])


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(container: AppContainer = Depends(get_container)) -> DocumentListResponse:
    return DocumentListResponse(documents=container.vector_store.list_documents())


@app.get("/sessions", response_model=ChatSessionListResponse)
def list_sessions(container: AppContainer = Depends(get_container)) -> ChatSessionListResponse:
    return ChatSessionListResponse(sessions=container.chat_history_store.list_sessions())


@app.patch("/sessions/{session_id}", response_model=HealthResponse)
def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    container: AppContainer = Depends(get_container),
) -> HealthResponse:
    container.chat_history_store.rename_session(session_id=session_id, title=request.title)
    return HealthResponse()


@app.delete("/sessions/{session_id}", response_model=HealthResponse)
def delete_session(
    session_id: str,
    container: AppContainer = Depends(get_container),
) -> HealthResponse:
    container.chat_history_store.delete_session(session_id=session_id)
    return HealthResponse()


@app.get("/sessions/{session_id}/messages", response_model=ChatHistoryResponse)
def get_session_messages(
    session_id: str,
    container: AppContainer = Depends(get_container),
) -> ChatHistoryResponse:
    return ChatHistoryResponse(
        session_id=session_id,
        messages=container.chat_history_store.list_messages(session_id),
    )


@app.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    raw_request: Request,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    session_id = container.chat_history_store.ensure_session(
        request.session_id or raw_request.cookies.get(SESSION_COOKIE_NAME)
    )

    async def event_stream() -> AsyncIterator[str]:
        async for event in container.answer_service.stream_answer(
            session_id=session_id,
            question=request.question,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        ):
            yield format_sse(event["event"], event["data"])

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=False,
        samesite="lax",
    )
    return response
