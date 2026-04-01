from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import AskRequest, DocumentListResponse, HealthResponse, IngestManyResponse, IngestURLRequest
from src.core.dependencies import AppContainer, get_container


app = FastAPI(title="Engineering RAG Assistant")
STATIC_DIRECTORY = Path(__file__).parent / "static"
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


@app.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        async for event in container.answer_service.stream_answer(request.question, top_k=request.top_k):
            yield format_sse(event["event"], event["data"])

    return StreamingResponse(event_stream(), media_type="text/event-stream")

