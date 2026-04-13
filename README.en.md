# Engineering RAG Assistant

[中文](./README.md) | [English](./README.en.md)

A lightweight RAG assistant for engineering documents. The current phase is backend-first, with a strong focus on local reliability, grounded retrieval, cited answers, and stable streaming output.

## Features

- Multi-format ingestion for web pages, PDF, Word `.docx`, Markdown, CSV, JSON, and TXT
- Backend-first scope covering ingestion, chunking, retrieval, generation, evaluation, and API reliability
- Hybrid retrieval with FAISS dense recall, BM25 lexical recall, query expansion, routing, and reranking
- Source-grounded answers with citations and retrieval metadata carried through the pipeline
- Streaming Q&A via `POST /ask/stream` using SSE token, sources, and done events
- Persistent chat sessions backed by SQLite for follow-up questions

## Tech Stack

| Component | Technology |
| --- | --- |
| API | FastAPI, Pydantic, StreamingResponse |
| Ingestion | LangChain loaders, pymupdf4llm, Unstructured |
| Chunking | Structure-first chunking with token-budget-aware splitting |
| Retrieval | sentence-transformers, FAISS, rank-bm25 |
| Generation | DeepSeek API via the OpenAI-compatible SDK |
| Persistence | SQLite, local JSON snapshots |
| Testing | pytest, pytest-asyncio, FastAPI TestClient |

## Current Backend Capabilities

- Imported documents are normalized and persisted to `data/processed/`
- Retrieval now supports query routing, deterministic expansion, hybrid recall, heuristic reranking, and targeted retry
- Child hits can be hydrated back to parent content to preserve richer answer context
- The generator explicitly declines to answer when the evidence is insufficient
- A real DeepSeek API smoke test has passed through the streaming answer path

## Quick Start

Use a repository-local virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the environment template and fill in the DeepSeek settings:

```powershell
Copy-Item .env.example .env
```

Required `.env` values:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Start the API:

```powershell
.venv\Scripts\activate
uvicorn backend.src.api.main:app --reload
```

Common entry points:

- Web/API entry: `http://127.0.0.1:8000/`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## Common Commands

Build an index:

```powershell
.venv\Scripts\activate
python -m backend.scripts.build_index data\sample.txt
```

Ask from the CLI:

```powershell
.venv\Scripts\activate
python -m backend.scripts.ask "What does this document say?"
```

Run the test suite:

```powershell
.venv\Scripts\activate
python -m pytest -q
```

## API Overview

- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `GET /sessions`
- `GET /sessions/{session_id}/messages`
- `PATCH /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /ask/stream`

`POST /ask/stream` accepts `question` plus optional `top_k`, `session_id`, and `metadata_filter`.

## Project Structure

```text
bailing-agent/
|- backend/
|  |- scripts/
|  |- src/
|  |  |- api/
|  |  |- core/
|  |  |- eval/
|  |  |- generate/
|  |  |- ingest/
|  |  |- models/
|  |  `- retrieve/
|  `- tests/
|- data/
|- docs/
|- frontend/
|- storage/
|- .github/workflows/ci.yml
|- CHANGELOG.md
|- README.en.md
`- README.md
```

## Validation Status

- Full backend test suite: `60 passed`
- Real DeepSeek streaming smoke test: passed
- CI: GitHub Actions runs on `master`, `main`, and `codex/**`

## Repository Notes

- `data/uploads/`, `data/processed/`, and `storage/` are local runtime artifacts and are not pushed to GitHub
- Frontend work is maintenance-only in the current phase; active work is concentrated in the backend
- When evidence is weak, the system is expected to answer conservatively instead of fabricating details

## References

- Project rules: [AGENTS.md](./AGENTS.md)
- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
