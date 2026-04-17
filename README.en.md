# Bailing Agent

[中文](./README.md) | [English](./README.en.md)

A backend-first RAG project for engineering documents and standards-style PDFs, focused on local reliability, explainable retrieval, source-grounded answers, session-scoped file isolation, and reproducible offline evaluation.

## Overview

- document ingestion and normalization
- structure-aware chunking
- hybrid retrieval, routing, reranking, and retry
- citation-constrained answer generation
- session file upload and isolation
- DeepSeek-powered RAGAS evaluation

The project now uses `uv` for dependency management, environment setup, and lockfile handling.

## Key Features

- Multi-format ingestion for PDF, Word, Markdown, CSV, JSON, TXT, and web pages
- Hybrid retrieval with FAISS dense recall, BM25 lexical recall, query expansion, routing, and reranking
- Regulation-aware metadata such as `clause_id`, `table_id`, numeric anchors, and English aliases
- Session-scoped uploads to prevent cross-session file exposure
- Citation validation and conservative fallback behavior when evidence is weak
- DeepSeek-backed RAGAS evaluation and offline retrieval diagnostics
- Content-addressed file storage to avoid duplicate raw uploads in `data/uploads`

## Tech Stack

| Area | Technology |
| --- | --- |
| API | FastAPI, Pydantic, SSE |
| Ingestion | LangChain loaders, pymupdf4llm, Unstructured, MinerU fallback |
| Retrieval | sentence-transformers, FAISS, rank-bm25 |
| Generation | DeepSeek API via the OpenAI-compatible SDK |
| Persistence | SQLite, local JSON snapshots |
| Tooling | uv, pytest, ruff |

## Quick Start

If `uv` is not installed yet:

```powershell
winget install --id=astral-sh.uv -e
```

Initialize the environment:

```powershell
uv sync
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Minimum required values:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Start the API:

```powershell
uv run uvicorn backend.src.api.main:app --reload
```

Entry points:

- Web/API: `http://127.0.0.1:8000/`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## Common Commands

Build an index:

```powershell
uv run python -m backend.scripts.build_index path\to\document.pdf
```

Ask from the CLI:

```powershell
uv run python -m backend.scripts.ask "What is the key conclusion of this document?"
```

Run the full test suite:

```powershell
uv run pytest
```

Clean up legacy `data` storage:

```powershell
uv run python -m backend.scripts.cleanup_data_storage
uv run python -m backend.scripts.cleanup_data_storage --apply
```

## API Overview

- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `GET /sessions`
- `PATCH /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `GET /sessions/{session_id}/messages`
- `POST /sessions/{session_id}/files`
- `GET /sessions/{session_id}/files`
- `DELETE /sessions/{session_id}/files/{file_id}`
- `POST /sessions/{session_id}/files/{file_id}/recover-delete`
- `POST /ask/stream`

## Project Structure

This section is written as a directory tree with short responsibility notes for quick orientation.

```text
bailing-agent/
├─ .github/
│  └─ workflows/
│     └─ ci.yml                      # GitHub Actions workflow for CI
├─ backend/
│  ├─ scripts/                       # CLI utilities: indexing, asking, evaluation, cleanup, recovery
│  ├─ src/
│  │  ├─ api/                        # FastAPI routes and API schemas
│  │  ├─ core/                       # Config, dependency wiring, shared models, chat store, text utilities
│  │  ├─ eval/                       # Offline evaluation dataset helpers
│  │  ├─ generate/                   # Answer generation, citation validation, prompt assembly
│  │  ├─ ingest/                     # Ingestion, PDF parsing, cleaning, chunking, object storage
│  │  ├─ models/                     # Reserved package, currently mostly a placeholder
│  │  └─ retrieve/                   # Retrieval, index management, routing, expansion, reranking, retry
│  ├─ tests/                         # Backend test suite
│  └─ test_runtime/                  # Temporary runtime fixtures for tests
├─ data/                             # Raw document objects and processed snapshots
│  ├─ uploads/                       # Raw content-addressed file objects
│  └─ processed/                     # Normalized parse snapshots
├─ docs/
│  ├─ README.md                      # Extra Chinese documentation
│  ├─ README.en.md                   # Extra English documentation
│  ├─ PROJECT_STRUCTURE.md           # Detailed project structure notes
│  └─ roadmap.txt                    # Planning notes
├─ frontend/
│  ├─ index.html                     # Static page shell
│  ├─ app.js                         # Frontend interaction logic
│  └─ styles.css                     # Styles
├─ storage/                          # Live RAG runtime state
│  ├─ chat_history.sqlite3           # Chat history and session-file metadata
│  ├─ faiss/                         # Vector index files
│  ├─ index/                         # Index sqlite state and manifest
│  └─ eval/                          # Evaluation output directory
├─ .env.example                      # Environment template
├─ CHANGELOG.md                      # Bilingual changelog
```

## About `data/` vs the Knowledge Base

- `data/uploads/`: system-managed raw file objects
- `data/processed/`: normalized parse snapshots
- `storage/`: the actual live retrieval and chat state

## References

- Structure notes: [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md)
- Changelog: [CHANGELOG.md](./CHANGELOG.md)
