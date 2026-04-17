# Bailing Agent

[中文](./README.md) | [English](./README.en.md)

A backend-first RAG project for engineering documents and standards-style PDFs, focused on local reliability, explainable retrieval, source-grounded answers, session-scoped file isolation, and reproducible offline evaluation.

## Overview

This repository is not centered on a generic chat UI. The main value is the end-to-end backend pipeline:

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

Install evaluation dependencies:

```powershell
uv sync --group eval
```

Run RAGAS evaluation:

```powershell
uv run python -m backend.scripts.run_ragas_eval --dataset backend/tests/fixtures/eval_dataset.json
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
│  ├─ .pytest-tmp/                   # pytest runtime artifacts, not source code
│  └─ test_runtime/                  # Temporary runtime fixtures for tests
├─ data/                             # Raw document objects and processed snapshots, not the live KB itself
│  ├─ uploads/                       # Raw content-addressed file objects
│  └─ processed/                     # Normalized parse snapshots such as doc-xxxx.normalized.json
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
├─ .gitignore                        # Git ignore rules
├─ CHANGELOG.md                      # Bilingual changelog
├─ pyproject.toml                    # Python/uv/ruff project config
├─ pytest.ini                        # pytest config
├─ README.en.md                      # Main English README
├─ README.md                         # Main Chinese README
└─ uv.lock                           # uv lockfile
```

## About `data/` vs the Knowledge Base

`data/` is not a “drop files here and they become the knowledge base” folder.

A more accurate model is:

- `data/uploads/`: system-managed raw file objects
- `data/processed/`: normalized parse snapshots
- `storage/`: the actual live retrieval and chat state

So in practice:

- `data` is for source files and intermediate artifacts
- `storage` is for the active RAG runtime state

## Global Knowledge Base vs Chat Uploads

The project distinguishes two logical document scopes:

- Global knowledge-base documents
  - visible to all sessions
  - logically `global scope`

- Chat-uploaded documents
  - visible only inside the owning session
  - logically `session scope`
  - constrained by `session_id`

These two categories may reuse the same underlying raw file object, but they must not share the same retrieval visibility boundary. In other words:

- physical storage can be deduplicated
- retrieval permissions must stay isolated

## Current Validation Status

- Full backend test suite: `104 passed`
- `ruff check`: passed
- DeepSeek streaming answer path: validated
- DeepSeek-backed RAGAS evaluation flow: enabled

## GitHub Push Notes

The following should not be pushed to GitHub:

- `.env`
- runtime data under `data/`
- runtime data under `storage/`
- `AGENTS.md`
- temporary test artifacts such as `backend/.pytest-tmp/` and `test_runtime/`

Before pushing, it is still a good idea to verify:

```powershell
git status --short
```

## References

- Structure notes: [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md)
- Changelog: [CHANGELOG.md](./CHANGELOG.md)
