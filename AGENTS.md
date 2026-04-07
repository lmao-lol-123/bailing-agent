# AGENTS.md - Engineering RAG Assistant Guide

This file defines working rules for Codex and other AI coding agents in this repository.

## Windows Tooling Rules

- If `apply_patch` fails, immediately fall back to PowerShell file writes and continue with tests. Do not retry alternative patch methods.
- All `shell_command` calls must use `login:false` to avoid `profile.ps1` execution policy errors.

## Project Goal

Build a lightweight Python RAG assistant for engineering documents.

Core requirements:

- Answer questions across multiple documents.
- Keep hallucination rate low.
- Prefer simple and stable architecture over feature breadth.
- Run locally on Windows during development.
- Return answers with source references.
- Support streaming output.

## Current Development Priority

Backend comes first.

- Prioritize backend ingestion, retrieval, generation, evaluation, and API reliability.
- Frontend under `frontend/` is maintenance-only for now.
- Do not spend time on UI refactors, new frontend features, or frontend architecture unless the user explicitly asks for it.
- If an API change requires UI compatibility, keep the frontend change minimal and local.

## Product Scope

In scope for the current phase:

- Document ingestion for web pages, PDF, Word `.docx`, Markdown, CSV, JSON, and TXT.
- Chunking, embedding, FAISS indexing, retrieval, and answer generation.
- Source-grounded answers with citations.
- FastAPI service and lightweight CLI scripts.
- Basic offline evaluation for retrieval and answer quality.

Out of scope for the current phase:

- Multi-agent orchestration.
- Complex workflow engines unless there is a clear need.
- Heavy frontend work, multi-tenant auth, or distributed deployment.
- Full-site crawling, recursive crawling, or sitemap ingestion.
- Premature support for multiple vector databases.

## Non-Negotiable RAG Principles

- Ground every answer in retrieved context.
- If evidence is weak or missing, answer with uncertainty instead of fabricating.
- Return cited snippets or source locations whenever possible.
- Keep retrieval metadata attached through the full pipeline.
- Prefer conservative generation over broad free-form answering.
- Support streaming output without losing citation structure.
- Keep dependencies lean and favor boring, maintainable components.

## Repository Layout

```text
backend/
  src/
    api/
    core/
    ingest/
    retrieve/
    generate/
    eval/
    models/
  scripts/
  tests/
frontend/
docs/
data/
storage/
```

Directory conventions:

- `backend/src/api/`: FastAPI routes, request/response schemas, SSE streaming, static entrypoint wiring.
- `backend/src/core/`: configuration, dependency assembly, shared data models, chat history store, text utilities.
- `backend/src/ingest/`: file/url loading, parser routing, text normalization, chunking, ingestion orchestration.
- `backend/src/retrieve/`: embedding provider setup, FAISS vector store adapter, similarity search, citation construction.
- `backend/src/generate/`: prompt templates and answer streaming service.
- `backend/src/eval/`: offline benchmark dataset models and loaders.
- `backend/src/models/`: reserved package for future model abstractions.
- `backend/scripts/`: developer CLI entrypoints only, not core business logic.
- `backend/tests/`: pytest suite and fixtures.
- `frontend/`: minimal static Web UI. Keep changes small unless explicitly requested.
- `docs/`: project documentation.
- `data/`: local sample/input documents only. Do not commit sensitive documents.
- `storage/`: local generated FAISS index and SQLite chat history.

## Module Tech Stack

| Module | Responsibility | Stack |
| --- | --- | --- |
| `backend/src/api` | HTTP API, SSE streaming, serving frontend entrypoint | FastAPI, Pydantic, StreamingResponse, StaticFiles |
| `backend/src/core` | Settings, DI container, domain models, chat persistence, text helpers | pydantic-settings, Pydantic, SQLite, Python stdlib |
| `backend/src/ingest` | Document loading, normalization, semantic chunking, ingestion orchestration | LangChain loaders, SemanticChunker, RecursiveCharacterTextSplitter, PyPDFLoader, Unstructured loaders |
| `backend/src/retrieve` | Embedding model creation, FAISS index, similarity search, citation formatting | sentence-transformers, all-MiniLM-L6-v2, FAISS, LangChain FAISS vector store |
| `backend/src/generate` | Prompt construction and grounded streaming answer generation | OpenAI SDK, DeepSeek API, FastAPI jsonable_encoder |
| `backend/src/eval` | Evaluation dataset schema and loader | Pydantic, JSON |
| `backend/scripts` | CLI ingestion and ask entrypoints | argparse, asyncio |
| `backend/tests` | Unit/integration tests | pytest, pytest-asyncio, FastAPI TestClient |
| `frontend` | Minimal browser UI | HTML, CSS, JavaScript |

## Technical Direction

Baseline stack:

- Language: Python
- API: FastAPI
- Validation: Pydantic
- Config: pydantic-settings
- Document orchestration/parsing: LangChain loaders and splitters
- Vector store: FAISS
- Embeddings: `all-MiniLM-L6-v2` via `sentence-transformers`
- Retrieval: similarity search first
- LLM: DeepSeek API through the OpenAI-compatible SDK
- Testing: pytest, pytest-asyncio

Decision rules:

- Choose the simpler option unless evaluation proves more complexity is needed.
- Keep a clean adapter boundary for model and vector-store providers.
- Do not switch away from FAISS or add rerank/hybrid retrieval without evaluation evidence.
- Avoid coupling ingestion, retrieval, and generation into a single large module.

## Document Loading Policy

Supported input types in this phase:

- Web pages
- PDF
- Word `.docx`
- Markdown
- CSV
- JSON
- TXT

Default loader mapping:

- Web pages: `WebBaseLoader`
- PDF: `PyPDFLoader` by default
- Word `.docx`: `UnstructuredWordDocumentLoader`
- Markdown: `UnstructuredMarkdownLoader`
- CSV: `CSVLoader`
- JSON: `JSONLoader`
- TXT: simple text loader

PDF policy:

- Prefer `PyPDFLoader` for normal text PDFs.
- Allow MinerU fallback for scanned or layout-heavy PDFs when extraction quality is poor.
- Keep manual override support for forcing MinerU on a document.
- Do not make MinerU the universal default.

Web policy:

- Only single-URL page loading is in scope for now.
- Do not implement full-site crawling, recursive crawling, or sitemap ingestion in this phase.

Word policy:

- Support `.docx` only for now.
- Do not add legacy `.doc` handling without a confirmed need.

## Expected Pipeline Shape

1. Load documents.
2. Clean and normalize text.
3. Split into chunks with metadata.
4. Build embeddings and FAISS index.
5. Retrieve top-k chunks by similarity.
6. Optionally rerank only after evaluation evidence.
7. Generate answers strictly from retrieved context.
8. Stream the answer to the client.
9. Return final answer plus citations.

Preferred answer format:

- Short answer first.
- Supporting evidence next.
- Source references last.

## Environment Management

Use a repository-local virtual environment only.

Rules:

- Do not install project dependencies into Conda `base`.
- Use `.venv` or another repo-local venv.
- Keep the venv directory in `.gitignore`.
- Keep test/runtime temp files inside the repository when possible.

Setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Developer Commands

Start API:

```powershell
.venv\Scripts\activate
uvicorn backend.src.api.main:app --reload
```

Build index from local files:

```powershell
.venv\Scripts\activate
python -m backend.scripts.build_index data\sample.txt
```

Ask from CLI:

```powershell
.venv\Scripts\activate
python -m backend.scripts.ask "your question"
```

## Test Execution Policy

Do not run the full test suite automatically during normal edits.

Rules:

- Prefer lightweight static checks or targeted tests relevant to the change.
- Do not invoke plain full `pytest` by default after every edit.
- If pytest is needed, keep runtime files under `backend/.pytest-tmp`.
- Current `pytest.ini` disables cacheprovider to avoid root-level `.pytest*` artifacts on Windows.

Examples:

```powershell
.venv\Scripts\activate
pytest
python -m pytest backend\tests\test_api.py -q
```

## Evaluation Expectations

Minimum RAG evaluation expectations:

- Keep a small gold dataset of questions and expected source documents.
- Measure retrieval hit quality for top-k results.
- Manually inspect hallucination failures.
- Record failure cases before adding retrieval/generation complexity.
- Verify that streamed responses keep citation correctness.

Useful metrics:

- Retrieval hit rate
- Citation correctness
- Unsupported-answer rate
- Latency

## Coding Rules

- Keep functions small and composable.
- Add type hints for public functions and core pipeline modules.
- Prefer explicit Pydantic/domain models over loose dictionaries.
- Avoid hidden global state.
- Keep model providers and vector-store adapters isolated behind clear interfaces.
- Keep document loaders modular by file type.
- Write code so embedding model or LLM provider changes are localized.
- Do not hardcode secrets, tokens, or machine-specific absolute paths.
- Avoid async complexity unless it materially improves the API layer.

## Documentation Expectations

Keep docs focused on execution and backend behavior:

- How to create the local venv.
- How to configure model keys.
- How to ingest documents and rebuild indexes.
- How to run API/CLI and targeted tests.
- Which loader each document type uses.
- Known limitations, hallucination risks, and evaluation status.

## Context7 Usage

Use Context7 only as a documentation lookup helper for mature framework APIs.

Allowed:

- Checking current FastAPI, Pydantic, LangChain, FAISS, sentence-transformers, or pytest usage.
- Verifying version-specific API details before coding.

Not allowed:

- Replacing project architecture decisions with third-party examples.
- Adding complexity just because a framework example exists.

## Reference

- Deeptoai RAG docs: https://rag.deeptoai.com/docs

Use that site as reference material, not as a mandate.


