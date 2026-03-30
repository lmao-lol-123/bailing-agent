# Engineering RAG Assistant

A lightweight Python RAG question answering assistant for engineering documents.

## Goals

- Answer questions across multiple documents.
- Keep hallucination rate low.
- Return answers with source references.
- Stay lightweight and practical for local development on Windows.

## First-Phase Scope

- Input types: web pages, PDF, Word `.docx`, Markdown, CSV, JSON, TXT
- Vector store: Chroma when installed, otherwise in-memory fallback for local development and tests
- Embeddings: `text-embedding-3-small`
- Retrieval: similarity search
- LLM: DeepSeek API
- Output: streaming answers with citations

## Environment Setup

This project uses a repository-local virtual environment. Do not install project dependencies into the Conda `base` environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional Chroma install on Windows:

```powershell
pip install -r requirements-chroma.txt
```

Note: `chromadb` may require native build support on Windows + Python 3.12. The codebase automatically falls back to an in-memory vector store when Chroma is unavailable, so tests and local API development can still run.

## Development Workflow

```powershell
.venv\Scripts\activate
pytest
uvicorn src.api.main:app --reload
```

## API Endpoints

- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `POST /ask/stream`

## Document Loading Plan

- Web pages: `WebBaseLoader`
- PDF: `PyPDFLoader` by default, fallback to MinerU for scanned or poor-quality extraction
- Word `.docx`: `UnstructuredWordDocumentLoader`
- Markdown: `UnstructuredMarkdownLoader`
- CSV: `CSVLoader`
- JSON: `JSONLoader`
- TXT: simple text loader

## Scripts

- `python -m scripts.build_index <paths...>`
- `python -m scripts.ask "your question"`

## Git Rules

- Do not commit `.venv/` or other local environment directories.
- Do not commit `.env` or API keys.
- Do not commit private raw documents.
- Do not commit generated Chroma data unless intentionally versioning a sample dataset.
- Run basic tests before pushing to GitHub.

## Reference

- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
- Project rules: [AGENTS.md](/D:/bailing-agent/AGENTS.md)
