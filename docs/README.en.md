# Engineering RAG Assistant

[涓枃](./README.md) | [English](./README.en.md)

A lightweight RAG assistant for engineering documents. The current version focuses on local usability, low hallucination risk, and source-grounded answers.

## Goals

- Answer questions across multiple documents
- Minimize hallucinations by grounding answers in retrieved context
- Return citations and source snippets
- Support streaming output
- Run locally on Windows during development

## Stack

- Python
- FastAPI
- LangChain loaders + splitters
- `sentence-transformers` / `all-MiniLM-L6-v2`
- FAISS
- DeepSeek API

## Current Features

- Ingest web pages, PDF, `.docx`, Markdown, CSV, JSON, and TXT
- Normalize text, chunk documents, embed, index, and retrieve
- Stream answers with citations
- Built-in lightweight Web UI
- CLI and API entrypoints

## Quick Start

Use a repository-local virtual environment. Do not install project dependencies into Conda `base`.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the app:

```powershell
.venv\Scripts\activate
uvicorn backend.src.api.main:app --reload
```

Then open:

- Web UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

Run tests:

```powershell
.venv\Scripts\activate
pytest
```

## API Endpoints

- `GET /`
- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `POST /ask/stream`

## Loader Strategy

- Web: `WebBaseLoader`
- PDF: `pymupdf4llm` by default, with MinerU as fallback for scanned or low-quality PDFs
- Word `.docx`: `UnstructuredWordDocumentLoader`
- Markdown: `UnstructuredMarkdownLoader`
- CSV: `CSVLoader`
- JSON: `JSONLoader`
- TXT: simple text loader

## Current Limitations

- Single knowledge base for now
- PDF ingestion preserves page markers, page boundaries, reading order, and figure/table/caption relationships before chunking
- Complex or scanned PDFs still depend on fallback parsing
- Retrieval filtering can still be improved, for example with `doc_id` or `source_type`
- The repository has minimal CI, but no full deployment pipeline yet

## References

- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
- Project rules: [AGENTS.md](/D:/bailing-agent/AGENTS.md)

## Chat Sessions

- `POST /ask/stream` accepts an optional `session_id`.
- If the client does not send one, the API creates or reuses a `rag_session_id` cookie so browser follow-up questions keep conversation context.
- Persisted messages can be read from `GET /sessions/{session_id}/messages`.
- Chat history is stored in SQLite at `storage/chat_history.sqlite3` by default.


## CLI Scripts

Build an index from local files:

```powershell
.venv\Scripts\activate
python -m backend.scripts.build_index data\sample.txt
```

Ask from CLI:

```powershell
.venv\Scripts\activate
python -m backend.scripts.ask "What does this document say?"
```

