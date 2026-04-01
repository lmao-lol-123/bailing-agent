# Engineering RAG Assistant

[中文](./README.md) | [English](./README.en.md)

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
uvicorn src.api.main:app --reload
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
- PDF: `PyPDFLoader` by default, MinerU as fallback when needed
- Word `.docx`: `UnstructuredWordDocumentLoader`
- Markdown: `UnstructuredMarkdownLoader`
- CSV: `CSVLoader`
- JSON: `JSONLoader`
- TXT: simple text loader

## Current Limitations

- Single knowledge base for now
- Similarity retrieval is the default; reranking is not enabled by default
- Complex or scanned PDFs still depend on fallback parsing
- Retrieval filtering can still be improved, for example with `doc_id` or `source_type`
- The repository has minimal CI, but no full deployment pipeline yet

## References

- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
- Project rules: [AGENTS.md](/D:/bailing-agent/AGENTS.md)
