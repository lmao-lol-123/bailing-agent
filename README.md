# Engineering RAG Assistant

A lightweight Python RAG question answering assistant for engineering documents.

## Goals

- Answer questions across multiple documents.
- Keep hallucination rate low.
- Return answers with source references.
- Stay lightweight and practical for local development on Windows.

## First-Phase Scope

- Input types: web pages, PDF, Word `.docx`, Markdown, CSV, JSON, TXT
- Vector store: FAISS
- Embeddings: `all-MiniLM-L6-v2`
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

## Development Workflow

```powershell
.venv\Scripts\activate
pytest
uvicorn src.api.main:app --reload
```

启动后可直接打开 `http://127.0.0.1:8000/` 使用内置前端页面。

## API Endpoints

- `GET /`
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

## Runtime Config

- `SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2`
- `DEEPSEEK_API_KEY=...`
- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_MODEL=deepseek-chat`
- `FAISS_INDEX_DIRECTORY=storage/faiss`

## Git Rules

- Do not commit `.venv/` or other local environment directories.
- Do not commit `.env` or API keys.
- Do not commit private raw documents.
- Do not commit generated FAISS index data unless intentionally versioning a sample dataset.
- Run basic tests before pushing to GitHub.

## Reference

- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
- Project rules: [AGENTS.md](/D:/bailing-agent/AGENTS.md)

