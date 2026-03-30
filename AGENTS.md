# AGENTS.md - Engineering RAG Assistant Guide

This file defines the working rules for Codex and other AI coding agents in this repository.

## Project Goal

Build a lightweight Python RAG question answering assistant for engineering documents.

Core requirements:

- Answer questions across multiple documents.
- Keep hallucination rate low.
- Prefer simple and stable architecture over feature breadth.
- Use mature frameworks and common engineering practices.
- Run locally on Windows during development.
- Return answers with source references.
- Support streaming output.

## Product Scope

The first milestone is a minimal usable system, not a full platform.

In scope:

- Document ingestion for web pages, PDF, Word, Markdown, CSV, JSON, and TXT.
- Chunking, embedding, indexing, retrieval, and answer generation.
- Source-grounded answers with citations.
- Simple API service and a minimal CLI or test entrypoint.
- Basic offline evaluation for retrieval and answer quality.

Out of scope for the first phase:

- Multi-agent orchestration.
- Complex workflow engines unless clearly necessary.
- Heavy UI, multi-tenant auth, or distributed deployment.
- Full-site crawling, recursive crawling, or sitemap ingestion.
- Premature support for many vector databases.

## Non-Negotiable Principles

- Ground every answer in retrieved context. If evidence is weak, answer with uncertainty instead of fabricating.
- Return cited source snippets or source locations whenever possible.
- Prefer conservative generation over aggressive free-form answering.
- Keep dependencies lean. Do not add large frameworks without clear justification.
- Prefer boring, maintainable solutions.

## Technical Direction

Language:

- Python only.

Recommended baseline stack:

- Orchestration: LangChain.
- API: FastAPI.
- Validation: Pydantic.
- Vector store: Chroma for the first version.
- Embeddings: `text-embedding-3-small`.
- Retrieval: similarity search.
- LLM: DeepSeek API.
- Chunking: semantic chunking.
- Rerank: add only if evaluation shows retrieval precision needs help.
- Testing: pytest.

Reasoning:

- LangChain and FastAPI are mature enough, widely used, and sufficient for a lightweight first version.
- Chroma is lightweight enough for local development and provides simpler persistence and collection management than raw FAISS.
- Milvus is not justified in the first phase because it adds service and operations complexity.

## Environment Management

Use a project-local Python virtual environment for dependency isolation.

Rules:

- Do not install project dependencies into the Conda `base` environment.
- Use a repository-local virtual environment such as `.venv` or `venv`.
- Keep the project environment isolated from global Python and Conda environments.
- Add the local virtual environment directory to `.gitignore`.
- Document the chosen environment directory name in project docs.

Recommended commands:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Conda is used as a Python distribution provider on the machine, do not install project packages into `base`. The project runtime still belongs in the repository-local virtual environment.

## Document Loading Strategy

Supported first-phase inputs:

- Web pages
- PDF
- Word `.docx`
- Markdown
- CSV
- JSON
- TXT

Loader policy:

- Use mature, lightweight loaders by default for text-first formats.
- Do not route every file type through one heavy parser.
- Keep loader selection explicit by file type.

Default loader mapping in phase one:

- Web pages: `WebBaseLoader`
- PDF: `PyPDFLoader` by default
- Word `.docx`: `UnstructuredWordDocumentLoader`
- Markdown: `UnstructuredMarkdownLoader`
- CSV: `CSVLoader`
- JSON: `JSONLoader`
- TXT: a simple text loader

Web loading policy:

- First phase supports fetching main text content from a single URL.
- Do not implement full-site crawling, recursive crawling, or sitemap ingestion in the first phase.
- Prefer main body text extraction over page styling or navigation retention.

PDF parsing policy:

- For normal text PDFs, prefer a lighter parser first.
- For scanned PDFs or layout-heavy PDFs, allow an OCR-capable parser path.
- MinerU is allowed as the advanced PDF parsing option for scanned or complex PDFs.
- Do not make MinerU the universal default for all documents, because it adds weight and operational complexity.
- Use `PyPDFLoader` as the default PDF path.
- If extraction yields too little usable text, severe garbling, or poor text quality, fallback to MinerU.
- Keep a manual override so a document can be forced to use MinerU when needed.

Word parsing policy:

- First phase supports `.docx` only.
- Do not implement legacy `.doc` support unless there is a confirmed business need.
- Prefer a mature DOCX-oriented loader.

## Architecture Priorities

The initial pipeline should stay close to this shape:

1. Load documents.
2. Clean and normalize text.
3. Split into chunks with metadata.
4. Build embeddings and Chroma index.
5. Retrieve top-k chunks with similarity search.
6. Optionally rerank retrieved chunks.
7. Generate answer strictly from retrieved context.
8. Stream the answer to the client.
9. Return answer plus citations.

Prefer a single retrieval pipeline first:

- Similarity retrieval first.
- Add hybrid retrieval only if evaluation shows clear gaps.
- Add rerank only if top-k recall is acceptable but answer precision is weak.

Vector store selection policy:

- Default to Chroma in phase one.
- Keep a clean adapter boundary so FAISS can still be evaluated later if needed.
- Consider Milvus only for later service-oriented or larger-scale deployment.
- Do not switch vector stores without evaluation evidence.

## Low-Hallucination Requirements

All implementations should optimize for factual grounding.

Required behaviors:

- Use a prompt that tells the model to answer only from provided context.
- If the answer is not supported by the retrieved material, say so explicitly.
- Include references to the chunks or source documents used.
- Keep retrieval metadata attached through the full pipeline.
- Avoid summarizing beyond the evidence.
- Support streaming output without losing citations or answer structure.

Preferred answer policy:

- Short answer first.
- Then supporting evidence.
- Then source references.

## Suggested Repository Layout

Keep the layout small and explicit:

```text
src/
  api/
  core/
  ingest/
  retrieve/
  generate/
  eval/
  models/
tests/
docs/
scripts/
data/
```

Conventions:

- `src/ingest/`: loaders, parsing, chunking, metadata extraction.
- `src/retrieve/`: embedding, indexing, retrieval, rerank.
- `src/generate/`: prompt building and answer generation.
- `src/eval/`: offline evaluation utilities and benchmark scripts.
- `scripts/`: developer scripts only, not core business logic.
- `data/`: local sample data only; avoid committing sensitive source documents.

## Coding Rules

- Keep functions small and composable.
- Prefer explicit data models over loose dictionaries.
- Add type hints for public functions and core pipeline modules.
- Avoid hidden global state.
- Isolate model providers and vector store adapters behind clear interfaces.
- Keep document loaders modular so PDF OCR and normal text extraction can evolve independently.
- Write code that can be swapped from one embedding model or LLM to another with minimal change.

Do not:

- Hardcode secrets, tokens, or machine-specific paths.
- Couple ingestion, retrieval, and generation tightly in one file.
- Introduce async complexity unless it materially improves the API layer.

## Evaluation Expectations

No RAG work is complete without at least lightweight evaluation.

Minimum evaluation expectations:

- Maintain a small gold dataset of questions and expected source documents.
- Measure retrieval hit quality for top-k results.
- Manually inspect hallucination cases.
- Record failure cases before adding more complexity.
- Test that citations remain correct in streamed responses.

Useful evaluation dimensions:

- retrieval hit rate
- citation correctness
- unsupported answer rate
- latency

## Developer Commands

These commands are the baseline workflow:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
pytest
uvicorn src.api.main:app --reload
```

If a script is added for indexing, keep it explicit, for example:

```powershell
python -m scripts.build_index
```

## Documentation Expectations

Keep docs focused on execution:

- How to create and activate the local virtual environment.
- How to configure model keys or local model settings.
- How to ingest documents.
- How to build or refresh the index.
- How to run tests and evaluation.
- Which loaders are used for each supported file type.
- Known limitations and current hallucination risks.

## Tooling Guidance

Use Context7 as a documentation assistant when implementing with mature frameworks.

Allowed uses:

- Look up current LangChain, FastAPI, Pydantic, Chroma, and testing documentation.
- Check idiomatic API usage before introducing new framework code.
- Verify version-specific usage when local knowledge may be stale.

Do not use Context7 for:

- Replacing project architecture decisions.
- Inventing complexity not required by this repository.
- Treating third-party examples as authoritative without adapting them to this project's lightweight goals.

## Decision Heuristics For Agents

When making implementation choices:

- Choose the simpler option unless evaluation data justifies more complexity.
- Prefer local, inspectable components over opaque platform dependencies.
- Prefer changes that improve grounding, observability, and testability.
- If a feature increases hallucination risk, reject it or gate it behind evaluation.

## Reference Material

This project may refer to the Deeptoai RAG docs for architecture ideas and terminology:

- https://rag.deeptoai.com/docs

Use that site as a reference, not as a mandate. This repository should remain lightweight and practical.
