# Engineering RAG Assistant

[中文](./README.md) | [English](./README.en.md)

面向工程文档的轻量级 RAG 问答助手。当前版本以“本地可运行、低幻觉、可追溯来源”为核心目标，优先保证简单、稳定、可维护。

## 项目目标

- 支持多文档问答
- 降低幻觉，尽量只基于检索上下文作答
- 返回来源引用与片段
- 支持流式输出
- 适合 Windows 本地开发

## 当前技术栈

- Python
- FastAPI
- LangChain loaders + splitters
- `sentence-transformers` / `all-MiniLM-L6-v2`
- FAISS
- DeepSeek API

## 当前能力

- 支持导入网页、PDF、`.docx`、Markdown、CSV、JSON、TXT
- 支持文档清洗、分块、向量化、索引、相似度检索
- 支持问答流式输出和来源引用
- 内置轻量 Web UI
- 提供 CLI 与 API 两种入口

## 快速开始

项目使用仓库本地虚拟环境，不要把依赖安装到 Conda `base`。

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

启动服务：

```powershell
.venv\Scripts\activate
uvicorn src.api.main:app --reload
```

启动后访问：

- Web UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

运行测试：

```powershell
.venv\Scripts\activate
pytest
```

## API 入口

- `GET /`
- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `POST /ask/stream`

## 文档加载策略

- Web: `WebBaseLoader`
- PDF: 默认 `PyPDFLoader`，必要时 fallback 到 MinerU
- Word `.docx`: `UnstructuredWordDocumentLoader`
- Markdown: `UnstructuredMarkdownLoader`
- CSV: `CSVLoader`
- JSON: `JSONLoader`
- TXT: 简单文本加载器

## 当前限制

- 当前默认是单知识库
- 检索以相似度搜索为主，rerank 尚未默认启用
- 复杂 PDF 和扫描件仍依赖 fallback 方案
- 检索过滤能力仍可继续增强，例如 `doc_id` / `source_type` 过滤
- 目前只有最小 CI，尚未配置真正的自动部署

## 参考

- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
- 项目规则: [AGENTS.md](/D:/bailing-agent/AGENTS.md)
