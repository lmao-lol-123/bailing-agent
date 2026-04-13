# Engineering RAG Assistant

[中文](./README.md) | [English](./README.en.md)

面向工程文档的轻量级 RAG 助手，当前阶段以后端能力为核心，强调本地可运行、检索可解释、回答带引用、流式输出稳定。

## 功能特性

- 多格式接入：支持 Web、PDF、Word `.docx`、Markdown、CSV、JSON、TXT
- 后端优先：围绕 ingestion、chunking、retrieval、generation、evaluation 和 API 稳定性建设
- 混合检索：FAISS 向量召回结合 BM25 词法召回、查询改写、路由和 rerank
- 引用可追溯：回答附带 citation，检索 metadata 贯穿索引、召回和生成
- 流式问答：`POST /ask/stream` 基于 SSE 输出 token、sources 和完成事件
- 会话持久化：SQLite 存储会话与消息，支持多轮追问

## 技术栈

| 组件 | 技术 |
| --- | --- |
| API | FastAPI, Pydantic, StreamingResponse |
| Ingestion | LangChain loaders, pymupdf4llm, Unstructured |
| Chunking | 结构优先切块 + 预算约束切分 |
| Retrieval | sentence-transformers, FAISS, rank-bm25 |
| Generation | DeepSeek API via OpenAI-compatible SDK |
| Persistence | SQLite, local JSON snapshots |
| Testing | pytest, pytest-asyncio, FastAPI TestClient |

## 当前后端能力

- 文档导入后会做归一化，并将快照写入 `data/processed/`
- 检索链路已支持查询路由、确定性查询扩展、混合召回、启发式 rerank 和定向重试
- 子块命中后会尽量回填父块正文，提升回答上下文完整性
- 当证据不足时，生成层会明确返回无法确认，而不是编造答案
- DeepSeek 真实 API key 联调已验证通过，流式生成链路可用

## 快速开始

推荐使用仓库本地虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

复制环境变量模板并填写 DeepSeek 配置：

```powershell
Copy-Item .env.example .env
```

`.env` 关键项：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

启动 API：

```powershell
.venv\Scripts\activate
uvicorn backend.src.api.main:app --reload
```

常用入口：

- Web/API 入口：`http://127.0.0.1:8000/`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`

## 常用命令

构建索引：

```powershell
.venv\Scripts\activate
python -m backend.scripts.build_index data\sample.txt
```

CLI 提问：

```powershell
.venv\Scripts\activate
python -m backend.scripts.ask "What does this document say?"
```

运行测试：

```powershell
.venv\Scripts\activate
python -m pytest -q
```

## API 概览

- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `GET /sessions`
- `GET /sessions/{session_id}/messages`
- `PATCH /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /ask/stream`

`POST /ask/stream` 支持 `question`、可选 `top_k`、可选 `session_id` 和可选 `metadata_filter`。

## 项目结构

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

## 验证状态

- 后端完整测试：`60 passed`
- DeepSeek 真实流式生成联调：已通过
- CI：GitHub Actions 在 `master`、`main` 和 `codex/**` 分支触发测试

## 仓库约定

- `data/uploads/`、`data/processed/` 和 `storage/` 属于本地运行产物，不提交到 GitHub
- 当前阶段前端为维护模式，本仓库迭代以 backend 为主
- 如果证据不足，系统优先返回不确定，而不是自由发挥

## 参考

- 项目规则：[AGENTS.md](./AGENTS.md)
- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
