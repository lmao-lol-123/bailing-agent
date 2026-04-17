# Bailing Agent

[中文](./README.md) | [English](./README.en.md)

面向工程文档与规范类 PDF 的后端优先 RAG 项目，强调本地可运行、检索可解释、引用可追溯、会话文件隔离，以及可离线评测的工程化链路。

## 项目简介

当前项目的重点不在“通用聊天 UI”，而在一条完整的文档问答后端链路：

- 文档导入与清洗
- 结构感知 chunking
- 混合检索、路由、重排、重试
- 引用约束生成
- 会话文件上传与隔离
- DeepSeek 驱动的 RAGAS 评测

项目现已使用 `uv` 管理依赖、虚拟环境与锁文件。

## 主要特性

- 多格式导入：支持 PDF、Word、Markdown、CSV、JSON、TXT 和网页
- 混合检索：FAISS 向量召回 + BM25 词法召回 + 查询扩展 + 路由 + rerank
- 结构增强：对规范类条文补充 `clause_id`、`table_id`、数值锚点、英文术语等元数据
- 会话上传：聊天时上传的文件按会话隔离，避免跨会话暴露
- 引用校验：回答要求带 citation，弱证据时优先保守回答
- 评测链路：支持 DeepSeek 作为评测模型跑 RAGAS，并输出离线检索指标
- 存储治理：原始文件按内容哈希去重存储，减少 `data/uploads` 重复副本

## 技术栈

| 模块 | 技术 |
| --- | --- |
| API | FastAPI, Pydantic, SSE |
| Ingestion | LangChain loaders, pymupdf4llm, Unstructured, MinerU fallback |
| Retrieval | sentence-transformers, FAISS, rank-bm25 |
| Generation | DeepSeek API via OpenAI-compatible SDK |
| Persistence | SQLite, local JSON snapshots |
| Tooling | uv, pytest, ruff |

## 快速开始

如未安装 `uv`：

```powershell
winget install --id=astral-sh.uv -e
```

初始化环境：

```powershell
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

最少需要配置：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

启动服务：

```powershell
uv run uvicorn backend.src.api.main:app --reload
```

入口：

- Web/API：`http://127.0.0.1:8000/`
- OpenAPI：`http://127.0.0.1:8000/docs`

## 常用命令

构建索引：

```powershell
uv run python -m backend.scripts.build_index path\to\document.pdf
```

CLI 提问：

```powershell
uv run python -m backend.scripts.ask "这份文档的核心结论是什么？"
```

运行全量测试：

```powershell
uv run pytest
```


整理历史 `data` 存储：

```powershell
uv run python -m backend.scripts.cleanup_data_storage
uv run python -m backend.scripts.cleanup_data_storage --apply
```

## API 概览

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

下面这部分按“目录 + 职责说明”的方式写，便于快速理解项目边界。

```text
bailing-agent/
├─ .github/
│  └─ workflows/
│     └─ ci.yml                      # GitHub Actions，负责 CI 测试
├─ backend/
│  ├─ scripts/                       # 命令行脚本：建索引、提问、评测、存储整理、恢复工具
│  ├─ src/
│  │  ├─ api/                        # FastAPI 路由与 API schema
│  │  ├─ core/                       # 配置、依赖注入、公共模型、聊天存储、文本工具
│  │  ├─ eval/                       # 离线评测数据结构与样本加载
│  │  ├─ generate/                   # 回答生成、引用校验、Prompt 组织
│  │  ├─ ingest/                     # 文档导入、PDF 解析、清洗、chunking、对象存储
│  │  ├─ models/                     # 预留模型包，当前基本为空壳
│  │  └─ retrieve/                   # 检索、索引管理、路由、扩展、重排、重试
│  ├─ tests/                         # 后端测试
│  └─ test_runtime/                  # 运行时测试辅助目录
├─ data/                             # 文档原件与解析快照存储
│  ├─ uploads/                       # 原始文件对象存储
│  └─ processed/                     # 标准化解析快照
├─ docs/
│  ├─ README.md                      # 额外中文说明
│  ├─ README.en.md                   # 额外英文说明
│  ├─ PROJECT_STRUCTURE.md           # 详细项目结构说明
│  └─ roadmap.txt                    # 路线草稿
├─ frontend/
│  ├─ index.html                     # 静态页面骨架
│  ├─ app.js                         # 前端交互逻辑
│  └─ styles.css                     # 页面样式
├─ storage/                          # RAG 运行时状态
│  ├─ chat_history.sqlite3           # 聊天记录和会话文件元数据
│  ├─ faiss/                         # 向量索引文件
│  ├─ index/                         # 索引状态 sqlite 与 manifest
│  └─ eval/                          # 评测输出目录
├─ .env.example                      # 环境变量模板
├─ pyproject.toml                    # Python/uv/ruff 项目配置
```

## 关于 `data/` 和知识库

- `data/uploads/`：系统管理的原始文件对象存储
- `data/processed/`：解析后的标准化快照
- `storage/`： RAG 运行时知识库和聊天状态


## 参考文档

- 结构说明：[docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md)
- 变更记录：[CHANGELOG.md](./CHANGELOG.md)
