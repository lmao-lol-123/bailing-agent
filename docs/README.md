# Engineering RAG Assistant

[中文](./README.md) | [English](./README.en.md)

面向工程文档的轻量级 RAG 问答助手。当前阶段以后台能力为核心，优先保证本地可运行、低幻觉、来源可追溯、流式输出稳定。

## 1. 项目定位

项目目标：

- 面向工程文档提供问答能力
- 支持多文档、多格式统一接入
- 以检索证据为依据，尽量降低幻觉
- 回答中携带来源引用与片段
- 支持流式输出，适合 API 和轻量本地 UI
- 本地开发环境以 Windows 为主

当前阶段范围：

- 以后端为主，前端仅维护，不做重点投入
- 当前采用单知识库模式
- 优先打磨 ingestion、chunking、retrieval、generation、evaluation 和 API 稳定性

## 2. 当前技术栈

基础栈：

- Python
- FastAPI
- Pydantic
- pydantic-settings
- LangChain loaders + splitters
- sentence-transformers
- all-MiniLM-L6-v2
- FAISS
- DeepSeek API
- OpenAI-compatible SDK
- SQLite
- pytest
- pytest-asyncio

主要组件：

- API 层：FastAPI、StreamingResponse、SSE
- 配置与模型：Pydantic、pydantic-settings
- 文档加载：LangChain 社区 loaders
- 文本切分：结构化切分 + RecursiveCharacterTextSplitter，必要时语义辅助
- 向量化：sentence-transformers / `all-MiniLM-L6-v2`
- 向量库：FAISS
- 生成模型接入：DeepSeek API
- 会话存储：SQLite
- 测试：pytest、FastAPI TestClient

## 3. 当前支持的文档类型

- Web page
- PDF
- Word `.docx`
- Markdown
- CSV
- JSON
- TXT

当前 loader 映射：

- Web：`WebBaseLoader`
- PDF：默认 `pymupdf4llm`，扫描件或低质量文本使用 MinerU fallback
- Word `.docx`：`UnstructuredWordDocumentLoader`
- Markdown：`UnstructuredMarkdownLoader`
- CSV：`CSVLoader`
- JSON：`JSONLoader`
- TXT：`TextLoader`

## 4. 当前系统架构

主链路：

1. 加载文档
2. 文本归一化
3. 按结构优先进行分块
4. 为 chunk 生成 embedding
5. 写入 FAISS 索引
6. 检索阶段先做向量召回
7. 按 metadata 做过滤
8. 对召回结果进行 rerank
9. 将检索上下文送入 DeepSeek 生成回答
10. 通过 SSE 流式返回答案与引用

目录结构：

- [backend/src/api](/D:/bailing-agent/backend/src/api)
  API 路由、请求模型、SSE 输出、入口装配
- [backend/src/core](/D:/bailing-agent/backend/src/core)
  配置、依赖注入、领域模型、聊天历史、通用文本工具
- [backend/src/ingest](/D:/bailing-agent/backend/src/ingest)
  文档加载、归一化、分块、导入服务
- [backend/src/retrieve](/D:/bailing-agent/backend/src/retrieve)
  embedding provider、FAISS、检索与 citation 构造
- [backend/src/generate](/D:/bailing-agent/backend/src/generate)
  prompt 与回答生成服务
- [backend/src/eval](/D:/bailing-agent/backend/src/eval)
  离线评测数据模型与加载
- [backend/tests](/D:/bailing-agent/backend/tests)
  单元与集成测试
- [frontend](/D:/bailing-agent/frontend)
  轻量静态前端，当前仅维护

## 5. 后端当前实现状态

### 5.1 Ingestion

已实现：

- 多格式文档加载
- 文本归一化
- PDF 默认走 `pymupdf4llm`，保留页码、页面边界与阅读顺序；扫描件或低质量文本时回退到 MinerU
- 本地文件与 URL 两种接入方式
- normalized 文档快照写入 `data/processed/`

### 5.2 Chunking

已调整为“结构优先，语义辅助”：

- 优先按文档结构切块：标题层级、段落、列表、表格、代码块
- 对超过预算的块再做二次切分
- 在必要场景下用语义切分辅助长段落拆分
- chunk 大小受 embedding 模型 token 预算约束，避免 `all-MiniLM-L6-v2` 提前截断造成信息损失

### 5.3 Metadata

当前 chunk / citation 相关 metadata 已覆盖：

- `doc_id`
- `source_name`
- `source_uri_or_path`
- `title`
- `section_path`
- `page_or_section`
- `page`
- `doc_type`
- `updated_at`
- `block_type`

这些 metadata 用于：

- 提高检索过滤精度
- 让引用位置更稳定
- 支撑后续评测与错误分析

### 5.4 Retrieval

当前检索策略：

- FAISS 向量召回
- metadata 过滤
- 召回后 rerank

当前实现特点：

- 小规模数据场景下，优先采用稳定基线方案
- 当前索引路径为 `storage/faiss/default_index`
- 候选集先放大召回，再裁剪到最终 top-k
- rerank 使用轻量词项重叠信号与向量分数融合，先做可维护基线

### 5.5 Generation

当前生成策略：

- 严格基于检索上下文生成回答
- 无足够证据时返回不确定结果，而不是编造
- 输出中保留 citation 编号
- API 使用 SSE 流式输出 token 和 sources

### 5.6 Chat Session

已实现：

- `POST /ask/stream` 支持可选 `session_id`
- 如果客户端不传，会复用或创建 `rag_session_id` cookie
- 会话消息持久化在 SQLite
- 可通过接口读取历史消息和会话列表

## 6. 已完成事项

当前已完成的关键能力：

- 后端基础 API 路由
- 多格式文档 ingestion
- 文本归一化与落盘
- 结构优先分块
- chunk metadata 补齐
- sentence-transformers embedding
- FAISS 建库与检索
- metadata 过滤
- 召回后 rerank 基线
- DeepSeek 流式生成
- citation 构造
- chat session 持久化
- 后端定向测试覆盖核心链路

近期已验证通过的测试方向：

- chunking
- loaders
- vector store
- generate service
- API
- chat store

## 7. 当前 API 概览

核心接口：

- `GET /`
- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `GET /sessions`
- `GET /sessions/{session_id}/messages`
- `PATCH /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /ask/stream`

说明：

- `POST /ask/stream` 支持 `question`
- 支持可选 `top_k`
- 支持可选 `session_id`
- 支持可选 `metadata_filter`

## 8. 当前限制

- 当前仍是单知识库，不支持多库隔离
- rerank 目前是轻量基线，不是专门训练的 cross-encoder
- FAISS 索引策略以稳定基线为主，尚未做多索引方案比较
- 对扫描件、复杂版面 PDF 仍依赖 fallback 质量
- 前端不是当前重点，交互能力较轻
- 离线评测体系已预留目录，但还需要继续完善 gold dataset 与指标沉淀

## 9. 下一步建议

后端优先级建议：

1. 完善离线评测数据集
2. 增加 retrieval hit rate / citation correctness / unsupported-answer rate / latency 指标
3. 为 rerank 引入可插拔接口，便于后续替换为更强模型
4. 增强 metadata 过滤能力，例如按 source、doc_id、时间范围、章节前缀组合过滤
5. 为结构化 chunking 增加更多文档类型特化规则
6. 补充 ingestion 和 retrieval 的失败案例样本库
7. 增加索引重建、增量导入和诊断脚本

## 10. 本地开发

推荐使用仓库本地虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

启动 API：

```powershell
.venv\Scripts\activate
uvicorn backend.src.api.main:app --reload
```

运行定向测试：

```powershell
.venv\Scripts\activate
python -m pytest backend\tests\test_api.py -q
python -m pytest backend\tests\test_vector_store.py -q
```

CLI 示例：

```powershell
.venv\Scripts\activate
python -m backend.scripts.build_index data\sample.txt
python -m backend.scripts.ask "What does this document say?"
```

## 11. 参考

- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)
- Project rules: [AGENTS.md](/D:/bailing-agent/AGENTS.md)

