# Changelog / 更新日志

## 2026-04-17

### 中文

#### 新增

- 新增 `uv` 项目管理与锁文件，统一依赖安装、测试和脚本运行方式
- 新增 `ruff` 检查与格式化配置，并完成仓库级代码整理
- 新增会话文件上传、删除恢复、状态流转与相关回归测试
- 新增 DeepSeek 驱动的 RAGAS 评测链路与专项测试
- 新增规范类 PDF 检索增强：
  - 规范条文锚点提取
  - `regulation` 路由
  - 启发式重排增强
  - 轻量抽取式回答
- 新增内容寻址对象存储：
  - 原始文件按 SHA-256 去重保存
  - `data/processed` 快照按 `doc_id` 稳定命名
- 新增历史数据整理脚本 `backend/scripts/cleanup_data_storage.py`
- 新增仓库结构说明文档 `docs/PROJECT_STRUCTURE.md`

#### 变更

- 检索链路已支持查询路由、确定性查询扩展、混合召回、rerank、retry、parent hydration
- 会话上传文档默认不会再暴露给全局检索
- PDF / 规范类问答的结构元数据更完整，检索相关性较之前有所提升
- 文件删除逻辑改为引用安全：
  - 共享对象不会被提前删除
  - 最后一个引用消失后才清理物理文件和快照
- `README.md` 与 `README.en.md` 重写为 GitHub 面向版本，并补充带注释的项目结构说明
- `pytest` 已过滤 `faiss-cpu` 在 Python 3.12 下的已知 SWIG 导入告警

#### 仓库治理

- `AGENTS.md` 不再计划推送到 GitHub
- `data/` 运行数据不再计划推送到 GitHub
- `.env` 保持不入库，避免 API Key 泄露

#### 验证

- 全量后端测试：`104 passed`
- `ruff check`：通过
- DeepSeek 流式问答链路：通过
- DeepSeek 驱动的 RAGAS 评测链路：可运行

### English

#### Added

- Added `uv` project management and lockfile support for dependency installation, testing, and script execution
- Added `ruff` linting and formatting configuration and applied repository-wide cleanup
- Added session file upload, delete-recovery flows, status transitions, and regression coverage
- Added a DeepSeek-backed RAGAS evaluation flow and dedicated tests
- Added regulation-style PDF retrieval improvements:
  - regulation anchor extraction
  - `regulation` query routing
  - heuristic rerank improvements
  - lightweight extractive answering
- Added content-addressed object storage:
  - raw files are deduplicated by SHA-256
  - processed snapshots are now stably named by `doc_id`
- Added the legacy storage cleanup script `backend/scripts/cleanup_data_storage.py`
- Added the repository structure guide `docs/PROJECT_STRUCTURE.md`

#### Changed

- The retrieval stack now includes query routing, deterministic expansion, hybrid recall, reranking, retry, and parent hydration
- Session-uploaded documents are no longer visible to global retrieval by default
- Structure metadata for standards-style PDFs is richer, improving retrieval relevance for regulation queries
- File deletion is now reference-safe:
  - shared file objects are preserved while still referenced
  - physical files and snapshots are only cleaned after the last reference is gone
- Rewrote `README.md` and `README.en.md` for GitHub-facing usage and added an annotated project structure section
- `pytest` now suppresses the known `faiss-cpu` SWIG import warnings on Python 3.12

#### Repository Hygiene

- `AGENTS.md` is no longer intended to be pushed to GitHub
- runtime data under `data/` is no longer intended to be pushed to GitHub
- `.env` remains excluded from version control to avoid leaking API keys

#### Validation

- Full backend test suite: `104 passed`
- `ruff check`: passed
- DeepSeek streaming answer path: validated
- DeepSeek-backed RAGAS evaluation flow: runnable
