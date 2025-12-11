# LLM Project – NYU Course Selection & Syllabus QA

**English | [中文介绍](#中文简介-chinese)**

A fast, memory-aware course advisor built with Flask + local Qwen + RAG. Instantly answers syllabus questions, recommends courses, and remembers prior chats.
- RAG over syllabi with Faiss + multilingual-e5-large.
- Qwen handles query rewrite, answering, course selection, and comparisons.
- Conversational memory (memories.json + Faiss) personalizes replies.
- Clean web UI calling `/ask`, with memory reset and retrieval debug endpoints.

## Core Files & Modules
- `app.py`: Flask entry; routes `/`, `/ask`, `/reset_memory`, `/debug_retrieval`.
- `advisor_module.py`: course selection/comparison + memory-only chat logic.
- `rag_module.py`: syllabus RAG (query rewrite, Faiss search, chunk ranking).
- `memory_module.py`: dialogue memory store/aggregation/retrieval (`memory_store/`).
- `qwen_client.py`: local Qwen HTTP client (default `http://127.0.0.1:8000/v1/chat/completions`).
- `ingest_syllabi.py`: parse `docs/syllabus/*.pdf` → `vector_store/index.faiss` + `texts.json`.
- `templates/index.html` + `static/nyu-logo.png`: web UI.
- `test.py`: unit tests with stubbed deps.
- `test_integration_qwen.py`: real Qwen round-trip (opt-in).

## Setup
- Python ≥ 3.10
- Install deps:
  ```bash
  pip install flask requests sentence-transformers faiss-cpu numpy pypdf pytest
  ```
- Configure Qwen endpoint/model in `config/paths.py`:
  - `QWEN_API_URL` (default `http://127.0.0.1:8000/v1/chat/completions`)
  - `QWEN_MODEL_NAME` (default `Qwen/Qwen3-4B-Instruct-2507-FP8`)
  - `E5_MODEL_NAME` (default `intfloat/multilingual-e5-large` for embeddings)

## Data & Vector Store
1) Drop syllabus PDFs into `docs/syllabus/`.
2) Build vectors (downloads models if absent):
   ```bash
   python ingest_syllabi.py
   ```
   Outputs:
   - `vector_store/index.faiss`
   - `vector_store/texts.json`
3) Conversation memory lives at `memory_store/memories.json` and `memory_store/mem_index.faiss`.
   - Reset via `/reset_memory` or the auto-reset on `/` (see `app.py`).

## Run
```bash
export FLASK_APP=app.py
python app.py  # default 0.0.0.0:5000, debug=True
```
Frontend at `http://localhost:5000/` (uses `templates/index.html`, calls `/ask`).

## API Quick View
- `POST /ask`: `{ "question": "..." }` (JSON/form). Routes to:
  - comparison → `answer_course_comparison_question`
  - selection → `answer_course_selection_question`
  - syllabus details → `call_qwen_with_rag`
  - general chat → `chat_with_memory_only`
- `POST /reset_memory`: clears `memory_store/*`.
- `POST /debug_retrieval`: returns refined query, retrieved chunks, `rag_module.LAST_RETRIEVAL_DEBUG`.

## Memory & Personalization
- Every turn calls `add_memory_from_turn`; aggregators capture career/profile signals.
- `retrieve_memories` mixes embedding similarity, time decay, and importance.
- Storage: `memory_store/memories.json`, `memory_store/mem_index.faiss`.

## Selection & Comparison Logic
- Course candidates in `advisor_module.py::COURSE_PROFILES` (also `config/courses.json`).
- Comparison extracts two course codes, retrieves both syllabi, then personalizes the recommendation.

## Tests
- Unit (no real model calls):
  ```bash
  pytest test.py -q
  ```
- Integration (needs reachable Qwen, `QWEN_TEST_ENABLED=1`):
  ```bash
  QWEN_TEST_ENABLED=1 pytest test_integration_qwen.py -q -m integration
  ```

## FAQ
- Missing vector store: run `python ingest_syllabi.py` and ensure `vector_store/index.faiss` + `texts.json`.
- Memory not updating: check `memory_store/` write access or call `/reset_memory`.
- Qwen errors: verify endpoint, model name, and connectivity.

---

## 中文简介 (Chinese)

这是一个基于 Flask + 本地 Qwen + RAG 的高效率选课与 syllabus 问答助手，强调“秒回 + 记忆个性化”：
- Faiss + multilingual-e5-large 做 syllabus 向量检索。
- Qwen 负责 query 改写、答案生成、选课和课程对比。
- 对话记忆（memories.json + Faiss）让回答贴合个人背景。
- 前端直接调用 `/ask`，并提供记忆重置与检索调试接口。

### 目录与核心文件
- `app.py`：Flask 入口；路由 `/`, `/ask`, `/reset_memory`, `/debug_retrieval`。
- `advisor_module.py`：选课/对比/记忆聊天逻辑。
- `rag_module.py`：syllabus RAG（改写、Faiss 检索、chunk 选取）。
- `memory_module.py`：对话记忆存储/聚合/检索（`memory_store/`）。
- `qwen_client.py`：本地 Qwen HTTP 调用。
- `ingest_syllabi.py`：解析 `docs/syllabus/*.pdf` → `vector_store/index.faiss` 与 `texts.json`。
- `templates/index.html` + `static/nyu-logo.png`：前端页面。
- `test.py` / `test_integration_qwen.py`：单测与可选的真实 Qwen 测试。

### 环境与配置
- Python ≥ 3.10
- 依赖安装：
  ```bash
  pip install flask requests sentence-transformers faiss-cpu numpy pypdf pytest
  ```
- 在 `config/paths.py` 配置 Qwen：`QWEN_API_URL`、`QWEN_MODEL_NAME`、`E5_MODEL_NAME`。

### 数据与向量库
1) 将 syllabus PDF 放入 `docs/syllabus/`。
2) 执行：
   ```bash
   python ingest_syllabi.py
   ```
   生成 `vector_store/index.faiss`、`vector_store/texts.json`。
3) 记忆文件：`memory_store/memories.json`、`memory_store/mem_index.faiss`（`/` 会自动清空，也可调 `/reset_memory`）。

### 运行
```bash
export FLASK_APP=app.py
python app.py
```
前端：`http://localhost:5000/`，调用 `/ask`。

### API 速览
- `POST /ask`：按意图路由（对比/选课/syllabus/纯记忆聊天）。
- `POST /reset_memory`：清空 `memory_store/*`。
- `POST /debug_retrieval`：查看改写 query、检索上下文及调试信息。

### 记忆与个性化
- 每轮用 `add_memory_from_turn` 写入，并通过聚合器提取职业方向/个人档案。
- `retrieve_memories` 综合相似度、时间衰减、重要度。

### 选课/对比逻辑
- 课程列表维护在 `advisor_module.py::COURSE_PROFILES`（亦有 `config/courses.json`）。
- 对比问题抓取两个课程号，分别检索 syllabus，再结合记忆给建议。

### 测试
- 快速单测：
  ```bash
  pytest test.py -q
  ```
- 集成（需可用 Qwen，`QWEN_TEST_ENABLED=1`）：
  ```bash
  QWEN_TEST_ENABLED=1 pytest test_integration_qwen.py -q -m integration
  ```

### 常见问题
- 找不到向量库：先跑 `python ingest_syllabi.py`，确保 `vector_store/index.faiss`、`texts.json` 存在。
- 记忆未更新：检查 `memory_store/` 写权限，或调用 `/reset_memory`。
- Qwen 调用失败：核对端点、模型名和网络连通性。
