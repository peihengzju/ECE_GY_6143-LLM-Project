# NYU Course Advisor (Flask + Qwen + RAG)

A fast syllabus QA and course advisor. It rewrites queries with Qwen, ranks syllabus chunks with Faiss + E5 embeddings, remembers prior turns, and can auto-build a weekly schedule.

## Highlights
- RAG over syllabus PDFs (`vector_store/index.faiss` + `vector_store/texts.json`).
- Memory-aware chat (`memory_store/*`) to keep user context.
- Course advisor logic (selection/comparison) + schedule generator with HTML preview.
- Strong course-code boosting in retrieval: chunks whose `meta.course`/`meta.file` match the queried course number are force-ranked to the top.

## Project Layout
- `app.py` — Flask entry; routes `/`, `/ask`, `/reset_memory`, `/debug_retrieval`.
- `advisor_module.py` — selection/comparison logic.
- `rag_module.py` — query rewrite, Faiss search, heuristic boosting, dominant-course filtering.
- `memory_module.py` — dialogue memory store/retrieval (`memory_store/`).
- `course_db.py` — loads structured courses from `config/courses.json` for schedule filling.
- `schedule_module.py` — parses course mentions, fills times via CourseDB, renders `templates/schedule.html`.
- `templates/index.html` — chat UI with inline schedule iframe preview; `templates/schedule.html` — weekly grid.
- `ingest_syllabi.py` — build vectors from `docs/syllabus/*.pdf`.
- `config/paths.py` — model and file paths; `config/courses.json` — course metadata.
- Tests: `test.py` (unit), `test_integration_qwen.py` (requires live Qwen, opt-in).

## Setup
1) Python >= 3.10  
2) Install deps:
```bash
pip install flask requests sentence-transformers faiss-cpu numpy pypdf pytest
```
3) Configure `config/paths.py` if needed:
- `QWEN_API_URL` (default points to local Qwen)
- `QWEN_MODEL_NAME`
- `E5_MODEL_NAME`

## Data Prep (RAG)
1) Put syllabus PDFs in `docs/syllabus/`.  
2) Build the vector store:
```bash
python ingest_syllabi.py
```
Outputs: `vector_store/index.faiss`, `vector_store/texts.json`.  
3) Conversation memory lives in `memory_store/memories.json` and `memory_store/mem_index.faiss` (reset via `/` or `/reset_memory`).

## Run
```bash
export FLASK_APP=app.py
python app.py   # serves on 0.0.0.0:5000 by default
```
Frontend: `http://localhost:5000/`.

## API Quick Reference
- `POST /ask` — `{ "question": "..." }`
  - Routes to comparison, selection, syllabus QA (RAG), or memory-only chat.
  - When schedule intent fires, returns `schedule_html` + `conflicts` and the UI embeds the schedule iframe.
- `POST /reset_memory` — clears conversation memory.
- `POST /debug_retrieval` — returns refined query, retrieved chunks, and `rag_module.LAST_RETRIEVAL_DEBUG`.

## Retrieval Notes
- `_compute_boost` in `rag_module.py` gives a large boost when the question mentions a course number that appears in `meta.course` or `meta.file`, ensuring the right course dominates even if other syllabi are semantically similar.
- Additional boosts consider type priority, grading/exam keywords, and direct course-code mentions.

## Schedule Flow
1) `schedule_module.extract_courses_from_text` parses course mentions.  
2) Missing times/rooms are filled from `CourseDB` (uses `config/courses.json`).  
3) `generate_schedule_html_from_courses` renders `templates/schedule.html`; the chat UI shows an iframe preview and a new-tab link.

## Tests
- Unit (no live model):
```bash
pytest test.py -q
```
- Integration (needs reachable Qwen):
```bash
QWEN_TEST_ENABLED=1 pytest test_integration_qwen.py -q -m integration
```

## Common Issues
- Missing vector store: rerun `python ingest_syllabi.py` and ensure `vector_store/index.faiss` + `texts.json` exist.
- Memory not updating: check write access to `memory_store/` or call `/reset_memory`.
- Qwen errors: verify endpoint, model name, and connectivity.
