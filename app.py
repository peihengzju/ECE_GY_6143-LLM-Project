# app.py
import os
import re
import json
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from flask import Flask, request, jsonify
from typing import Optional
# ====== RAG 部分配置 ======
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

E5_MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 10                 # 最终送给大模型的 chunk 数量
FAISS_RAW_K = 24           # 先从 Faiss 拿更多候选，再做二次排序
MAX_CONTEXT_CHARS = 3600   # 拼接给大模型的 syllabus 总长度上限

# ====== Qwen + vLLM OpenAI 接口配置 ======
QWEN_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
QWEN_MAX_TOKENS = 1024


# ====== 加载向量索引和文本 ======
print("[RAG] Loading Faiss index and texts ...")
faiss_index = faiss.read_index(INDEX_PATH)
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    DOCS: List[Dict] = json.load(f)
print(f"[RAG] Loaded {len(DOCS)} chunks")

emb_model = SentenceTransformer(E5_MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    """对用户问题做 embedding（e5 标准：query 前缀）"""
    text = f"query: {query}"
    emb = emb_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    return emb.astype("float32").reshape(1, -1)

def _infer_target_file(question: str, idx_array: np.ndarray) -> Optional[str]:
    """
    根据用户问题 + Faiss 检索结果，推断“最可能被问的是哪一个 syllabus 文件”。

    规则：
    1）如果问题里有课程号（ECE-GY 6143 / CS-GY 6923 等），优先匹配文本/文件名里含该课程号的文件；
    2）否则，用 Faiss 排名前面的结果统计哪个文件出现频率最高，就认为在问那门课。
    """
    q = question.lower()
    codes = _extract_course_codes(question)  # 你已经实现过的
    file_scores: Dict[str, float] = {}

    n = len(idx_array[0])
    for pos, i in enumerate(idx_array[0]):
        doc = DOCS[int(i)]
        meta = doc.get("meta", {})
        fname = meta.get("file", "unknown")
        text = (doc.get("text", "") or "").lower()

        # 越靠前的检索结果权重越大
        base_weight = float(n - pos)

        score = 0.0
        if codes:
            # 有课程号时，只给匹配到课程号的文件加分
            for code in codes:
                c = code.lower().replace(" ", "").replace("-", "")
                if c and (
                    c in text.replace(" ", "").replace("-", "") or
                    c in fname.lower().replace(" ", "").replace("-", "")
                ):
                    score += base_weight * 3.0
        else:
            # 没课程号时，靠前的结果都给一点分，用于选“出现最多的那个文件”
            score += base_weight

        if score > 0:
            file_scores[fname] = file_scores.get(fname, 0.0) + score

    if not file_scores:
        return None

    # 选得分最高的文件作为“目标课程”
    target_file, _ = max(file_scores.items(), key=lambda kv: kv[1])
    return target_file

# ====== 意图识别：映射到 meta["type"] ======

def detect_priority_types(question: str) -> set[str]:
    """
    根据用户问题，推断最可能需要哪些 syllabus chunk type。
    返回一个 type 集合，用来给这些 type 的 chunk 加权。
    """
    q = question.lower()
    types: set[str] = set()

    # 成绩 / 占比 / 考试
    if any(k in q for k in ["占比", "成绩", "评分", "grading", "grade", "weight", "percentage", "%"]):
        types.update({"grading_section", "grading_line", "project", "homework_lab"})
    if any(k in q for k in ["期末", "期中", "考试", "quiz", "exam", "midterm", "final"]):
        types.update({"exam", "grading_section", "grading_line", "project"})

    # 作业 / 实验 / project
    if any(k in q for k in ["作业", "homework", "assignment", "problem set", "ps ", "ps.", "lab", "labs"]):
        types.update({"homework_lab", "grading_section", "grading_line"})
    if any(k in q for k in ["项目", "project"]):
        types.update({"project", "grading_section", "grading_line"})

    # 出勤 / 上课形式 / 线上
    if any(k in q for k in ["出勤", "attendance", "participation"]):
        types.update({"online_format", "lecture_info"})
    if any(k in q for k in ["线上", "网课", "remote", "zoom", "online", "pre-recorded"]):
        types.update({"online_format", "lecture_info"})

    # 上课时间 / 地点 / schedule
    if any(k in q for k in ["上课时间", "几点", "几点上课", "哪天上课", "周几", "schedule", "课程安排", "每周讲什么"]):
        types.update({"schedule", "lecture_info"})
    if "final exam" in q or "midterm" in q:
        types.add("schedule")

    # 老师 / 助教
    if any(k in q for k in ["老师", "教授", "professor", "instructor", "office hour", "office hours"]):
        types.add("instructor")
    if any(k in q for k in ["助教", "ta", "grader"]):
        types.add("grader")

    # 先修课 / 难度
    if any(k in q for k in ["先修", "先修课", "prereq", "pre-requisite", "prerequisite"]):
        types.add("prerequisites")

    # 课程内容 / 学什么
    if any(k in q for k in ["课程内容", "学什么", "讲什么", "cover", "内容是什么", "介绍一下这门课"]):
        types.update({"course_description", "materials", "schedule"})

    # 课程材料 / GitHub / 资料
    if any(k in q for k in ["教材", "github", "资料", "slides", "lecture notes", "class material", "materials"]):
        types.update({"materials", "course_description"})

    return types


def _extract_course_codes(question: str) -> List[str]:
    """
    从问题里抽取类似 ECE-GY 6143 / CS-GY 6923 这样的课程号。
    """
    # 允许中间有空格或连字符
    pattern = re.compile(r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b")
    codes = pattern.findall(question)
    return [c.strip() for c in codes]


def _compute_boost(question: str, text: str, meta: Dict, base_rank: int,
                   priority_types: set[str], course_codes: List[str]) -> int:
    """
    综合关键词、type 匹配、课程号匹配，给每个候选 chunk 一个启发式加权分。
    """
    q = question.lower()
    t = text.lower()
    meta_type = meta.get("type", "normal")

    boost = 0

    # ===== 1) 关键词级 boost（原来的逻辑为基础） =====

    # 考试相关
    if any(k in q for k in ["期末", "期中", "考试", "quiz", "exam", "midterm", "final"]):
        if any(k in t for k in ["exam", "midterm", "final", "quiz", "test", "exams"]):
            boost += 3

    # 成绩 / 占比
    if any(k in q for k in ["占比", "成绩", "评分", "grading", "grade", "weight", "percentage", "%"]):
        if any(k in t for k in ["grading", "grade", "weight", "percentage", "%", "assessment", "evaluation"]):
            boost += 3

    # 作业 / 项目
    if any(k in q for k in ["作业", "homework", "assignment", "project"]):
        if any(k in t for k in ["homework", "assignment", "project", "lab", "problem set"]):
            boost += 2

    # 出勤
    if any(k in q for k in ["出勤", "attendance", "participation"]):
        if any(k in t for k in ["attendance", "participation"]):
            boost += 2

    # 上课时间 / 时间信息
    if any(k in q for k in ["上课时间", "几点", "几点上课", "哪天上课", "周几", "schedule"]):
        if any(k in t for k in ["lecture", "schedule", "tuesday", "thursday", "monday", "room", "class time"]):
            boost += 2

    # ===== 2) type 匹配 boost =====
    if priority_types:
        if meta_type in priority_types:
            boost += 6
        elif meta_type == "normal":
            boost += 0
        else:
            boost -= 1

    # ===== 3) 课程号匹配 boost =====
    # 问题里带了具体课程号，就优先那些文本里包含该课程号的 chunk
    for code in course_codes:
        if code.lower() in t:
            boost += 8

    # 你也可以在这里加入“同一个文件多次重复 page”之类的惩罚，这里先不搞太复杂

    return boost

def lexical_grading_candidates(question: str, max_hits: int = 5) -> List[str]:
    """
    兜底：直接在 DOCS 里扫包含百分比的 grading 段落，
    不依赖向量相似度，保证“35% midterm, 35% final...” 这种行能进上下文。
    """
    q = question.lower()
    need_grading = any(k in q for k in ["占比", "grading", "grade", "成绩", "percentage", "%"])
    if not need_grading:
        return []

    course_codes = _extract_course_codes(question)  # 你前面已经写过了
    hits = []

    for doc in DOCS:
        meta = doc.get("meta", {})
        t = doc.get("text", "")
        if meta.get("type") not in {"grading_section", "grading_line"}:
            continue
        if "%" not in t:
            continue

        low = t.lower()

        # 如果问题里提到了具体课程号，就仅保留匹配该课程的 chunk
        if course_codes:
            if not any(code.lower() in low for code in course_codes):
                # 有的 syllabus 课程号只在页头出现，不在 grading 行里，这种情况可以退一步按 file 粗匹配
                fname = meta.get("file", "").lower()
                if not any(code.lower().replace(" ", "")[:7] in fname for code in course_codes):
                    continue

        hits.append((meta.get("file", "unknown"), meta.get("page", 0), t))

    # 简单排序：按文件名 + 页码
    hits.sort(key=lambda x: (x[0], x[1]))

    results = []
    for fname, page, text in hits[:max_hits]:
        header = f"[{fname} | page {page + 1} | type=grading_fallback]"
        results.append(f"{header}\n{text}")
    return results


def retrieve_context(question: str,
                     top_k: int = TOP_K,
                     faiss_raw_k: int = FAISS_RAW_K) -> List[str]:
    """
    从向量库里检索最相关的若干 chunk：
    1）先从 Faiss 拿 faiss_raw_k 个候选；
    2）根据问题关键词 + type + 课程号做启发式加权；
    3）如果问题是在问“某一门课”的细节（比如这门课期末占比多少），
       则只保留推断出来的目标课程文件的 chunks，避免混入其他课的 %；
    4）排序后取前 top_k，并控制总长度。
    """
    q = question.lower()
    q_emb = embed_query(question)
    scores, idx = faiss_index.search(q_emb, faiss_raw_k)

    priority_types = detect_priority_types(question)
    course_codes = _extract_course_codes(question)

    # 是否把问题视为“单门课”的问题？
    # 出现“这门课 / 该课 / this course / the course”，或者显式给了课程号，就认为是单门课。
    single_course_question = (
        bool(course_codes) or
        any(k in q for k in ["这门课", "该课", "本课", "the course", "this course"])
    )

    target_file = _infer_target_file(question, idx) if single_course_question else None

    candidates = []
    for rank, i in enumerate(idx[0]):
        doc = DOCS[int(i)]
        text = doc["text"]
        meta = doc.get("meta", {})
        fname = meta.get("file", "unknown")
        page = meta.get("page", 0) + 1  # 页码从 1 开始
        meta_type = meta.get("type", "normal")

        # 如果是“问这一门课”的问题，并且已经推断出目标文件，
        # 那么只保留该文件的 chunks（避免其他课程的 grading 干扰）
        if single_course_question and target_file is not None and fname != target_file:
            continue

        header = f"[{fname} | page {page} | type={meta_type}]"
        full_text = f"{header}\n{text}"

        boost = _compute_boost(
            question=question,
            text=text,
            meta=meta,
            base_rank=rank,
            priority_types=priority_types,
            course_codes=course_codes,
        )

        candidates.append({
            "rank": rank,
            "boost": boost,
            "text": full_text,
        })

    # 按 boost（降序）+ 原 rank（升序）排序
    candidates.sort(key=lambda x: (-x["boost"], x["rank"]))

    pieces: List[str] = []
    total_len = 0
    for c in candidates:
        if len(pieces) >= top_k:
            break
        p = c["text"]
        if total_len + len(p) > MAX_CONTEXT_CHARS:
            continue
        pieces.append(p)
        total_len += len(p)

    return pieces


def call_qwen_with_rag(question: str) -> str:
    """Use RAG + Qwen to answer the question in English based on syllabi."""
    contexts = retrieve_context(question)

    context_block = "\n\n".join(contexts) if contexts else "(No relevant syllabus snippets were retrieved.)"

    system_prompt = (
        "You are a teaching assistant familiar with NYU Tandon courses. "
        "You answer students' questions strictly based on the provided syllabus snippets.\n\n"
        "Global rules:\n"
        "1) Always answer in **English**, even if the question is in another language.\n"
        "2) Keep answers **short and focused**: usually 1–3 sentences, and at most 120 words.\n"
        "3) Only talk about information types that are **explicitly requested in the question**.\n"
        "   - If the question does NOT mention grading, exam dates, workload, project, textbook, or GitHub, "
        "     do NOT bring them up.\n"
        "4) Do not compare this course with other courses unless the user explicitly asks for a comparison.\n"
        "5) Never invent numbers, percentages, policies, or rules; if something is not specified in the snippets, "
        "   clearly say that it is not specified.\n"
    )

    user_prompt = (
        "Here are raw text snippets from one or more NYU course syllabi:\n\n"
        f"{context_block}\n\n"
        "Based **only** on the information in these snippets, answer the following question in English:\n"
        f"{question}\n\n"
        "Answering style:\n"
        "1) Start with a direct answer that addresses the question.\n"
        "2) Optionally add 1–2 short supporting points **only if they are directly relevant** to the question.\n"
        "3) Do NOT include extra details such as exam dates, grading breakdowns, URLs, or project info "
        "   unless the question explicitly asks about them.\n"
        "4) If the syllabus does not specify the requested detail, say that it is not specified in the available snippets.\n"
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.8,
    }

    resp = requests.post(QWEN_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]



# ====== Flask Web 部分 ======
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>NYU Course Selection Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>

        .side-logo {
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            width: var(--side-logo-width); 
            height: auto;
            object-fit: contain;
            z-index: 1;
            opacity: 1;                  
        }

        .side-logo-left {
            left: calc((50% - 480px - var(--side-logo-width)) / 2);
        }

        .side-logo-right {
            right: calc((50% - 480px - var(--side-logo-width)) / 2);
        }

        @media (max-width: 900px) {
            .side-logo {
                display: none;
            }
        }
            :root {
                --side-logo-width: clamp(200px, 25vw, 400px);
                --bg: #57068c;
                --panel: #ffffff;
                --border: #e0e0e5;
                --accent: #57068c;
                --accent-soft: #f5e8ff;
                --text-main: #f9fafb;   /* default text on purple background */
                --text-muted: #e5d2ff;  /* muted text on purple background */
                --error: #b91c1c;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: var(--bg);
                color: var(--text-main);
            }
            .page {
                max-width: 960px;
                margin: 0 auto;
                padding: 24px 16px 32px;
            }
            header.site-header {
                border-bottom: 1px solid rgba(255,255,255,0.25);
                padding-bottom: 16px;
                margin-bottom: 24px;
            }
            .site-title-row {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .site-logo {
                width: 70px;
                height: 56px;
                border-radius: 0;
                object-fit: contain;
                background: transparent;
                padding: 0;
            }
            .site-title-text {
                display: flex;
                flex-direction: column;
            }
            .site-title {
                font-size: 18px;
                font-weight: 600;
                color: #ffffff;
            }
            .site-subtitle {
                font-size: 13px;
                color: var(--text-muted);
            }

            main {
                display: grid;
                grid-template-columns: minmax(0, 3fr);
                gap: 24px;
            }

            /* panels on white background use dark text */
            .intro-panel,
            .qa-list,
            .input-panel {
                background: var(--panel);
                border-radius: 12px;
                border: 1px solid var(--border);
                color: #111827;
            }

            .intro-panel {
                margin-bottom: 16px;
                padding: 16px 18px;
            }
            .intro-title {
                font-size: 15px;
                font-weight: 600;
                margin-bottom: 4px;
            }
            .intro-text {
                font-size: 13px;
                color: #4b5563;
                margin-bottom: 10px;
            }
            .example-questions {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 4px;
            }
            .example-pill {
                font-size: 12px;
                padding: 4px 8px;
                border-radius: 999px;
                border: 1px solid #d1d5db;
                background: #f9fafb;
                cursor: pointer;
            }
            .example-pill:hover {
                background: #f3f4f6;
            }

            .qa-section-title {
                font-size: 14px;
                font-weight: 600;
                margin: 16px 0 8px;
                color: #f9fafb;  /* title on purple background */
            }

            .qa-list {
                padding: 12px 16px;
                max-height: 60vh;
                overflow-y: auto;
            }
            .qa-empty-hint {
                font-size: 13px;
                color: #4b5563;
                text-align: center;
                margin: 24px 0 12px;
            }
            .qa-item {
                padding: 10px 0;
                border-bottom: 1px solid #e5e7eb;
            }
            .qa-item:last-child {
                border-bottom: none;
            }
            .qa-label {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 2px;
            }
            .qa-label-question {
                color: #003F88;
            }
            .qa-label-answer {
                color: #B01F24;
            }
            .qa-question {
                font-size: 14px;
                margin-bottom: 4px;
            }
            .qa-answer {
                font-size: 14px;
                background: #f9fafb;
                border-radius: 8px;
                padding: 8px 10px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }

            .qa-meta {
                font-size: 11px;
                color: #6b7280;
                margin-top: 4px;
            }

            .input-panel {
                margin-top: 24px;
                padding: 14px 16px 12px;
            }
            .input-label {
                font-size: 13px;
                font-weight: 500;
                margin-bottom: 6px;
            }
            .input-hint {
                font-size: 12px;
                color: #4b5563;
                margin-bottom: 8px;
            }
            .input-row {
                display: flex;
                gap: 10px;
                align-items: flex-start;
            }
            .input-wrap {
                flex: 1;
            }
            textarea#question {
                width: 100%;
                resize: none;
                border-radius: 8px;
                border: 1px solid var(--border);
                padding: 8px 10px;
                font-size: 14px;
                line-height: 1.4;
                font-family: inherit;
                min-height: 60px;
                max-height: 120px;
                outline: none;
                background: #ffffff;
                color: #111827;
            }
            textarea#question::placeholder {
                color: #9ca3af;
            }
            textarea#question:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 1px rgba(87,6,140,0.35);
            }
            button#sendBtn {
                border: none;
                border-radius: 999px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                background: var(--accent);
                color: #ffffff;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                flex-shrink: 0;
            }
            button#sendBtn:disabled {
                opacity: 0.5;
                cursor: default;
            }
            .send-icon {
                width: 14px;
                height: 14px;
                border-radius: 999px;
                border: 2px solid white;
                border-left-color: transparent;
                border-bottom-color: transparent;
                transform: rotate(45deg);
            }

            .status-bar {
                margin-top: 6px;
                font-size: 11px;
                color: #4b5563;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .status-error {
                color: var(--error);
            }
            .status-loading-dot::after {
                content: "…";
                animation: dots 1.2s steps(3, end) infinite;
            }
            @keyframes dots {
                0%, 20% { content: ""; }
                40% { content: "."; }
                60% { content: ".."; }
                80%, 100% { content: "..."; }
            }

            footer {
                margin-top: 18px;
                font-size: 11px;
                color: var(--text-muted);
                text-align: right;
            }

            @media (max-width: 640px) {
                main {
                    grid-template-columns: minmax(0, 1fr);
                }
                .page {
                    padding: 16px 10px 24px;
                }
            }
        </style>
    </head>
    <body>
        <img src="/static/nyu-logo.png" alt="NYU logo" class="side-logo side-logo-left">
        <div class="page">
            <header class="site-header">
                <div class="site-title-row">
                    <div class="site-title-text">
                        <div class="site-title">NYU Course Selection Assistant</div>
                        <div class="site-subtitle">
                            Ask syllabus-based questions to compare courses and plan your study path.
                        </div>
                    </div>
                </div>
            </header>

            <main>
                <section style="min-width:0;">
                    <div class="intro-panel">
                        <div class="intro-title">What does this tool do?</div>
                        <div class="intro-text">
                            This assistant answers questions strictly based on uploaded NYU syllabi.
                            It can help you understand grading, workload, topics, exam dates, and more.
                        </div>
                        <div class="intro-text">
                            Try one of the example questions:
                        </div>
                        <div class="example-questions">
                            <button class="example-pill" data-example="I want to learn machine learning. Which course should I take?">
                                I want to learn machine learning…
                            </button>
                            <button class="example-pill" data-example="What is the grading breakdown for ECE-GY 6143?">
                                Grading for ECE-GY 6143
                            </button>
                            <button class="example-pill" data-example="When is the final exam for ECE-GY 6143?">
                                Final exam date
                            </button>
                        </div>
                    </div>

                    <h2 class="qa-section-title">Question &amp; Answer History</h2>
                    <div id="qaList" class="qa-list">
                        <div class="qa-empty-hint">
                            No questions yet. Ask something about a course syllabus to get started.
                        </div>
                    </div>

                    <div class="input-panel">
                        <div class="input-label">Ask a new question</div>
                        <div class="input-hint">
                            Please be specific. For example: “What is the final exam weight for ECE-GY 6143?”
                            Press <strong>Ctrl+Enter</strong> to send.
                        </div>
                        <div class="input-row">
                            <div class="input-wrap">
                                <textarea id="question" rows="3"
                                    placeholder="Type your question here… (Enter = new line, Ctrl+Enter = send)"></textarea>
                            </div>
                            <button id="sendBtn" disabled>
                                <span>Send</span>
                                <span class="send-icon"></span>
                            </button>
                        </div>
                        <div class="status-bar">
                            <span id="statusText"></span>
                            <span>Backend: RAG over syllabi + local Qwen model</span>
                        </div>
                    </div>

                    <footer>
                        This tool only answers based on information explicitly written in the syllabi.
                    </footer>
                </section>
            </main>
        </div>

        <script>
            const textarea = document.getElementById("question");
            const sendBtn = document.getElementById("sendBtn");
            const qaList = document.getElementById("qaList");
            const statusText = document.getElementById("statusText");
            const exampleButtons = document.querySelectorAll(".example-pill");

            let isSending = false;

            function setStatus(text, isError = false, isLoading = false) {
                statusText.textContent = text || "";
                statusText.className = "";
                if (isError) statusText.classList.add("status-error");
                if (isLoading) statusText.classList.add("status-loading-dot");
            }

            function clearEmptyHint() {
                const hint = document.querySelector(".qa-empty-hint");
                if (hint) hint.remove();
            }

            function appendQA(question, answer) {
                clearEmptyHint();

                const item = document.createElement("div");
                item.className = "qa-item";

                const qLabel = document.createElement("div");
                qLabel.className = "qa-label qa-label-question";
                qLabel.textContent = "Question";

                const qText = document.createElement("div");
                qText.className = "qa-question";
                qText.textContent = question;

                const aLabel = document.createElement("div");
                aLabel.className = "qa-label qa-label-answer";
                aLabel.style.marginTop = "6px";
                aLabel.textContent = "Answer";

                const aText = document.createElement("div");
                aText.className = "qa-answer";
                aText.textContent = answer;

                item.appendChild(qLabel);
                item.appendChild(qText);
                item.appendChild(aLabel);
                item.appendChild(aText);

                qaList.appendChild(item);
                qaList.scrollTop = qaList.scrollHeight;
            }

            function updateSendButtonState() {
                const hasText = textarea.value.trim().length > 0;
                sendBtn.disabled = !hasText || isSending;
            }

            async function sendQuestion() {
                const question = textarea.value.trim();
                if (!question || isSending) return;

                isSending = true;
                updateSendButtonState();
                setStatus("Sending question to the backend", false, true);

                const currentQuestion = question;
                textarea.value = "";
                updateSendButtonState();

                try {
                    const resp = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: currentQuestion })
                    });

                    if (!resp.ok) {
                        const txt = await resp.text();
                        appendQA(currentQuestion, "Backend error: " + txt);
                        setStatus("Backend error", true, false);
                    } else {
                        const data = await resp.json();
                        if (data.error) {
                            appendQA(currentQuestion, "Error: " + data.error);
                            setStatus("Request failed", true, false);
                        } else {
                            appendQA(currentQuestion, data.answer || "(empty answer)");
                            setStatus("", false, false);
                        }
                    }
                } catch (err) {
                    console.error(err);
                    appendQA(currentQuestion, "Request failed. Is the backend running?");
                    setStatus("Request failed", true, false);
                } finally {
                    isSending = false;
                    updateSendButtonState();
                }
            }

            sendBtn.addEventListener("click", (e) => {
                e.preventDefault();
                sendQuestion();
            });

            textarea.addEventListener("input", () => {
                updateSendButtonState();
                textarea.style.height = "auto";
                const maxHeight = 120;
                textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + "px";
            });

            textarea.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        sendQuestion();
                    }
                }
            });

            exampleButtons.forEach(btn => {
                btn.addEventListener("click", () => {
                    const example = btn.getAttribute("data-example") || "";
                    textarea.value = example;
                    textarea.focus();
                    textarea.dispatchEvent(new Event("input"));
                });
            });

            setStatus("Tip: Ctrl+Enter to send");
            updateSendButtonState();
        </script>
    </body>
    </html>
    """



@app.route("/ask", methods=["POST"])
def ask():
    if request.is_json:
        question = request.json.get("question", "")
    else:
        question = request.form.get("question", "")

    if not question or not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    try:
        answer = call_qwen_with_rag(question)
    except Exception as e:
        return jsonify({"error": f"Failed to call Qwen backend: {e}"}), 500

    if request.is_json:
        return jsonify({"answer": answer})
    else:
        # Fallback HTML for non-AJAX usage (optional)
        return f"""
        <html>
        <head><meta charset="utf-8"><title>Answer</title></head>
        <body>
          <p><b>Question:</b> {question}</p>
          <hr>
          <pre>{answer}</pre>
          <a href="/">Back to main page</a>
        </body>
        </html>
        """


@app.route("/debug_retrieval", methods=["POST"])
def debug_retrieval():
    """Return the syllabus snippets selected for a given question (for debugging)."""
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    contexts = retrieve_context(question)
    return jsonify({
        "question": question,
        "num_contexts": len(contexts),
        "contexts": contexts,
    })



if __name__ == "__main__":
    print("RAG + Qwen server is running.")
    app.run(host="0.0.0.0", port=5000, debug=True)