# app.py
import os
import re
import json
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from flask import Flask, request, jsonify, render_template
from typing import Optional

LAST_RETRIEVAL_DEBUG = {}
# ====== RAG 部分配置 ======
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

E5_MODEL_NAME = "intfloat/multilingual-e5-large"

TOP_K = 10                 # 最终送给大模型的 chunk 数量
FAISS_RAW_K = 24           # 先从 Faiss 拿更多候选，再做二次排序
MAX_CONTEXT_CHARS = 3600   # 拼接给大模型的 syllabus 总长度上限

# 每门课的关键词（中文/英文都可以）
COURSE_FILE_HINTS = {
    "ECE-GY 6143": ["6143", "machine learning", "机器学习", "ml"],
    "ECE-GY 6483": ["6483", "embedded", "RISC-V", "嵌入式"],
    "ECE-GY 6484": ["6484", "embedded lab", "lab", "实验"],
    # TODO: 自己把其他课补上
}

COURSE_PROFILES = [
    {
        "code": "ECE-GY 6143",
        "name": "Introduction to Machine Learning",
        "focus": "machine learning, statistics, supervised and unsupervised learning"
    },
    {
        "code": "ECE-GY 6913",
        "name": "Computer System Architecture",
        "focus": "RISC-V instruction set, pipelined processor, cache, memory hierarchy, low-level architecture"
    },
    {
        "code": "ECE-GY 6483",
        "name": "Real Time Embedded Systems",
        "focus": "ARM Cortex-M, STM32, real-time OS, peripherals, embedded C"
    }
    # 你想让哪些课参与“选课推荐”，就加进去
]

# ====== Qwen + vLLM OpenAI 接口配置 ======
QWEN_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
QWEN_MAX_TOKENS = 2048


# ====== 加载向量索引和文本 ======
print("[RAG] Loading Faiss index and texts ...")
faiss_index = faiss.read_index(INDEX_PATH)
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    DOCS: List[Dict] = json.load(f)
print(f"[RAG] Loaded {len(DOCS)} chunks")

emb_model = SentenceTransformer(E5_MODEL_NAME)

def refine_question_with_qwen(question: str) -> str:
    """
    第一次调用 Qwen：把用户的原始提问，压缩成一条英文检索用的关键词/短句。
    不要带解释，只要一行简短 query。
    """
    system_prompt = (
        "You are a query rewriting assistant for a RAG system over NYU course syllabi.\n"
        "Your task: rewrite the student's question into a SHORT English search query or keyword list.\n"
        "Rules:\n"
        "1) Output ONLY the rewritten query, in English, on a single line.\n"
        "2) Do NOT explain, do NOT add any extra text.\n"
        "3) Preserve any course codes like 'ECE-GY 6143' exactly if they appear.\n"
        "4) Use 5–20 words that best capture the intent (topics, grading, exam, workload, etc.)."
    )

    user_prompt = (
        "Student question:\n"
        f"{question}\n\n"
        "Rewrite this as a concise English search query for retrieving relevant syllabus snippets."
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.8,
    }

    try:
        resp = requests.post(QWEN_API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        refined = (raw or "").strip()
        # 千问如果抽风或者返回空，就退回原始问题
        if not refined:
            return question
        return refined
    except Exception:
        # 千问第一次调用挂了，也退回原始问题，不影响整体功能
        return question

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

def is_course_selection_question(question: str) -> bool:
    q = question.lower()
    # 英文触发词
    if any(k in q for k in [
        "which course should i take",
        "which course should i choose",
        "which course is better",
        "which course is more suitable",
        "recommend a course",
        "what course should i take",
        "what courses should i take",
        "which courses should i take",
    ]):
        return True

    # 中文触发词（放原文，不转小写）
    if any(k in question for k in [
        "选哪门课",
        "选哪个课",
        "选哪门",
        "选什么课",
        "该选什么课",
        "推荐哪门课",
        "推荐什么课",
        "哪些课适合我",
        "我要学什么课",
        "学什么课",
        "学哪门课",
        "哪门课比较适合",
        "选一些课",
        "选几门课",
    ]):
        return True

    return False

def _looks_like_comparison(question: str, course_codes: List[str]) -> bool:
    """简单判断是不是在比较多门课，如果是就允许多个 syllabus 一起出现。"""
    q = question.lower()
    if " or " in q or "vs" in q or "versus" in q:
        return True
    if "对比" in question or "比较" in question or "还是" in question:
        return True
    # 问题里点名了两个及以上课程号，也视为对比
    return len(course_codes) >= 2


def _extract_course_codes(question: str) -> List[str]:
    """
    从问题里抽取类似 ECE-GY 6143 / CS-GY 6923 这样的课程号。
    """
    # 允许中间有空格或连字符
    pattern = re.compile(r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b")
    codes = pattern.findall(question)
    return [c.strip() for c in codes]

def _normalize_course_code(code: str) -> str:
    """
    把各种写法规范成 'ECE-GY 6143' 这种格式，方便和 meta['course'] 对齐。
    """
    c = code.upper().replace("_", " ").replace("-", " ")
    parts = c.split()
    if len(parts) == 3 and parts[0] == "ECE" and parts[1] == "GY":
        return f"{parts[0]}-{parts[1]} {parts[2]}"
    return code.strip()


def _course_from_explicit_code(question: str) -> tuple[Optional[str], str]:
    codes = _extract_course_codes(question)
    if len(codes) == 1:
        return _normalize_course_code(codes[0]), "explicit"
    return None, "explicit"


def _course_from_keywords(question: str) -> tuple[Optional[str], str]:
    text = question.lower()
    best_course = None
    best_score = 0

    for course, hints in COURSE_FILE_HINTS.items():
        score = 0
        for h in hints:
            if h.lower() in text:
                score += 1
        if score > best_score:
            best_score = score
            best_course = course

    # 阈值你可以调，大于等于 2 再认定
    if best_score >= 1:
        return best_course, "keywords"
    return None, "keywords"


def _vote_course_by_embedding(scores: np.ndarray,
                              idx: np.ndarray) -> tuple[Optional[str], str]:
    """
    用 Faiss top-k 的结果在不同课程之间投票，选一个“主课程”。
    scores: shape [1, k]，越小越近（L2 distance）
    idx:    shape [1, k]
    """
    course_scores: Dict[str, float] = {}

    for dist, doc_idx in zip(scores[0], idx[0]):
        doc = DOCS[int(doc_idx)]
        meta = doc.get("meta", {})
        course = meta.get("course")
        if not course:
            continue

        # L2 距离越小越相似，所以用负号
        score = -float(dist)
        course_scores[course] = course_scores.get(course, 0.0) + score

    if not course_scores:
        return None, "embedding"

    best_course, best_score = max(course_scores.items(), key=lambda kv: kv[1])
    total_pos = sum(v for v in course_scores.values() if v > 0)

    if total_pos <= 0:
        return None, "embedding"

    confidence = best_score / total_pos
    # 你可以自己调阈值
    if confidence < 0.3:
        return None, "embedding"

    return best_course, "embedding"


def route_course(question: str,
                 scores: np.ndarray,
                 idx: np.ndarray) -> tuple[Optional[str], str]:
    """
    综合三步：
      1) 问题里显式课程号；
      2) 关键词映射；
      3) embedding 在 top-k 里投票。
    返回 (课程号 or None, 来源标签)
    """
    # 1. 显式课程号
    course, src = _course_from_explicit_code(question)
    if course:
        return course, src

    # 2. 关键词
    course, src = _course_from_keywords(question)
    if course:
        return course, src

    # 3. embedding 投票
    course, src = _vote_course_by_embedding(scores, idx)
    if course:
        return course, src

    return None, "unknown"

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
                     faiss_raw_k: int = FAISS_RAW_K,
                     forced_course: Optional[str] = None,
                     analysis_question: Optional[str] = None) -> List[str]:
    """
    从向量库里检索最相关的若干 chunk：
    1）先从 Faiss 拿 faiss_raw_k 个候选；
    2）根据问题关键词 + type + 课程号做启发式加权；
    3）如果问题不像是“比较多门课”，就只保留一门课（同一个 pdf）的内容；
    4）排序后取前 top_k，并控制总长度。
    """
    aq = analysis_question or question  # 没传就退回 question，保证兼容旧路径

    q_emb = embed_query(question)
    scores, idx = faiss_index.search(q_emb, faiss_raw_k)

    selected_course, route_source = route_course(aq, scores, idx)

    if forced_course is not None:
            selected_course = forced_course
            route_source = "forced"

    priority_types = detect_priority_types(aq)
    course_codes = _extract_course_codes(aq)
    is_comparison = _looks_like_comparison(aq, course_codes)

    candidates = []
    for rank, i in enumerate(idx[0]):
        doc = DOCS[int(i)]
        text = doc["text"]
        meta = doc.get("meta", {})
        fname = meta.get("file", "unknown")
        page = meta.get("page", 0) + 1  # 页码从 1 开始

        header = f"[{fname} | page {page} | type={meta.get('type', 'normal')}]"
        full_text = f"{header}\n{text}"

        boost = _compute_boost(
            question=aq,
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
            "meta": meta,
        })

    # 先按 boost（降序）+ 原 rank（升序）排一下
    candidates.sort(key=lambda x: (-x["boost"], x["rank"]))

    global LAST_RETRIEVAL_DEBUG
    LAST_RETRIEVAL_DEBUG = {
        "question": aq,
        "selected_course": selected_course,
        "route_source": route_source,
        "course_codes_in_question": course_codes,
        "is_comparison": is_comparison,
        # 看前 10 个原始 faiss 结果的距离和 idx
        "faiss_top10": [
            {
                "rank": int(r),
                "dist": float(d),
                "doc_idx": int(doc_idx),
                "file": DOCS[int(doc_idx)].get("meta", {}).get("file"),
                "course": DOCS[int(doc_idx)].get("meta", {}).get("course"),
                "type": DOCS[int(doc_idx)].get("meta", {}).get("type"),
            }
            for r, (d, doc_idx) in enumerate(zip(scores[0][:10], idx[0][:10]))
        ],
        # 看按 boost 排序后的前 20 个候选
        "candidates_top20": [
            {
                "rank": c["rank"],
                "boost": c["boost"],
                "file": c["meta"].get("file"),
                "course": c["meta"].get("course"),
                "type": c["meta"].get("type"),
            }
            for c in candidates[:20]
        ],
    }
    # 单课程模式：优先用“课程号路由”，如果没有再退回“按文件 majority vote”
    dominant_course = None
    dominant_file = None

    if not is_comparison and candidates:
        if selected_course is not None:
            dominant_course = selected_course
        else:
            file_counts = {}
            for c in candidates[:min(len(candidates), faiss_raw_k)]:
                fname = c["meta"].get("file", "")
                file_counts[fname] = file_counts.get(fname, 0) + 1
            if file_counts:
                dominant_file = max(file_counts.items(), key=lambda kv: kv[1])[0]


    pieces: List[str] = []
    total_len = 0

    for c in candidates:
        if len(pieces) >= top_k:
            break

        meta = c["meta"]
        # 单课程模式：优先按 dominant_course 过滤；没有的话再按 dominant_file
        if dominant_course is not None:
            if meta.get("course") != dominant_course:
                continue
        elif dominant_file is not None:
            if meta.get("file") != dominant_file:
                continue

        p = c["text"]
        if total_len + len(p) > MAX_CONTEXT_CHARS:
            continue

        pieces.append(p)
        total_len += len(p)

    return pieces

def classify_course_for_selection(question: str) -> Optional[str]:
    """
    用 Qwen 在 COURSE_PROFILES 中选出最适合这次提问的一门课，返回课程代码。
    """
    courses_block_lines = []
    for c in COURSE_PROFILES:
        line = f"- {c['code']}: {c['name']} (focus: {c['focus']})"
        courses_block_lines.append(line)
    courses_block = "\n".join(courses_block_lines)

    system_prompt = (
        "You are an academic advisor at NYU Tandon.\n"
        "You will be given:\n"
        "1) A list of available courses with their code, name, and focus.\n"
        "2) A student's question about what they want to learn.\n\n"
        "Your job: choose exactly one best course from the list that matches the student's interests.\n"
        "If none of the courses clearly match, answer \"unknown\".\n"
        "Output ONLY a JSON object like:\n"
        "{\"course\": \"ECE-GY 6913\"}\n"
        "or\n"
        "{\"course\": \"unknown\"}\n"
    )

    user_prompt = (
        "Available courses:\n"
        f"{courses_block}\n\n"
        "Student question:\n"
        f"{question}\n\n"
        "Choose exactly one course code from the list above, or \"unknown\" if nothing fits."
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.8,
    }

    resp = requests.post(QWEN_API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]

    try:
        result = json.loads(raw)
        course = result.get("course")
        if course and course != "unknown":
            return course
    except Exception:
        # 模型没按 JSON 输出，就认为分类失败
        pass
    return None

def answer_course_selection_question(original_question: str,
                                     retrieval_question: Optional[str] = None) -> str:
    """
    选课类问题：先用 classify_course_for_selection 选课，再用 RAG+Qwen 解释。
    """
    course = classify_course_for_selection(original_question)
    if course is None:
        # 分类失败，退回普通 RAG（这里就相当于：第一次精炼失败，直接用原问题做检索）
        return call_qwen_with_rag(original_question, retrieval_question)

    # 第二步：强制锁定这门课做检索
    rq = retrieval_question or original_question
    contexts = retrieve_context(
        question=rq,
        forced_course=course,
        analysis_question=original_question   # 需要你给 retrieve_context 加这个参数
    )
    context_block = "\n\n".join(contexts) if contexts else "(No relevant syllabus snippets were retrieved.)"

    system_prompt = (
        "You are an academic advisor and teaching assistant at NYU Tandon.\n"
        f"You are currently recommending and explaining one specific course: {course}.\n"
        "Use only the provided syllabus snippets when describing what this course covers.\n"
        "Keep answers short (1–3 sentences)."
    )

    user_prompt = (
        f"The student asked:\n{original_question}\n\n"
        f"You (as advisor) have decided that the best matching course is: {course}.\n\n"
        "Here are syllabus snippets (may include multiple courses, but you must only rely on the parts clearly related to this course):\n"
        f"{context_block}\n\n"
        "Based on these snippets, briefly explain why this course matches the student's request "
        "and what low-level / architecture / embedded topics it covers. "
        "If the syllabus does not show relevant low-level content, say so honestly."
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.8,
    }

    resp = requests.post(QWEN_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def call_qwen_with_rag(original_question: str,
                       retrieval_question: Optional[str] = None) -> str:
    """
    第二次调用 Qwen：先用 retrieval_question 做 RAG 检索，再用 syllabus 片段回答 original_question。
    retrieval_question 一般是第一次千问输出的英文关键词；为空时退回 original_question。
    """
    rq = retrieval_question or original_question

    # 检索用的是精炼后的 rq
    contexts = retrieve_context(
        question=rq,                 # 用 refined 做 embedding
        analysis_question=original_question  # 用原始问题做路由/关键词
    )
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
        "If the question is asking about a single course (not comparing multiple courses), then even if the retrieved "
        "context contains snippets from several syllabi, you must answer strictly based on the syllabus of that single "
        "course only. Do not mention instructors, grading policies, exam information, or any details from other courses."
    )

    user_prompt = (
        "Here are raw text snippets from one or more NYU course syllabi:\n\n"
        f"{context_block}\n\n"
        "The student originally asked:\n"
        f"{original_question}\n\n"
        "Internally, the system used the following refined search query to retrieve context:\n"
        f"{rq}\n\n"
        "Based **only** on the information in the syllabus snippets, answer the student's original question in English.\n\n"
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
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if request.is_json:
        question = request.json.get("question", "")
    else:
        question = request.form.get("question", "")

    if not question or not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    try:
            refined = refine_question_with_qwen(question)

            if is_course_selection_question(question):
                # 专门的选课/比较 path
                answer = answer_course_selection_question(
                    original_question=question,
                    retrieval_question=refined
                )
            else:
                # 普通 syllabus 细节问题
                answer = call_qwen_with_rag(
                    original_question=question,
                    retrieval_question=refined
                )
    except Exception as e:
        return jsonify({"error": f"Failed to call Qwen backend: {e}"}), 500

    if request.is_json:
        return jsonify({"answer": answer})
    else:
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
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    refined = refine_question_with_qwen(question)

    contexts = retrieve_context(
        question=refined,
        analysis_question=question
    )
    return jsonify({
        "question": question,
        "refined": refined,
        "num_contexts": len(contexts),
        "contexts": contexts,
        "debug": LAST_RETRIEVAL_DEBUG,
    })

if __name__ == "__main__":
    print("RAG + Qwen server is running.")
    app.run(host="0.0.0.0", port=5000, debug=True)