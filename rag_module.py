# rag_module.py
"""
负责 syllabus RAG 的模块：

- 加载向量库 index.faiss + texts.json
- 提供问题改写、课程路由、chunk 选择等逻辑
- 对外暴露：
    
    - retrieve_context(question: str, ...) -> List[str]
    - is_course_selection_question(question: str) -> bool
    - is_syllabus_question(question: str) -> bool
    - is_course_comparison_question(question: str) -> bool
    - LAST_RETRIEVAL_DEBUG: dict  // 给 /debug_retrieval 看内部细节
"""

import os
import re
import json
from typing import List, Dict, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.paths import INDEX_PATH, TEXTS_PATH, E5_MODEL_NAME
from qwen_client import call_qwen

# ================== 全局配置 ==================

TOP_K = 10                 # 最终送给大模型的 chunk 数量
FAISS_RAW_K = 24           # 先从 Faiss 拿更多候选，再做二次排序
MAX_CONTEXT_CHARS = 3600   # 拼接给大模型的 syllabus 总长度上限

# 每门课的关键词（中文/英文都可以）
# 注意：后面想维护新的课程，只用在这里加就行
COURSE_FILE_HINTS: Dict[str, List[str]] = {
    # --- 1. Applied Matrix Theory (应用矩阵论) ---
    "ECE-GY 5253": [
        "5253", 
        "matrix", "matrix theory", "linear algebra", "amt", # 英文常用词/缩写
        "矩阵", "矩阵论", "线性代数" # 中文关键词
    ],

    # --- 2. Introduction to Electric Power Systems (电力系统) ---
    # 注意：JSON里原ID是 el5613，这里为了统一格式，建议Mapping Key保持规范，或者兼容旧代码
    "EL5613": [
        "5613", 
        "power", "power systems", 
        "电力", "电力系统", "强电"
    ],

    # --- 3. Digital Signal Processing I (DSP) ---
    # JSON里是 ECE-GY 6113 / BE-GY 6403 (生物医学工程跨列)
    "ECE-GY 6113": [
        "6113", "6403", # 两个课号都放进去
        "dsp", "digital signal processing", "fft",
        "数字信号处理", "信号处理", "信号"
    ],

    # --- 4. Introduction to Machine Learning (机器学习) ---
    # 补充了 CS-GY 6923 这个跨列课号
    "ECE-GY 6143": [
        "6143", "6923", 
        "machine learning", "ml", 
        "机器学习"
    ],

    # --- 5. Linear Systems (线性系统) ---
    "EL6253": [
        "6253", 
        "linear systems", "control", "control theory",
        "线性系统", "线系", "控制理论", "控制"
    ],

    # --- 6. Real Time Embedded Systems (嵌入式) ---
    "ECE-GY 6483": [
        "6483", 
        "embedded", "rtos", "real time", 
        "嵌入式", "实时系统"
    ],

    # --- 7. Embedded Lab (这门课不在JSON里，保留原样) ---
    "ECE-GY 6484": [
        "6484", 
        "embedded lab", "lab", 
        "实验", "嵌入式实验"
    ],
    # --- 8. Computing Systems Architecture (计算机体系结构) ---
    "ECE 6913": [
        "6913", 
        "architecture", "comp arch", "risc-v", "csa",
        "计算机架构", "体系结构", "架构"
    ]
}

# 调试信息给 /debug_retrieval 用
LAST_RETRIEVAL_DEBUG: Dict = {}

# ================== 模型 & 向量库加载 ==================

# e5 embedding 模型（和 memory_module 用的是同一个名字）
_emb_model = SentenceTransformer(E5_MODEL_NAME)

# syllabus Faiss index
if not os.path.exists(INDEX_PATH):
    raise RuntimeError(f"Syllabus Faiss index not found at {INDEX_PATH}")

faiss_index = faiss.read_index(INDEX_PATH)

# syllabus chunk 文本
if not os.path.exists(TEXTS_PATH):
    raise RuntimeError(f"Syllabus texts.json not found at {TEXTS_PATH}")

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    DOCS: List[Dict] = json.load(f)

print(f"[RAG] Loaded {len(DOCS)} chunks from {TEXTS_PATH}")
# ================== Qwen：意图检测 ==================
import json

import json
import re
from typing import Dict, Tuple

def analyze_request_with_qwen(question: str) -> Tuple[str, str]:
    """
    [二合一原子操作] 意图分类 + 搜索词改写
    增强版：已添加 SCHEDULE 意图支持
    """
    # [MODIFIED] 在提示词中增加了 SCHEDULE 类别和定义
    system_prompt = (
        "You are an intelligent router and query rewriter for a university RAG system.\n"
        "Your goal is to analyze the student's input and output a JSON object containing two fields:\n"
        "1. 'intent': One of [COMPARISON, SELECTION, SYLLABUS, SCHEDULE, CHAT].\n"
        "2. 'query': A concise English search query (5-20 words).\n\n"
        "Rules:\n"
        "- Output ONLY raw JSON.\n"
        "- Intent Definitions:\n"
        "  * COMPARISON: Comparing 2+ courses (e.g., 'diff between A and B').\n"
        "  * SELECTION: Asking for recommendations (e.g., 'easy courses for AI').\n"
        "  * SYLLABUS: Asking for course details (grading, exams, prerequisites, topics).\n"
        "  * SCHEDULE: Requests to manage the calendar (e.g., 'add 6143', 'remove course', 'show my schedule', 'check conflicts').\n"
        "  * CHAT: General greeting or non-academic conversation.\n"
        "- If intent is CHAT, 'query' can be null.\n"
    )

    user_prompt = f"User Input: \"{question}\"\n\nResponse JSON:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = call_qwen(messages, max_tokens=4096, temperature=0.1)
        
        # 提取 JSON
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if match:
            json_str = match.group(0)
        else:
            print(f"[Analyze] JSON parsing failed. Raw: {raw_response}")
            return "CHAT", question

        data = json.loads(json_str)
        
        # =====================================================
        # [核心修复] 健壮的数据类型处理
        # =====================================================
        
        # 1. 处理 Intent
        raw_intent = data.get("intent")
        if isinstance(raw_intent, list):
            intent = str(raw_intent[0]) if raw_intent else "CHAT"
        else:
            intent = str(raw_intent) if raw_intent else "CHAT"
        
        intent = intent.strip().upper()

        # 2. 处理 Query
        raw_query = data.get("query")
        if isinstance(raw_query, list):
            refined_query = " ".join([str(x) for x in raw_query])
        else:
            refined_query = str(raw_query) if raw_query else ""
            
        refined_query = refined_query.strip()

        # =====================================================

        # 3. 兜底校验 [MODIFIED]
        # 这里必须把 SCHEDULE 加进去，否则会被当成异常重置为 CHAT
        valid_intents = ["COMPARISON", "SELECTION", "SYLLABUS", "SCHEDULE", "CHAT"]
        
        if intent not in valid_intents:
            q_upper = question.upper()
            # 简单的关键词补救逻辑
            if "VS" in q_upper or "比较" in question: 
                intent = "COMPARISON"
            elif "RECOMMEND" in q_upper or "推荐" in question: 
                intent = "SELECTION"
            # [NEW] 增加排课的关键词兜底
            elif any(k in q_upper for k in ["ADD", "REMOVE", "SCHEDULE", "CALENDAR", "TIME TABLE"]):
                intent = "SCHEDULE"
            elif "SYLLABUS" in q_upper or "EXAM" in q_upper or "GRADE" in q_upper: 
                intent = "SYLLABUS"
            else: 
                intent = "CHAT"

        # 如果 query 为空，用原问题兜底
        if not refined_query:
            refined_query = question

        print(f"[Analyze] Intent: {intent} | Query: {refined_query}")
        return intent, refined_query

    except Exception as e:
        print(f"[Analyze ❌ ERROR] {e}")
        # 出错降级
        return "CHAT", question


def _fallback_intent_check(model_output: str, original_question: str) -> str:
    """
    辅助函数：如果模型返回的 intent 不在标准集合里，
    尝试从模型输出或原始问题中找关键词进行补救。
    """
    text_to_check = (model_output + " " + original_question).upper()
    
    if "VS" in text_to_check or "COMPAR" in text_to_check or "对比" in text_to_check:
        return "COMPARISON"
    if "SELECT" in text_to_check or "CHOOSE" in text_to_check or "RECOMMEND" in text_to_check or "哪" in text_to_check:
        return "SELECTION"
    if "SYLLABUS" in text_to_check or "EXAM" in text_to_check or "GRAD" in text_to_check or re.search(r"\d{4}", text_to_check):
        return "SYLLABUS"
    
    return "CHAT"


# ================== embedding & 课程号工具 ==================

def embed_query(query: str) -> np.ndarray:
    """对用户问题做 embedding（e5 标准：query 前缀）"""
    text = f"query: {query}"
    emb = _emb_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return emb.astype("float32").reshape(1, -1)


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


def _course_from_explicit_code(question: str) -> Tuple[Optional[str], str]:
    codes = _extract_course_codes(question)
    if len(codes) == 1:
        return _normalize_course_code(codes[0]), "explicit"
    return None, "explicit"


def _course_from_keywords(question: str) -> Tuple[Optional[str], str]:
    """
    用 COURSE_FILE_HINTS 里的关键词，根据问题粗略猜一门课。
    """
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

    # 阈值你可以后面再调，这里先 >=1 就认
    if best_score >= 1:
        return best_course, "keywords"
    return None, "keywords"


def _vote_course_by_embedding(
    scores: np.ndarray,
    idx: np.ndarray,
) -> Tuple[Optional[str], str]:
    """
    用 Faiss top-k 的结果在不同课程之间投票，选一个“主课程”。
    scores: shape [1, k]，越小越近（注意这是 L2 距离）
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
    # 这个阈值你以后觉得太严/太松可以改
    if confidence < 0.3:
        return None, "embedding"

    return best_course, "embedding"


def route_course(
    question: str,
    scores: np.ndarray,
    idx: np.ndarray,
) -> Tuple[Optional[str], str]:
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


# ================== 意图识别 ==================

def detect_priority_types(question: str) -> Set[str]:
    """
    根据用户问题，推断最可能需要哪些 syllabus chunk type。
    返回一个 type 集合，用来给这些 type 的 chunk 加权。
    """
    q = question.lower()
    types: Set[str] = set()

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


def _looks_like_comparison(question: str, course_codes: List[str]) -> bool:
    """简单判断是不是在比较多门课，如果是就允许多个 syllabus 一起出现。"""
    q = question.lower()
    if " or " in q or "vs" in q or "versus" in q:
        return True
    if "对比" in question or "比较" in question or "还是" in question:
        return True
    # 问题里点名了两个及以上课程号，也视为对比
    return len(course_codes) >= 2


# ================== chunk 打分 & 检索 ==================

def _compute_boost(
    question: str,
    text: str,
    meta: Dict,
    base_rank: int,
    priority_types: Set[str],
    course_codes: List[str],
) -> int:
    """
    综合关键词、type 匹配、课程号匹配，给每个候选 chunk 一个启发式加权分。
    """
    q = question.lower()
    t = text.lower()
    meta_type = meta.get("type", "normal")

    boost = 0

    # ==== 0) 课程号数字后缀的强制加分 ====
    # 例如问题提到 5613，meta.course/meta.file 也含 5613，则直接给高权重，压过语义噪声
    def _last_num(val) -> Optional[str]:
        if not val:
            return None
        nums = re.findall(r"\d{3,4}", str(val))
        return nums[-1] if nums else None
    def _collect_nums(val: str) -> Set[str]:
        return set(re.findall(r"\d{3,5}", val)) if val else set()

    query_nums: Set[str] = set()
    for c in course_codes:
        num = _last_num(c)
        if num:
            query_nums.add(num)
    # 再从原问题里捞所有 3-5 位数字，避免正则漏掉 “ECE 5613” 这种写法
    query_nums |= _collect_nums(question)

    meta_nums: Set[str] = set()
    mc = _last_num(meta.get("course"))
    mf = _last_num(meta.get("file"))
    if mc:
        meta_nums.add(mc)
    if mf:
        meta_nums.add(mf)
    meta_nums |= _collect_nums(meta.get("course", ""))
    meta_nums |= _collect_nums(meta.get("file", ""))

    if query_nums and meta_nums and query_nums.intersection(meta_nums):
        boost += 120  # 硬加分，确保命中课号的 chunk 排最前

    # 若完整课程码（去空格/下划线/连字符）直接对齐，也给次级加分
    def _canon(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", s.upper()) if s else ""
    q_canon = _canon(question)
    course_canon = _canon(meta.get("course", ""))
    file_canon = _canon(meta.get("file", ""))
    for cc in course_codes:
        if _canon(cc) and (_canon(cc) in course_canon or _canon(cc) in file_canon or _canon(cc) in q_canon):
            boost += 20
            break

    # ===== 1) 关键词级 boost =====

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
    for code in course_codes:
        if code.lower() in t:
            boost += 8

    return boost


def lexical_grading_candidates(question: str, max_hits: int = 5) -> List[str]:
    """
    兜底：直接在 DOCS 里扫包含百分比的 grading 段落，
    不依赖向量相似度，保证“35% midterm, 35% final...” 这种行能进上下文。
    （现在主流程没用到，你以后想加兜底可以在 retrieve_context 里拼进去）
    """
    q = question.lower()
    need_grading = any(k in q for k in ["占比", "grading", "grade", "成绩", "percentage", "%"])
    if not need_grading:
        return []

    course_codes = _extract_course_codes(question)
    hits = []

    for doc in DOCS:
        meta = doc.get("meta", {})
        t = doc.get("text", "")
        if meta.get("type") not in {"grading_section", "grading_line"}:
            continue
        if "%" not in t:
            continue

        low = t.lower()

        if course_codes:
            if not any(code.lower() in low for code in course_codes):
                fname = meta.get("file", "").lower()
                if not any(code.lower().replace(" ", "")[:7] in fname for code in course_codes):
                    continue

        hits.append((meta.get("file", "unknown"), meta.get("page", 0), t))

    hits.sort(key=lambda x: (x[0], x[1]))

    results = []
    for fname, page, text in hits[:max_hits]:
        header = f"[{fname} | page {page + 1} | type=grading_fallback]"
        results.append(f"{header}\n{text}")
    return results


def retrieve_context(
    question: str,
    top_k: int = TOP_K,
    faiss_raw_k: int = FAISS_RAW_K,
    forced_course: Optional[str] = None,
    analysis_question: Optional[str] = None,
) -> List[str]:
    """
    从向量库里检索最相关的若干 chunk：

    1）先从 Faiss 拿 faiss_raw_k 个候选；
    2）根据问题关键词 + type + 课程号做启发式加权；
    3）如果问题不像是“比较多门课”，就只保留一门课（同一个 pdf）的内容；
    4）排序后取前 top_k，并控制总长度。
    """
    aq = analysis_question or question

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

    candidates.sort(key=lambda x: (-x["boost"], x["rank"]))

    

    global LAST_RETRIEVAL_DEBUG
    LAST_RETRIEVAL_DEBUG = {
        "question": aq,
        "selected_course": selected_course,
        "route_source": route_source,
        "course_codes_in_question": course_codes,
        "is_comparison": is_comparison,
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

    dominant_course: Optional[str] = None
    dominant_file: Optional[str] = None

    if not is_comparison and candidates:
        if selected_course is not None:
            dominant_course = selected_course
        else:
            file_counts: Dict[str, int] = {}
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


__all__ = [
    
    "retrieve_context",
    "is_course_selection_question",
    "is_syllabus_question",
    "is_course_comparison_question",
    "lexical_grading_candidates",
    "LAST_RETRIEVAL_DEBUG",
]
