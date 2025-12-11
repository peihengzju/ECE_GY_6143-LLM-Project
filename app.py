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
from itertools import count
from abc import ABC, abstractmethod

TURN_COUNTER = count(1)  # 1,2,3,... 每次 next() 加 1
LAST_RETRIEVAL_DEBUG = {}
# ====== Memory RAG 配置 ======
MEMORY_DIR = "memory_store"
MEMORY_INDEX_PATH = os.path.join(MEMORY_DIR, "mem_index.faiss")
MEMORY_TEXTS_PATH = os.path.join(MEMORY_DIR, "memories.json")

os.makedirs(MEMORY_DIR, exist_ok=True)

if not os.path.exists(MEMORY_TEXTS_PATH):
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# 记忆槽类型
MEMORY_SLOTS = {
    "profile",        # 个人信息：姓名、学校、专业、背景
    "preference",     # 长期偏好：想学什么、风格偏好
    "fact",           # 重要事实：比如“我是 NYU ECE”、“我想走 AI infra”
    "recent"          # 临时上下文：最近 5～10 轮
}

# 不同槽采用不同的过期 / 权重策略
MEMORY_EXPIRY_DAYS = {
    "profile":   None,   # 永不过期
    "preference": 365,   # 软过期，一年
    "fact":      180,    # 半年
    "recent":     7      # 一周
}

# ====== RAG 部分配置 ======
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

E5_MODEL_NAME = "intfloat/multilingual-e5-large"
emb_model = SentenceTransformer(E5_MODEL_NAME)
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

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal, Any

MemorySlot = Literal["profile", "preference", "fact", "recent"]

@dataclass
class MemoryItem:
    id: int
    slot: MemorySlot
    text: str                  # 可能是摘要后的文本
    importance: int            # 0 = 一般，1 = 中等，2 = 很重要
    created_at: str            # ISO 字符串
    last_used_at: str          # 上次被检索到的时间
    extra: Dict[str, Any]      # 额外 meta, 比如 { "source_turn": 12 }

class MemoryAggregator(ABC):
    """
    通用聚合记忆基类：
    - kind: 用于 extra["kind"] 标记聚合类型（如 'career_direction', 'profile_aggregate'）
    - slot: 这个聚合记忆属于哪个槽（profile / preference / fact / recent）
    - base_importance: 聚合记忆的基础重要度
    """
    kind: str
    slot: MemorySlot
    base_importance: int

    @abstractmethod
    def extract_entities(self, text: str) -> list[str]:
        """
        从一轮对话（question+answer）里抽取该聚合类型的“实体列表”。
        没有就返回 []。
        """
        ...

    def render_text(self, entities: list[str]) -> str:
        """
        把实体列表渲染成一条人类可读的记忆文本。
        子类可以重写。
        """
        return f"{self.kind}: " + ", ".join(entities)

    def upsert(self,
               items: List[MemoryItem],
               question: str,
               answer: str,
               source_turn: int) -> List[MemoryItem]:
        """
        在 MemoryItem 列表里“更新/插入”一条聚合记忆：
        - 找到 extra['kind'] == self.kind 的那条
        - 合并实体列表
        - 更新 text / last_used_at / importance
        - 如果不存在就新建一条
        """
        raw = (question or "") + "\n" + (answer or "")
        entities = self.extract_entities(raw)
        if not entities:
            return items

        now = datetime.utcnow().isoformat()
        existing: Optional[MemoryItem] = None
        for m in items:
            if m.extra.get("kind") == self.kind:
                existing = m
                break

        if existing is not None:
            old_entities = set(existing.extra.get("entities", []))
            merged = sorted(old_entities | set(entities))
            existing.extra["entities"] = merged
            existing.text = self.render_text(merged)
            existing.last_used_at = now
            # importance 用聚合自身的 base_importance 覆盖一下原值（可选）
            existing.importance = self.base_importance
        else:
            new_id = (max((m.id for m in items), default=0) + 1) if items else 1
            merged = sorted(set(entities))
            mem = MemoryItem(
                id=new_id,
                slot=self.slot,
                text=self.render_text(merged),
                importance=self.base_importance,
                created_at=now,
                last_used_at=now,
                extra={
                    "kind": self.kind,
                    "entities": merged,
                    "source_turn": source_turn,
                },
            )
            items.append(mem)

        return items

from pathlib import Path

if not Path(MEMORY_TEXTS_PATH).exists():
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def _load_memories() -> List[MemoryItem]:
    # 文件不存在 → 没有任何记忆
    if not os.path.exists(MEMORY_TEXTS_PATH):
        return []

    try:
        with open(MEMORY_TEXTS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # 文件存在但被清空了（内容是空字符串）→ 当作空列表
        if not content:
            return []

        arr = json.loads(content)
    except Exception:
        # 文件内容损坏 / 不是合法 JSON → 重置为 []
        arr = []
        with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    return [MemoryItem(**m) for m in arr]


def _save_memories(items: List[MemoryItem]) -> None:
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in items], f, ensure_ascii=False, indent=2)

def _rebuild_mem_index(items: Optional[List[MemoryItem]] = None) -> None:
    """
    根据当前所有 MemoryItem 重新构建 mem_index。
    聚合记忆会修改已有条目，所以不能只做 incremental add。
    """
    global mem_index

    if items is None:
        items = _load_memories()

    # 还没有任何记忆，建一个空 index 即可
    if not items:
        dummy = emb_model.encode(["query: dummy"], convert_to_numpy=True, normalize_embeddings=True)[0]
        d = len(dummy)
        mem_index = faiss.IndexFlatIP(d)
        _save_mem_index()
        return

    # 正常重建
    dummy = emb_model.encode(["query: dummy"], convert_to_numpy=True, normalize_embeddings=True)[0]
    d = len(dummy)
    mem_index = faiss.IndexFlatIP(d)

    texts = [f"passage: {m.text}" for m in items]
    embs = emb_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    mem_index.add(embs)
    _save_mem_index()

def reset_memories() -> None:
    """
    清空所有记忆：
    - 把 memories.json 重置为 []
    - 把 mem_index 清空
    """
    global mem_index

    # 1) 重置 JSON 文件为合法的空数组
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    # 2) 重置 Faiss index（保持维度不变）
    d = mem_index.d  # 当前向量维度
    mem_index = faiss.IndexFlatIP(d)
    _save_mem_index()

# 记忆向量索引，全局变量
if os.path.exists(MEMORY_INDEX_PATH):
    mem_index = faiss.read_index(MEMORY_INDEX_PATH)
else:
    # 维度和 emb_model 一致
    dummy = emb_model.encode(["query: dummy"], convert_to_numpy=True, normalize_embeddings=True)[0]
    d = len(dummy)
    mem_index = faiss.IndexFlatIP(d)  # 内积，embedding 已经 normalize，相当于余弦相似度
    faiss.write_index(mem_index, MEMORY_INDEX_PATH)

def _save_mem_index():
    faiss.write_index(mem_index, MEMORY_INDEX_PATH)

def _build_dialogue_snippet(question: str, answer: str) -> str:
    return f"User: {question}\nAssistant: {answer}"

def summarize_dialogue_with_qwen(snippet: str, max_words: int = 80) -> str:
    """
    把一段较长的对话压缩成一条简短摘要，用于记忆存储。
    """
    system_prompt = (
        "You are a memory compression assistant.\n"
        "Task: Summarize the given user–assistant dialogue into a short factual memory.\n"
        "Rules:\n"
        f"1) Use at most {max_words} English words.\n"
        "2) Capture stable facts (user profile, preferences, goals, important decisions).\n"
        "3) Do not include ephemeral details or step-by-step reasoning.\n"
        "4) Output only the summary, no explanation."
    )

    user_prompt = (
        "Dialogue:\n"
        f"{snippet}\n\n"
        "Summarize this into a single short memory sentence."
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.8,
    }

    try:
        resp = requests.post(QWEN_API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        summary = (data["choices"][0]["message"]["content"] or "").strip()
        return summary if summary else snippet[:200]
    except Exception:
        # 摘要失败就退回截断原文
        return snippet[:200]

def classify_memory_slot(question: str, answer: str) -> tuple[MemorySlot, int]:
    """
    粗暴版分类：
    - 提到学校/专业/背景 → profile
    - 提到“想做什么 / 想学什么 / 兴趣方向” → preference
    - 明确的事实陈述（“我已经…”，“我打算…”）→ fact
    - 其他默认 recent
    importance: 2 = 很关键, 1 = 一般重要, 0 = 临时
    """
    q = question.lower()
    a = answer.lower()
    text = q + " " + a

    # profile
    if any(k in text for k in ["nyu", "tandon", "ece", "major", "degree", "master", "phd"]):
        return "profile", 2

    # preference
    if any(k in text for k in ["i want to learn", "i want to do", "i prefer", "我想学", "我想做", "我更想"]):
        return "preference", 2

    # important fact
    if any(k in text for k in ["i decided", "i will", "i plan to", "我决定", "打算", "已经辞职", "已经报名"]):
        return "fact", 2

    # 默认 recent
    return "recent", 0

class CareerDirectionAggregator(MemoryAggregator):
    kind = "career_direction"
    slot: MemorySlot = "preference"
    base_importance = 3

    def extract_entities(self, text: str) -> list[str]:
        t = text.lower()
        dirs = set()

        # AI infra / 系统底层
        if any(k in t for k in ["ai infra", "ai infrastructure", "系统底层", "system-level", "low-level"]):
            dirs.add("AI infrastructure / low-level systems")

        # 嵌入式
        if any(k in t for k in ["embedded", "嵌入式"]):
            dirs.add("embedded systems")

        # 分布式 / 存储 / GPU 调度
        if any(k in t for k in ["distributed system", "distributed systems", "分布式"]):
            dirs.add("distributed systems")
        if any(k in t for k in ["storage", "存储"]):
            dirs.add("storage systems")
        if any(k in t for k in ["gpu 调度", "gpu scheduling", "gpu scheduler"]):
            dirs.add("GPU scheduling")

        # 芯片 / VLSI / ASIC / IC design
        if any(k in t for k in ["芯片", "chip", "ic design", "集成电路"]):
            dirs.add("chip / IC design")
        if "vlsi" in t:
            dirs.add("VLSI design")
        if "asic" in t:
            dirs.add("ASIC design")

        return sorted(dirs)

    def render_text(self, entities: list[str]) -> str:
        return "User's long-term career directions: " + ", ".join(entities)

class ProfileAggregator(MemoryAggregator):
    kind = "profile_aggregate"
    slot: MemorySlot = "profile"
    base_importance = 3

    def extract_entities(self, text: str) -> list[str]:
        t = text.lower()
        ents = set()

        if "nyu" in t and "tandon" in t:
            ents.add("NYU Tandon")
        if "ece" in t or "electrical and computer engineering" in t:
            ents.add("ECE master's student")
        if "brooklyn" in t:
            ents.add("based in Brooklyn")
        if "master" in t or "硕士" in t:
            ents.add("graduate student")

        return sorted(ents)

    def render_text(self, entities: list[str]) -> str:
        return "User's profile: " + ", ".join(entities)
    
AGGREGATORS: List[MemoryAggregator] = [
    CareerDirectionAggregator(),
    ProfileAggregator(),
    # 以后你要加 SkillsAggregator、PreferenceAggregator，直接加到这里
]

def add_memory_from_turn(question: str, answer: str, source_turn: int) -> None:
    snippet = _build_dialogue_snippet(question, answer)
    # 超过 700 字做摘要
    if len(snippet) > 700:
        text = summarize_dialogue_with_qwen(snippet, max_words=80)
    else:
        text = snippet

    # ===== 1) 先写一条“原始记忆条目” =====
    slot, importance = classify_memory_slot(question, answer)

    items = _load_memories()
    new_id = (max((m.id for m in items), default=0) + 1) if items else 1
    now = datetime.utcnow().isoformat()

    raw_mem = MemoryItem(
        id=new_id,
        slot=slot,
        text=text,
        importance=importance,
        created_at=now,
        last_used_at=now,
        extra={"source_turn": source_turn}
    )
    items.append(raw_mem)

    # ===== 2) 跑所有 Aggregator，生成/更新聚合记忆条目 =====
    for agg in AGGREGATORS:
        items = agg.upsert(items, question, answer, source_turn)

    # ===== 3) 统一落盘 + 重建向量索引 =====
    _save_memories(items)
    _rebuild_mem_index(items)


def _time_decay(slot: MemorySlot, created_at: str) -> float:
    """
    根据创建时间和槽类型，算一个 [0, 1] 的时间权重。
    """
    expiry_days = MEMORY_EXPIRY_DAYS.get(slot)
    if expiry_days is None:
        return 1.0  # 永不过期，时间权重恒 1

    created = datetime.fromisoformat(created_at)
    now = datetime.utcnow()
    delta_days = (now - created).days

    if delta_days <= 0:
        return 1.0
    if delta_days >= expiry_days:
        return 0.0

    # 简单线性衰减 (你也可以改成 exp 衰减)
    return max(0.0, 1.0 - delta_days / expiry_days)


def retrieve_memories(question: str,
                      top_k: int = 5,
                      alpha: float = 0.5,
                      beta: float = 1.0) -> List[MemoryItem]:
    """
    检索记忆：
    overall_score = embedding_sim + alpha * time_decay + beta * importance
    embedding_sim = 内积结果（[-1,1]）
    """
    items = _load_memories()
    if not items or mem_index.ntotal == 0:
        return []

    # query embedding
    q_text = f"query: {question}"
    q_emb = emb_model.encode(
        [q_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # 向量检索
    k = min(top_k * 4, mem_index.ntotal)  # 先拿多一点候选
    sims, idxs = mem_index.search(q_emb, k)  # 内积 → sims 越大越相似
    sims = sims[0]
    idxs = idxs[0]

    scored: List[tuple[float, MemoryItem]] = []

    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(items):
            continue
        m = items[idx]
        t_decay = _time_decay(m.slot, m.created_at)
        imp = m.importance

        overall = float(sim) + alpha * t_decay + beta * imp

        kind = m.extra.get("kind")
        if kind == "career_direction":
            overall += 1.0
        elif kind == "profile_aggregate":
            overall += 0.8
        # 以后你有 skills_aggregate 等，可以在这里扩展

        scored.append((overall, m))

    # 排序取前 top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [m for _, m in scored[:top_k]]

    # 更新 last_used_at
    now = datetime.utcnow().isoformat()
    id_set = {m.id for m in top_items}
    for m in items:
        if m.id in id_set:
            m.last_used_at = now
    _save_memories(items)

    return top_items

def format_memories_block(mem_items: List[MemoryItem]) -> str:
    if not mem_items:
        return "(No retrieved memories.)"
    lines = []
    for m in mem_items:
        # 简单一点：用 [slot#importance] 前缀
        lines.append(f"[{m.slot} | importance={m.importance}] {m.text}")
    return "\n".join(lines)

# ====== 加载向量索引和文本 ======
print("[RAG] Loading Faiss index and texts ...")
faiss_index = faiss.read_index(INDEX_PATH)
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    DOCS: List[Dict] = json.load(f)
print(f"[RAG] Loaded {len(DOCS)} chunks")

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

def is_course_comparison_question(question: str) -> bool:
    q = question.lower()
    return (
        "还是" in question or
        "比较" in question or
        "对比" in question or
        " or " in q or
        "vs" in q or
        "versus" in q
    )

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

def is_syllabus_question(question: str) -> bool:
    """
    判断是不是“问课程 / syllabus / grading / exam / 作业”这类问题。
    只有这类问题才触发 syllabus RAG。
    """
    q = question.lower()

    # 1) 有明确课程号
    if _extract_course_codes(question):
        return True

    # 2) 出现 NYU Tandon 课程常见关键词
    if any(k in q for k in [
        "ece-gy", "cs-gy", "syllabus",
        "grading", "grade", "exam", "midterm", "final", "quiz",
        "homework", "assignment", "project", "lab",
        "office hour", "instructor", "professor",
    ]):
        return True

    # 3) 中文关键词：课程 / 作业 / 考试 / 占比 等
    if any(k in question for k in [
        "这门课", "课程内容", "上课时间", "作业", "考试",
        "期末", "期中", "占比", "评分", "成绩", "先修课", "难度",
    ]):
        return True

    return False

def answer_course_comparison_question(original_question, retrieval_question):
    # 1) 抽取两个课程代码
    codes = _extract_course_codes(original_question)
    # 如果没抓到两个，你可以 fallback 到 general mode

    # 2) 对这两个课程分别 retrieve_context
    contexts_a = retrieve_context(retrieval_question, forced_course=codes[0], analysis_question=original_question)
    contexts_b = retrieve_context(retrieval_question, forced_course=codes[1], analysis_question=original_question)

    # 3) memory 用来做 personalization
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    # 4) prompt：先比较，再结合用户方向给微调建议


def chat_with_memory_only(original_question: str) -> str:
    """
    纯对话模式：不查 syllabus，只用记忆 + 通用知识聊天。
    用于：自我介绍、职业规划、生活问题、随便聊天等。
    """
    mem_items = retrieve_memories(original_question, top_k=8)
    memory_block = format_memories_block(mem_items)

    system_prompt = (
        "You are a helpful assistant in a multi-turn chat.\n"
        "You see the user's latest message and some internal memory snippets about them.\n"
        "Your job is to reply like in a normal conversation, not to write a third-person profile.\n\n"
        "Rules:\n"
        "1) Always address the user in second person (\"you\"), never as \"the student\" or \"the user\".\n"
        "2) Write a natural chat-style answer in 1–3 sentences.\n"
        "3) If the user is sharing background or future plans, briefly acknowledge it and, if helpful, "
        "   restate their background/goals in second person.\n"
        "4) You may use the memory snippets to stay consistent, but DO NOT copy them verbatim; rephrase in your own words.\n"
        "5) In this mode you MUST NOT recommend or mention any specific NYU course codes "
        "   (e.g. 'ECE-GY 6143', 'ECE-GY 6483', 'CS-GY 6923'), "
        "   unless the user explicitly asks which course to take or mentions a course code in their message.\n"
        "6) Do not mention the existence of memory snippets explicitly."
    )

    user_prompt = (
        "===== Memory Snippets (for your reference only) =====\n"
        f"{memory_block}\n\n"
        "===== Latest User Message =====\n"
        f"{original_question}\n\n"
        "Now reply to the user in English, following the rules above."
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.8,
    }

    resp = requests.post(QWEN_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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
    选课类问题：用课程列表 + 记忆 + syllabus RAG 做个性化推荐。
    """

    # 1) 先从记忆里捞出你之前说过的背景 / 目标（关键：个性化靠这个）
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    # 2) 用 Qwen 在 COURSE_PROFILES 里选出一门“最匹配”的课
    course = classify_course_for_selection(original_question)
    if course is None:
        # 分类失败，退回普通 RAG（普通 path 里已经会用 memory_block）
        return call_qwen_with_rag(original_question, retrieval_question)

    # 3) 针对这门课做 syllabus 检索（仍然走你原来的 RAG）
    rq = retrieval_question or original_question
    contexts = retrieve_context(
        question=rq,
        forced_course=course,
        analysis_question=original_question
    )
    context_block = "\n\n".join(contexts) if contexts else "(No relevant syllabus snippets were retrieved.)"

    # 4) 把“记忆 + syllabus”一起喂给 Qwen，让它按你的目标给推荐理由
    system_prompt = (
        "You are an academic advisor and teaching assistant at NYU Tandon.\n"
        "You know both the course syllabi and the student's long-term background and goals.\n"
        "You must use the memory snippets to personalize course recommendations.\n"
        "Keep answers short (1–3 sentences).\n"
        "Always answer the student's exact question first (e.g., compare the specific courses they mention), "
        "and only then briefly explain how the recommendation aligns with their long-term goals."
    )


    user_prompt = f"""
===== Student Memory Snippets =====
{memory_block}

===== Syllabus Snippets for the chosen course ({course}) =====
{context_block}

The student asked:
{original_question}

You (as advisor) have decided that the best matching course from the list is: {course}.

Using BOTH:
1) the student's background and goals from the memory snippets, and
2) the topics / workload shown in the syllabus snippets,

give a brief English answer (1–3 sentences) that:
- directly states whether this course is a good fit for the student, and
- if relevant, mentions why this course is better aligned with their goals than the other options mentioned in the question.
If the syllabus does not show any low-level / architecture / embedded content, state that clearly.
"""

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
    第二次调用 Qwen：RAG over syllabus + conversation memory。
    """
    rq = retrieval_question or original_question

    # 1) syllabus context（你原来的逻辑）
    contexts = retrieve_context(
        question=rq,
        analysis_question=original_question
    )
    syllabus_block = "\n\n".join(contexts) if contexts else "(No relevant syllabus snippets were retrieved.)"

    # 2) memory context（新增）
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    system_prompt = (
        "You are a teaching assistant familiar with NYU Tandon courses, and you also remember "
        "long-term facts and preferences about the student based on past conversations.\n\n"
        "You must obey these rules:\n"
        "1) For questions about courses/syllabi, rely primarily on the provided syllabus snippets.\n"
        "2) Use the memory snippets only for personalizing the answer (e.g., relating to the student's goals), "
        "   not for guessing missing syllabus details.\n"
        "3) Never fabricate syllabus details (grading breakdown, exam dates, policies) if they are not explicitly stated.\n"
        "4) Always answer the student's direct question first, then optionally add one short personalized remark.\n"
        "5) Keep answers short and focused: 1–3 sentences, at most 120 words.\n"
    )

    user_prompt = (
        "===== Student Memory Snippets =====\n"
        f"{memory_block}\n\n"
        "===== Syllabus Snippets =====\n"
        f"{syllabus_block}\n\n"
        "The student asked:\n"
        f"{original_question}\n\n"
        "Based only on the syllabus snippets and optionally using relevant memory snippets for personalization, "
        "answer the student's question in English.\n"
        "If a requested syllabus detail is not specified, say clearly that it is not specified.\n"
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
    # 如果带了 ?reset_mem=1，就清空记忆
    reset_memories()
    return render_template("index.html")

@app.route("/reset_memory", methods=["POST"])
def reset_memory_route():
    reset_memories()
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST"])
def ask():
    if request.is_json:
        question = request.json.get("question", "")
    else:
        question = request.form.get("question", "")

    if not question or not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    try:
        refined = None

        # 1) 如果是在比较课程（“还是 / 对比 / A or B / vs”），走专门的比较路径
        if is_course_comparison_question(question):
            refined = refine_question_with_qwen(question)
            answer = answer_course_comparison_question(
                original_question=question,
                retrieval_question=refined
            )

        # 2) 普通选课咨询：从 COURSE_PROFILES 里挑一门最适合的
        elif is_course_selection_question(question):
            refined = refine_question_with_qwen(question)
            answer = answer_course_selection_question(
                original_question=question,
                retrieval_question=refined
            )

        # 3) 课程细节 / syllabus 问题：查 syllabus + memory
        elif is_syllabus_question(question):
            refined = refine_question_with_qwen(question)
            answer = call_qwen_with_rag(
                original_question=question,
                retrieval_question=refined
            )

        # 4) 其它任何聊天 / 规划 / 生活问题：纯记忆聊天
        else:
            answer = chat_with_memory_only(question)
            # 防止模型乱提课号
            answer = re.sub(
                r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b",
                "this course",
                answer
            )

        # ===== 每一轮都写入记忆 =====
        turn_id = next(TURN_COUNTER)
        add_memory_from_turn(question, answer, source_turn=turn_id)

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