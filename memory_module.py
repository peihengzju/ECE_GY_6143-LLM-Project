# memory_module.py
"""
负责“对话记忆”的模块：
- 存储到 memory_store/memories.json
- 用 Faiss 做向量检索 (mem_index.faiss)
- 提供：reset_memories / add_memory_from_turn / retrieve_memories / format_memories_block
"""

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.paths import (
    MEMORY_DIR,
    MEMORY_INDEX_PATH,
    MEMORY_TEXTS_PATH,
    E5_MODEL_NAME,
)
from qwen_client import call_qwen

# ========== 基础配置 ==========

MemorySlot = Literal["profile", "preference", "fact", "recent"]

# 记忆槽类型（现在主要是做说明）
MEMORY_SLOTS = {
    "profile",        # 个人信息：姓名、学校、专业、背景
    "preference",     # 长期偏好：想学什么、风格偏好
    "fact",           # 重要事实：比如“我是 NYU ECE”、“我想走 AI infra”
    "recent",         # 临时上下文：最近 5～10 轮
}

# 不同槽采用不同的过期 / 权重策略
MEMORY_EXPIRY_DAYS: Dict[MemorySlot, Optional[int]] = {
    "profile":   None,   # 永不过期
    "preference": 365,   # 软过期，一年
    "fact":      180,    # 半年
    "recent":     7,     # 一周
}


@dataclass
class MemoryItem:
    id: int
    slot: MemorySlot
    text: str                  # 可能是摘要后的文本
    importance: int            # 0 = 一般，1 = 中等，2 = 很重要，3 = 更重要（聚合）
    created_at: str            # ISO 字符串
    last_used_at: str          # 上次被检索到的时间
    extra: Dict[str, Any]      # 额外 meta, 比如 { "source_turn": 12, "kind": "career_direction" }


# ========== 初始化目录和文件 ==========

os.makedirs(MEMORY_DIR, exist_ok=True)

if not os.path.exists(MEMORY_TEXTS_PATH):
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# embedding 模型（记忆这边也用 e5）
_emb_model = SentenceTransformer(E5_MODEL_NAME)

# 记忆向量索引，全局变量
if os.path.exists(MEMORY_INDEX_PATH):
    mem_index = faiss.read_index(MEMORY_INDEX_PATH)
else:
    # 没有 index，用 dummy 算出维度
    dummy = _emb_model.encode(
        ["query: dummy"], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    d = len(dummy)
    mem_index = faiss.IndexFlatIP(d)  # 内积，embedding 已经 normalize，相当于余弦相似度
    faiss.write_index(mem_index, MEMORY_INDEX_PATH)


def _save_mem_index() -> None:
    faiss.write_index(mem_index, MEMORY_INDEX_PATH)


# ========== 通用工具 ==========

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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        summary = call_qwen(messages, max_tokens=128, temperature=0.1, top_p=0.8)
        summary = (summary or "").strip()
        return summary if summary else snippet[:200]
    except Exception:
        # 摘要失败就退回截断原文
        return snippet[:200]


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
        dummy = _emb_model.encode(
            ["query: dummy"], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        d = len(dummy)
        mem_index = faiss.IndexFlatIP(d)
        _save_mem_index()
        return

    # 正常重建
    dummy = _emb_model.encode(
        ["query: dummy"], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    d = len(dummy)
    mem_index = faiss.IndexFlatIP(d)

    texts = [f"passage: {m.text}" for m in items]
    embs = _emb_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
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


# ========== 槽分类 & 聚合器 ==========

class MemoryAggregator:
    """
    通用聚合记忆基类：
    - kind: 用于 extra["kind"] 标记聚合类型（如 'career_direction', 'profile_aggregate'）
    - slot: 这个聚合记忆属于哪个槽（profile / preference / fact / recent）
    - base_importance: 聚合记忆的基础重要度
    """
    kind: str
    slot: MemorySlot
    base_importance: int

    def extract_entities(self, text: str) -> List[str]:
        """
        从一轮对话（question+answer）里抽取该聚合类型的“实体列表”。没有就返回 []。
        子类必须重写。
        """
        raise NotImplementedError

    def render_text(self, entities: List[str]) -> str:
        """
        把实体列表渲染成一条人类可读的记忆文本。
        子类可以重写。
        """
        return f"{self.kind}: " + ", ".join(entities)

    def upsert(
        self,
        items: List[MemoryItem],
        question: str,
        answer: str,
        source_turn: int,
    ) -> List[MemoryItem]:
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
    if any(k in text for k in ["nyu", "tandon", "ece", "major", "degree", "master", "phd", "硕士", "博士"]):
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

    def extract_entities(self, text: str) -> List[str]:
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

    def render_text(self, entities: List[str]) -> str:
        return "User's long-term career directions: " + ", ".join(entities)


class ProfileAggregator(MemoryAggregator):
    kind = "profile_aggregate"
    slot: MemorySlot = "profile"
    base_importance = 3

    def extract_entities(self, text: str) -> List[str]:
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

    def render_text(self, entities: List[str]) -> str:
        return "User's profile: " + ", ".join(entities)


AGGREGATORS: List[MemoryAggregator] = [
    CareerDirectionAggregator(),
    ProfileAggregator(),
    # 以后你要加 SkillsAggregator、PreferenceAggregator，直接加到这里
]


# ========== 写入记忆 & 检索记忆 ==========

def add_memory_from_turn(question: str, answer: str, source_turn: int) -> None:
    """
    每一轮问答结束后调用：
    - question/answer 先拼 snippet，必要时用 Qwen 压缩
    - 写一条“原始记忆条目”
    - 让所有 AGGREGATOR 更新/生成聚合记忆
    - 统一写回文件并重建 Faiss index
    """
    snippet = _build_dialogue_snippet(question, answer)
    # 超过 700 字做摘要
    if len(snippet) > 700:
        text = summarize_dialogue_with_qwen(snippet, max_words=80)
    else:
        text = snippet

    # 1) 原始记忆条目
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
        extra={"source_turn": source_turn},
    )
    items.append(raw_mem)

    # 2) 跑所有 Aggregator，生成/更新聚合记忆条目
    for agg in AGGREGATORS:
        items = agg.upsert(items, question, answer, source_turn)

    # 3) 统一落盘 + 重建向量索引
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

    # 简单线性衰减
    return max(0.0, 1.0 - delta_days / expiry_days)


def retrieve_memories(
    question: str,
    top_k: int = 5,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> List[MemoryItem]:
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
    q_emb = _emb_model.encode(
        [q_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
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

        # 聚合记忆加点额外偏置
        kind = m.extra.get("kind")
        if kind == "career_direction":
            overall += 1.0
        elif kind == "profile_aggregate":
            overall += 0.8

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
    """
    把若干 MemoryItem 格式化成一个多行字符串，方便塞进 Qwen prompt。
    """
    if not mem_items:
        return "(No retrieved memories.)"
    lines = []
    for m in mem_items:
        lines.append(f"[{m.slot} | importance={m.importance}] {m.text}")
    return "\n".join(lines)


__all__ = [
    "MemoryItem",
    "MemorySlot",
    "reset_memories",
    "add_memory_from_turn",
    "retrieve_memories",
    "format_memories_block",
]