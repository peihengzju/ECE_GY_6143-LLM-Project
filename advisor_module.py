# advisor_module.py
"""
负责“学术 advisor”逻辑的模块：

- 纯记忆聊天（不查 syllabus）
- syllabus RAG + 记忆 混合回答
- 选课问题（在 COURSE_PROFILES 里挑课 + syllabus + 记忆）
- 课程对比问题（两门课 syllabus + 记忆）

对外暴露的主要函数：
    - chat_with_memory_only(original_question: str) -> str
    - call_qwen_with_rag(original_question: str, retrieval_question: Optional[str]) -> str
    - answer_course_selection_question(original_question: str, retrieval_question: Optional[str]) -> str
    - answer_course_comparison_question(original_question: str, retrieval_question: Optional[str]) -> str
    - classify_course_for_selection(question: str) -> Optional[str]
    - COURSE_PROFILES: List[dict]
"""

from __future__ import annotations
import json
import re
from typing import List, Optional, Dict

from memory_module import retrieve_memories, format_memories_block
from rag_module import retrieve_context
from qwen_client import call_qwen

# ================== 课程配置（只在这里维护） ==================

COURSE_PROFILES: List[Dict] = [
    {
        "code": "ECE-GY 6143",
        "name": "Introduction to Machine Learning",
        "focus": "machine learning, statistics, supervised and unsupervised learning",
    },
    {
        "code": "ECE-GY 6913",
        "name": "Computer System Architecture",
        "focus": "RISC-V instruction set, pipelined processor, cache, memory hierarchy, low-level architecture",
    },
    {
        "code": "ECE-GY 6483",
        "name": "Real Time Embedded Systems",
        "focus": "ARM Cortex-M, STM32, real-time OS, peripherals, embedded C",
    },
    # 以后要参与“选课推荐”的课，继续往下面加
]


# ================== 工具：从问题里抓课程号（只用在对比问题） ==================

def _extract_course_codes(question: str) -> List[str]:
    """
    从问题里抽取类似 ECE-GY 6143 / CS-GY 6923 这样的课程号。
    和 rag_module 里的逻辑一样，但这里只用于对比问题。
    """
    pattern = re.compile(r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b")
    codes = pattern.findall(question)
    return [c.strip() for c in codes]


# ================== 纯记忆聊天 ==================

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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=512,
        temperature=0.3,
        top_p=0.8,
    )
    return answer


# ================== syllabus RAG + 记忆 ==================

def call_qwen_with_rag(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    syllabus RAG + 记忆：
    - 先用 retrieve_context 拿 syllabus 片段
    - 再把记忆 snippets 一起喂给 Qwen
    """
    rq = retrieval_question or original_question

    # 1) syllabus context
    contexts = retrieve_context(
        question=rq,
        analysis_question=original_question,
    )
    syllabus_block = (
        "\n\n".join(contexts)
        if contexts
        else "(No relevant syllabus snippets were retrieved.)"
    )

    # 2) memory context
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=512,
        temperature=0.2,
        top_p=0.8,
    )
    return answer


# ================== 选课（单门推荐） ==================

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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = call_qwen(
        messages,
        max_tokens=128,
        temperature=0.1,
        top_p=0.8,
    )

    try:
        result = json.loads(raw)
        course = result.get("course")
        if course and course != "unknown":
            return course
    except Exception:
        # 模型没按 JSON 输出，就认为分类失败
        pass
    return None


def answer_course_selection_question(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    选课类问题：用课程列表 + 记忆 + syllabus RAG 做个性化推荐。
    """

    # 1) 记忆片段（个性化靠这个）
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    # 2) 在 COURSE_PROFILES 里选一门“最匹配”的课
    course = classify_course_for_selection(original_question)
    if course is None:
        # 分类失败，退回普通 syllabus RAG
        return call_qwen_with_rag(original_question, retrieval_question)

    # 3) 针对这门课做 syllabus 检索（强制该课程）
    rq = retrieval_question or original_question
    contexts = retrieve_context(
        question=rq,
        forced_course=course,
        analysis_question=original_question,
    )
    context_block = (
        "\n\n".join(contexts)
        if contexts
        else "(No relevant syllabus snippets were retrieved.)"
    )

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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=256,
        temperature=0.2,
        top_p=0.8,
    )
    return answer


# ================== 课程对比（两门课） ==================

def answer_course_comparison_question(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    课程对比问题（A vs B / A 还是 B）：
    - 抽两个课程号
    - 分别对每门课做 syllabus 检索
    - 加上记忆 snippets，一起给 Qwen 做对比 + 个性化建议
    """
    codes = _extract_course_codes(original_question)
    if len(codes) < 2:
        # 没抓到两个课号，就退回普通 RAG 路径
        return call_qwen_with_rag(original_question, retrieval_question)

    # 只用前两个
    code_a = codes[0]
    code_b = codes[1]

    rq = retrieval_question or original_question

    # 两门课分别检索 syllabus
    contexts_a = retrieve_context(
        question=rq,
        forced_course=code_a,
        analysis_question=original_question,
    )
    contexts_b = retrieve_context(
        question=rq,
        forced_course=code_b,
        analysis_question=original_question,
    )

    context_block_a = (
        "\n\n".join(contexts_a) if contexts_a else "(No syllabus snippets retrieved for this course.)"
    )
    context_block_b = (
        "\n\n".join(contexts_b) if contexts_b else "(No syllabus snippets retrieved for this course.)"
    )

    # 记忆用于 personalization
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    system_prompt = (
        "You are an academic advisor and teaching assistant at NYU Tandon.\n"
        "You will compare two specific courses for the same student using both syllabi and the student's background.\n\n"
        "Rules:\n"
        "1) First, directly answer the student's comparison question (which course is more suitable for them).\n"
        "2) Then briefly justify your choice using concrete topics / workload from the syllabus snippets of both courses.\n"
        "3) Use the memory snippets only to personalize the reasoning (e.g., relate to their interest in embedded systems, AI infra, etc.).\n"
        "4) Do NOT fabricate syllabus details that are not explicitly present in the snippets.\n"
        "5) Keep the answer concise: 2–4 sentences at most."
    )

    user_prompt = f"""
===== Student Memory Snippets =====
{memory_block}

===== Syllabus Snippets for {code_a} =====
{context_block_a}

===== Syllabus Snippets for {code_b} =====
{context_block_b}

The student asked:
{original_question}

Based on the two syllabi and the student's background, decide which of {code_a} and {code_b} is more suitable for this student.
First clearly state which course you recommend, then briefly compare them and explain why that choice fits the student's goals better.
Answer in English, in 2–4 sentences.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=320,
        temperature=0.2,
        top_p=0.8,
    )
    return answer


__all__ = [
    "COURSE_PROFILES",
    "chat_with_memory_only",
    "call_qwen_with_rag",
    "classify_course_for_selection",
    "answer_course_selection_question",
    "answer_course_comparison_question",
]