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
from config import COURSE_PROFILES, ID_TO_FULLCODE, ALIAS_TO_FULLCODE

# ================== 课程配置（只在这里维护） ==================

COURSE_PROFILES: List[Dict] = [
    # --- Existing Courses (已有的) ---
    {
        "code": "ECE-GY 6143",
        "name": "Introduction to Machine Learning",
        "focus": "machine learning, statistics, supervised and unsupervised learning, python, classification, regression",
    },
    {
        "code": "ECE-GY 6913",
        "name": "Computer System Architecture",
        "focus": "RISC-V instruction set, pipelined processor, cache, memory hierarchy, low-level architecture, GPU/TPU",
    },
    {
        "code": "ECE-GY 6483",
        "name": "Real Time Embedded Systems",
        "focus": "ARM Cortex-M, STM32, real-time OS, peripherals, embedded C, IoT, sensors",
    },

    # --- New Additions (从你的JSON数据中补充的) ---
    {
        "code": "ECE-GY 5253",
        "name": "Applied Matrix Theory",
        "focus": "linear algebra, SVD, eigenvalues, matrix decomposition, system stability, engineering mathematics",
    },
    {
        "code": "ECE-GY 6113",
        "name": "Digital Signal Processing I",
        "focus": "DSP, FFT, discrete-time signals, FIR/IIR filters, Z-transform, spectral analysis, MATLAB",
    },
    {
        "code": "EL5613", # 对应 Power Systems
        "name": "Introduction to Electric Power Systems",
        "focus": "power grid analysis, AC circuits, transmission lines, three-phase systems, fault analysis, load flow",
    },
    {
        "code": "EL6253", # 对应 Linear Systems
        "name": "Linear Systems",
        "focus": "control theory, state-space analysis, stability, controllability, observability, feedback systems",
    },
]

# ================== 工具：从问题里抓课程号（只用在对比问题） ==================

def _extract_course_codes(question: str) -> List[str]:
    """
    V3.0: 极速版。利用 config 计算好的映射表提取课程。
    """
    q_upper = question.upper()
    found_codes = set()

    # 1. 抓数字 (比如 6143, 6003)
    for m in re.finditer(r"\b(\d{4})\b", q_upper):
        num = m.group(1)
        if num in ID_TO_FULLCODE:
            found_codes.add(ID_TO_FULLCODE[num])
            
    # 2. 抓别名 (比如 ML, RISC-V, RTOS)
    # 直接利用 ALIAS_TO_FULLCODE 字典
    for alias, full_code in ALIAS_TO_FULLCODE.items():
        # 使用正则边界 \b 防止匹配单词内部
        if re.search(rf"\b{re.escape(alias)}\b", q_upper):
            found_codes.add(full_code)

    return sorted(list(found_codes))

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
        max_tokens=1024,
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
        max_tokens=2048,
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


# advisor_module.py

def answer_course_selection_question(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    [Token 优化版] 选课推荐：
    策略：
    1. 先分类选出一门最匹配的课。
    2. 提取该课的 Official Description (JSON) 作为核心依据。
    3. 仅检索 Top-1 Syllabus 片段作为补充。
    4. 结合记忆，生成高可信度的推荐理由。
    """

    # 1. 记忆片段 (Personalization) - 降为 Top-3 节省 Token
    mem_items = retrieve_memories(original_question, top_k=3)
    memory_block = format_memories_block(mem_items)

    # 2. 选课分类 (决策步骤)
    # 先决定推荐哪门课
    course_code = classify_course_for_selection(original_question)
    
    if course_code is None:
        # 分类失败（没找到合适的课），回退到通用 RAG
        return call_qwen_with_rag(original_question, retrieval_question)

    # 3. 准备核心数据 (Official Anchor)
    # 从 COURSE_PROFILES 里找到这门课的字典
    target_profile = next((c for c in COURSE_PROFILES if c["code"] == course_code), None)
    
    if target_profile:
        # 优先用 focus (我们在 config 里合成的 description + blurb)
        official_desc = target_profile.get("focus", "No official description available.")
        course_name = target_profile.get("name", "")
    else:
        official_desc = "Unknown Course"
        course_name = ""

    # 4. 准备补充数据 (RAG Supplement) - 限制 Top-1 & 截断
    rq = retrieval_question or original_question
    contexts = retrieve_context(
        question=rq,
        forced_course=course_code,
        analysis_question=original_question,
    )
    
    # 强制只取第 1 条，且最多取前 600 字符
    # 这样既能回答“有考试吗”这种细节，又不会引入太多噪声
    if contexts:
        best_rag_snippet = contexts[0][:600] 
        # 如果截断了，加个省略号提示模型
        if len(contexts[0]) > 600:
            best_rag_snippet += "...(truncated)"
    else:
        best_rag_snippet = "(No specific syllabus details found via search.)"

    # 5. 构造高密度输入块
    course_data_block = f"""
### Selected Course: {course_code} - {course_name}
* **Core Focus (Official Source - TRUST THIS):** {official_desc}

* **Syllabus Excerpt (Detail Source - Use for grading/exams):** {best_rag_snippet}
"""

    # 6. 构造 Prompt
    system_prompt = (
        "You are an academic advisor at NYU Tandon.\n"
        "Recommend the selected course to the student based on the provided data.\n\n"
        "Rules:\n"
        "1) TRUST 'Core Focus' for what the course is actually about.\n"
        "2) Use 'Syllabus Excerpt' ONLY for logistical details (exams, grading, tools). If it conflicts with Core Focus, ignore it.\n"
        "3) Connect the course to the student's background (from Memory).\n"
        "4) If the syllabus excerpt is missing details (e.g. no mention of exams), state that clearly. DO NOT INVENT.\n"
        "5) Answer in 2-4 sentences, encouraging and professional."
    )

    user_prompt = f"""
User Context (Memory):
{memory_block}

Course Data:
{course_data_block}

Student Question: "{original_question}"

Task:
Explain why {course_code} is the right choice for this student.
"""

    # 7. 调用生成
    # 增大 max_tokens，让模型有足够空间写出流畅的推荐语
    answer = call_qwen(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024, 
        temperature=0.2, # 稍微提高一点点温度让推荐语不那么生硬，但依然保持低位
        top_p=0.8,
    )
    
    return answer


# ================== 课程对比（两门课） ==================

# advisor_module.py

# 确保引入了配置里的课程信息
from config import COURSE_PROFILES

def answer_course_comparison_question(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    [Token 优化版] 课程对比：
    策略：
    1. 优先使用 JSON 中的 Official Description (高密度，低 Token 消耗)。
    2. 限制 RAG 检索仅返回 Top-1 片段 (仅用于补充细节)。
    这样可以大幅压缩 Input Token，让模型有更多余地生成 Output。
    """
    # 1. 获取课程号
    codes = _extract_course_codes(original_question) 
    
    if len(codes) < 2:
        return call_qwen_with_rag(original_question, retrieval_question)

    rq = retrieval_question or original_question
    
    # 建立查找表：快速获取官方描述 (这是最省 Token 的核心信息)
    # 结构: {"ECE-GY 6143": "Focuses on ML algorithms...", ...}
    # 注意：确保你的 config/__init__.py 里正确生成了 focus 字段
    course_desc_map = {c["code"]: c.get("focus", "No description.") for c in COURSE_PROFILES}
    
    syllabus_segments = []
    print(f"[Comparison] Comparing {len(codes)} courses: {codes}")

    for code in codes:
        # =====================================================
        # 优化点 1: 限制 RAG 检索数量
        # =====================================================
        # 对比时，我们不需要面面俱到，只需要最核心的那一段
        contexts = retrieve_context(
            question=rq,
            forced_course=code,
            analysis_question=original_question
        )
        
        # 强制只取 Top-1 (最相关的一段)，大幅节省 Token
        # 假设 contexts 是一个 list[str]
        best_rag_snippet = contexts[0] if contexts else "(No syllabus snippet found)"
        
        # 获取官方描述
        official_desc = course_desc_map.get(code, "Unknown Course")

        # =====================================================
        # 优化点 2: 结构化精简输入
        # =====================================================
        # 明确区分 "Official" (概览) 和 "Detail" (RAG)
        segment = f"""
### Course: {code}
* **Core Focus (Official):** {official_desc}
* **Syllabus Excerpt (Detail):** {best_rag_snippet[:500]} 
""" 
        # 注意：上面的 [:500] 是为了防止某一段 RAG 异常长，强制截断
        syllabus_segments.append(segment)

    # 3. 拼接
    all_syllabus_block = "\n".join(syllabus_segments)

    # 4. 记忆 (Personalization)
    mem_items = retrieve_memories(original_question, top_k=3) # 也可以把记忆从 top-5 降为 top-3
    memory_block = format_memories_block(mem_items)

    # 5. Prompt
    system_prompt = (
        "You are an academic advisor at NYU Tandon.\n"
        "Compare the courses based on the provided summaries.\n"
        "Rules:\n"
        "1) Use 'Core Focus' for the main topic comparison.\n"
        "2) Use 'Syllabus Excerpt' only for specific details like grading/exams.\n"
        "3) Rank/Recommend the best fit for the student.\n"
        "4) Answer in depth but stay structured."
    )

    course_list_str = ", ".join(codes)
    user_prompt = f"""
User Context:
{memory_block}

Course Data:
{all_syllabus_block}

Question: "{original_question}"

Task: Compare {course_list_str} regarding difficulty, topics, and workload. Recommend the best fit.
"""

    # 6. 调用模型
    # Input Token 省下来了，这里 max_tokens 就可以大胆给高
    answer = call_qwen(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2048,  # 给足够的空间生成内容
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
