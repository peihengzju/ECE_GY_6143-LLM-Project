# ingest_syllabi_optimized.py
import os
import json
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

SYLLABUS_DIR = "docs/syllabus"
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 200

E5_MODEL_NAME = "intfloat/multilingual-e5-large"


# ========== 课程号推断 ==========

COURSE_CODE_PATTERN = re.compile(r"(ECE[_\-\s]?GY[_\-\s]?(\d{4}))", re.IGNORECASE)


def infer_course_from_filename(fname: str) -> str | None:
    """
    从文件名里推断课程号，统一成形如 'ECE-GY 6143'
    例如：
      ECE_GY_6143 syllabus.pdf
      ece-gy6143_Syllabus_Fall2025.pdf
    """
    m = COURSE_CODE_PATTERN.search(fname)
    if not m:
        return None

    raw = m.group(1).upper()  # ECE_GY_6143 / ECE-GY6143 / ECE GY 6143
    raw = raw.replace("_", " ").replace("-", " ")
    parts = raw.split()  # ['ECE', 'GY', '6143'] / ['ECEGY6143'] 之类

    if len(parts) == 3 and parts[0] == "ECE" and parts[1] == "GY":
        # ECE GY 6143 -> ECE-GY 6143
        return f"{parts[0]}-{parts[1]} {parts[2]}"

    if len(parts) == 2 and parts[0].startswith("ECE") and "GY" in parts[0]:
        # 比如 ECEGY 6143 这种奇怪格式，兜底处理一下
        return f"ECE-GY {parts[1]}"

    # 兜底：至少保证有 ECE-GY 前缀
    if " " in raw:
        head, tail = raw.rsplit(" ", 1)
        return f"{head.replace(' ', '-') } {tail}"

    return raw


# ========== 基础工具 ==========

def load_all_pdfs():
    texts: List[str] = []
    meta: List[Dict] = []

    for fname in os.listdir(SYLLABUS_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(SYLLABUS_DIR, fname)
        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue
            texts.append(page_text)
            meta.append({"file": fname, "page": i})

    return texts, meta


def chunk_by_lines(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    coarse_chunks: List[str] = []
    cur = ""

    for ln in lines:
        candidate = (cur + "\n" + ln) if cur else ln
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                coarse_chunks.append(cur)
            cur = ln

    if cur:
        coarse_chunks.append(cur)

    final_chunks: List[str] = []
    for ch in coarse_chunks:
        if len(ch) <= max_chars:
            final_chunks.append(ch)
        else:
            start = 0
            while start < len(ch):
                part = ch[start:start + max_chars]
                if part:
                    final_chunks.append(part)
                start += max_chars - overlap_chars

    return final_chunks


def _join_window(lines: List[str], start: int, end: int) -> str:
    return "\n".join(ln for ln in lines[start:end] if ln.strip())


# ========== Grading 提取 ==========

def _is_grading_header(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return False

    if low.startswith("grading"):
        return True
    if low.startswith("grade distribution"):
        return True
    if low.startswith("evaluation"):
        return True
    if low.startswith("assessment"):
        return True
    if "grading" in low and ("evaluation" in low or "assessment" in low):
        return True

    return False


def extract_grading_sections(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    i = 0
    while i < n:
        line = lines[i]
        if _is_grading_header(line):
            start = i
            j = i + 1
            while j < n:
                next_line = lines[j]
                if _is_grading_header(next_line):
                    break
                stripped = next_line.strip()
                if stripped.isupper() and len(stripped.split()) <= 6:
                    break
                j += 1

            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)
            i = j
        else:
            i += 1

    return chunks


def extract_grading_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: List[str] = []

    for ln in lines:
        low = ln.lower()
        if "%" in ln or "percent" in low or "grade" in low:
            if len(ln) < 10:
                continue
            if ln not in chunks:
                chunks.append(ln)

    return chunks


# ========== 课程描述 / 老师 / TA / 上课信息等结构化提取 ==========

def extract_course_description(text: str) -> List[str]:
    """
    课程名 + 编号 + Description 那一段
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "description:" in low or low.startswith("description "):
            start = max(0, i - 1)
            j = i + 1
            while j < n:
                nxt = lines[j].strip()
                if not nxt:
                    break
                if nxt.isupper() and len(nxt.split()) <= 6:
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_instructor_info(text: str) -> List[str]:
    """
    Professor / Instructor / Office hours 等
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "professor" in low or "instructor" in low:
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["grader", "lecture", "class material", "prereq", "pre-requisites"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_graders_info(text: str) -> List[str]:
    """
    Graders 列表、邮件等
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        if "grader" in ln.lower():
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["lecture", "office hours", "class material", "tentative schedule"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_lecture_info(text: str) -> List[str]:
    """
    Lecture: 时间、地点、是否 zoom、出勤要求、带电脑等
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        if "lecture:" in ln.lower():
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["grading", "grader", "class material", "tentative schedule"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_schedule(text: str) -> List[str]:
    """
    Tentative schedule / per-week topics。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "tentative schedule" in low or "schedule of classes" in low:
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["grading", "prereq", "pre-requisites", "class material"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)

    if not chunks:
        date_pattern = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
        tmp_lines = []
        for ln in lines:
            if date_pattern.search(ln):
                tmp_lines.append(ln)
        if tmp_lines:
            window = "\n".join(tmp_lines)
            chunks.append(window)

    return chunks


def extract_exam_info(text: str) -> List[str]:
    """
    Midterm / Final exam 时间、review。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in ["midterm", "final exam", "final  exam", "exam review"]):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_prerequisites(text: str) -> List[str]:
    """
    Pre-requisites / Prerequisites 部分。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("pre-requisites") or low.startswith("prerequisites"):
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["class material", "online format", "schedule", "grading"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_materials(text: str) -> List[str]:
    """
    Class material / Textbook / GitHub / 网站等。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "class material" in low or "textbook" in low:
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["online format", "tentative schedule", "grading"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)

    urls = []
    for ln in lines:
        if "http://" in ln or "https://" in ln or "github.com" in ln.lower():
            urls.append(ln.strip())
    if urls:
        window = "\n".join(urls)
        if window not in chunks:
            chunks.append(window)

    return chunks


def extract_project_info(text: str) -> List[str]:
    """
    Optional project / Project 占比等。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "project" in low:
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_homework_lab(text: str) -> List[str]:
    """
    Homework / Labs 相关。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "homework" in low or "lab" in low or "labs" in low:
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_online_format(text: str) -> List[str]:
    """
    Online format / zoom / pre-recorded / attendance 等课程形式。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "online format" in low or "zoom" in low or "pre-recorded" in low or "attendance" in low:
            start = max(0, i - 1)
            end = min(len(lines), i + 4)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


# ========== 主流程 ==========

def main():
    print("加载 syllabus PDF ...")
    raw_texts, meta_list = load_all_pdfs()

    docs: List[Dict] = []
    seen_texts = set()

    for page_text, m in zip(raw_texts, meta_list):
        # 在这里根据文件名推断课程号，写进 meta
        course = infer_course_from_filename(m["file"])

        base_meta = {
            **m,
            "course": course,  # 关键字段
        }

        # 1) normal
        for c in chunk_by_lines(page_text):
            t = c.strip()
            if not t or t in seen_texts:
                continue
            seen_texts.add(t)
            docs.append({
                "text": t,
                "meta": {**base_meta, "type": "normal"}
            })

        # 2) grading
        for gc in extract_grading_sections(page_text):
            t = gc.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "grading_section"}
                })

        for gl in extract_grading_lines(page_text):
            t = gl.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "grading_line"}
                })

        # 3) 课程描述
        for chunk in extract_course_description(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "course_description"}
                })

        # 4) 老师 / TA / 上课信息
        for chunk in extract_instructor_info(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "instructor"}
                })

        for chunk in extract_graders_info(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "grader"}
                })

        for chunk in extract_lecture_info(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "lecture_info"}
                })

        # 5) schedule / exam
        for chunk in extract_schedule(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "schedule"}
                })

        for chunk in extract_exam_info(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "exam"}
                })

        # 6) prereq / materials / project / homework / online format
        for chunk in extract_prerequisites(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "prerequisites"}
                })

        for chunk in extract_materials(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "materials"}
                })

        for chunk in extract_project_info(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "project"}
                })

        for chunk in extract_homework_lab(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "homework_lab"}
                })

        for chunk in extract_online_format(page_text):
            t = chunk.strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                docs.append({
                    "text": t,
                    "meta": {**base_meta, "type": "online_format"}
                })

    print(f"共 {len(docs)} 个 chunk（含多种类型）")

    # ====== 计算 embedding ======
    print("加载 embedding 模型并计算向量 ...")
    model = SentenceTransformer(E5_MODEL_NAME)
    texts = [d["text"] for d in docs]

    emb_input = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        emb_input,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print("完成：")
    print(f"  - 向量索引 -> {INDEX_PATH}")
    print(f"  - 文本数据 -> {TEXTS_PATH}")


if __name__ == "__main__":
    main()