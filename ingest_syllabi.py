# ingest_syllabi.py
import os
import json
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

SYLLABUS_DIR = "docs/syllabus"
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

# 把 chunk 切小一点，方便抓细节
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

E5_MODEL_NAME = "intfloat/multilingual-e5-base"


def load_all_pdfs():
    """
    读取 docs/syllabus 下所有 pdf。
    返回：texts: List[str], meta: List[Dict]
    其中 meta 里至少包含 file, page。
    """
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


def chunk_words(text: str,
                size: int = CHUNK_SIZE_WORDS,
                overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """按词数滑动窗口切块。"""
    words = text.split()
    chunks: List[str] = []

    for i in range(0, len(words), size - overlap):
        part = " ".join(words[i:i + size])
        if part:
            chunks.append(part)

    return chunks


def extract_grading_chunks(text: str) -> List[str]:
    """
    专门在一页里挖出可能跟 grading / 占比 相关的局部小段，
    例如包含 'Grading', 'grading', 'Evaluation', '%' 等的行，
    以及其上下几行，作为单独的 chunk。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    grading_chunks: List[str] = []

    for i, line in enumerate(lines):
        low = line.lower()

        # 命中关键字 / 百分比
        if (
            "grading" in low
            or "grade distribution" in low
            or "evaluation" in low
            or "assessment" in low
            or "%" in line  # 行里有百分比，大概率是占比
        ):
            start = max(0, i - 3)
            end = min(n, i + 4)
            window = "\n".join(lines[start:end])
            # 去掉太短的噪声，避免重复加入
            if len(window) >= 30 and window not in grading_chunks:
                grading_chunks.append(window)

    return grading_chunks


def main():
    print("加载 syllabus PDF ...")
    raw_texts, meta_list = load_all_pdfs()

    docs: List[Dict] = []

    for page_text, m in zip(raw_texts, meta_list):
        # 1) 正常按词数切块（整页内容）
        for c in chunk_words(page_text):
            docs.append({
                "text": c,
                "meta": {**m, "type": "normal"}
            })

        # 2) 额外抽取“grading 相关小块”，单独当成 chunk
        grading_chunks = extract_grading_chunks(page_text)
        for gc in grading_chunks:
            docs.append({
                "text": gc,
                "meta": {**m, "type": "grading"}
            })

    print(f"共 {len(docs)} 个 chunk（包括 normal + grading 专门小块）")

    # ====== 计算 embedding ======
    model = SentenceTransformer(E5_MODEL_NAME)
    texts = [d["text"] for d in docs]

    # e5 推荐：输入前加前缀，这里我们当成 "passage"
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