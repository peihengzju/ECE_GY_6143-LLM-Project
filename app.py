# app.py
import os
import json
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from flask import Flask, request, jsonify

# ====== RAG 部分配置 ======
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

E5_MODEL_NAME = "intfloat/multilingual-e5-base"

# 检索参数（比之前更细一些）
TOP_K = 10                 # 最终送给大模型的 chunk 数量
FAISS_RAW_K = 24           # 先从 Faiss 拿更多候选，再做二次排序
MAX_CONTEXT_CHARS = 3600   # 拼接给大模型的 syllabus 总长度上限

# ====== Qwen + vLLM OpenAI 接口配置 ======
QWEN_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
QWEN_MAX_TOKENS = 512      # 输出长度，适当加一点，但不要太夸张


# ====== 加载向量索引和文本 ======
print("[RAG] Loading Faiss index and texts ...")
faiss_index = faiss.read_index(INDEX_PATH)
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    DOCS: List[Dict] = json.load(f)
print(f"[RAG] Loaded {len(DOCS)} chunks")

# 加载 embedding 模型（和 ingest_syllabi.py 保持一致）
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


def _keyword_boost(question: str, text: str) -> int:
    """
    根据问题和 chunk 内容的关键词做一点启发式加权：
    - 问“期末、考试、占比、grading”等，就优先那些包含 exam/grade/percentage 的段落。
    不追求很聪明，只要能把明显的表格和 grading policy 拎出来就行。
    """
    q = question.lower()
    t = text.lower()

    boost = 0

    # 问考试 / 期末 / quiz 之类
    if any(k in q for k in ["期末", "期中", "考试", "quiz", "exam", "midterm", "final"]):
        if any(k in t for k in ["exam", "midterm", "final", "quiz", "test", "exam.", "exams"]):
            boost += 3

    # 问成绩占比 / grading / 权重
    if any(k in q for k in ["占比", "成绩", "评分", "grading", "grade", "weight", "percentage", "%"]):
        if any(k in t for k in ["grading", "grade", "weight", "percentage", "%", "assessment", "evaluation"]):
            boost += 3

    # 问作业、project
    if any(k in q for k in ["作业", "homework", "assignment", "project"]):
        if any(k in t for k in ["homework", "assignment", "project", "lab", "problem set"]):
            boost += 2

    # 问出勤 / participation
    if any(k in q for k in ["出勤", "attendance", "participation"]):
        if any(k in t for k in ["attendance", "participation"]):
            boost += 2

    return boost


def retrieve_context(question: str,
                     top_k: int = TOP_K,
                     faiss_raw_k: int = FAISS_RAW_K) -> List[str]:
    """
    从向量库里检索最相关的若干 chunk：
    1）先从 Faiss 拿 faiss_raw_k 个候选；
    2）根据问题关键词做一个简单 boost；
    3）排序后取前 top_k；
    4）控制总长度，避免把 Qwen 上下文撑爆。
    """
    q_emb = embed_query(question)
    # 从 Faiss 拿更多候选，给后面做二次筛选用
    scores, idx = faiss_index.search(q_emb, faiss_raw_k)

    candidates = []
    for rank, i in enumerate(idx[0]):
        doc = DOCS[int(i)]
        text = doc["text"]
        meta = doc.get("meta", {})
        fname = meta.get("file", "unknown")
        page = meta.get("page", 0) + 1  # 页码从 1 开始

        header = f"[{fname} | page {page}]"
        full_text = f"{header}\n{text}"

        boost = _keyword_boost(question, text)

        # base_rank 越小说明 Faiss 越认为相似；这里用 (rank, -boost) 控制一下顺序基准
        # 实际排序：先看 boost（倒序），再看原 rank（正序）
        candidates.append({
            "rank": rank,
            "boost": boost,
            "text": full_text,
        })

    # 根据 boost + 原始 rank 排序
    candidates.sort(key=lambda x: (-x["boost"], x["rank"]))

    # 取前 top_k，并做长度裁剪
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
    """用 RAG + Qwen 回答问题。"""
    contexts = retrieve_context(question)

    context_block = "\n\n".join(contexts) if contexts else "（未检索到相关 syllabus 片段）"

    system_prompt = (
        "你是一名熟悉 NYU Tandon 课程的助教，任务是根据提供的 syllabus 片段，"
        "用简体中文回答学生的问题。\n"
        "必须严格以给出的 syllabus 为依据：\n"
        "1）尽量给出文中出现的数字、百分比、日期、权重等具体细节；\n"
        "2）如果 syllabus 中没有写明，就明确说“syllabus 里没写”或“从当前片段看不出来”；\n"
        "3）不要自己编造信息，比如不要凭空发明考试比例或作业次数；\n"
        "4）可以帮学生总结、对比课程、给出选课建议，但前提是基于 syllabus 内容；\n"
        "5）回答里尽量引用原文要点，并标出对应的 [file | page]。"
    )

    user_prompt = (
        "下面是若干门 NYU 课程 syllabus 的原文片段（可能来自不同课程或不同页面）：\n\n"
        f"{context_block}\n\n"
        "现在请你基于上面的 syllabus 内容，回答我的问题。\n"
        "我的问题是：\n"
        f"{question}\n\n"
        "要求：\n"
        "1）如果 syllabus 里对这个问题有明确说明（例如：期末考试占比 30%、作业占比 40%、是否允许带 cheat sheet 等），"
        "   请直接给出具体数字和原文关键信息，并说明来自哪门课、哪一页（用前面的 [file | page] 标记即可）。\n"
        "2）如果 syllabus 只有部分信息（比如只写了有期末考试，但没写比例），就如实说明“只提到了有期末考试，但没有看到具体占比”。\n"
        "3）如果完全没找到相关信息，一定要说“从目前看到的 syllabus 片段里，没有找到相关信息”。\n"
        "4）如果我继续问更细节（比如具体作业数量、每次 quiz 形式），你要继续在这些 syllabus 片段里帮我挖细节，而不是瞎编。"
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": QWEN_MAX_TOKENS,
        "temperature": 0.2,  # 降一点，减少瞎编
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
    # 简单表单版本
    return """
    <html>
    <head><meta charset="utf-8"><title>NYU Syllabus RAG</title></head>
    <body>
      <h2>NYU Syllabus 智能问答 (RAG + Qwen)</h2>
      <form action="/ask" method="post">
        <textarea name="question" rows="5" cols="80"
          placeholder="比如：ECE-GY 6143 这门课的期末考试占比多少？&#10;或者：我想走机器学习方向，该优先选哪些课？"></textarea><br>
        <button type="submit">提问</button>
      </form>
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
        return jsonify({"error": "问题不能为空"}), 400

    try:
        answer = call_qwen_with_rag(question)
    except Exception as e:
        return jsonify({"error": f"调用 Qwen 失败: {e}"}), 500

    # 如果是前端 ajax，可以直接返回 json
    if request.is_json:
        return jsonify({"answer": answer})
    # 如果是表单提交，就返回简单的 HTML
    else:
        return f"""
        <html>
        <head><meta charset="utf-8"><title>回答</title></head>
        <body>
          <p><b>问题：</b>{question}</p>
          <hr>
          <pre>{answer}</pre>
          <a href="/">返回继续提问</a>
        </body>
        </html>
        """


# 一个调试接口：看一下这次检索到了哪些 chunk（可选）
@app.route("/debug_retrieval", methods=["POST"])
def debug_retrieval():
    """传入 question，返回这次用到的 syllabus 片段，方便你自己检查 RAG 效果。"""
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "问题不能为空"}), 400

    contexts = retrieve_context(question)
    return jsonify({
        "question": question,
        "num_contexts": len(contexts),
        "contexts": contexts,
    })


if __name__ == "__main__":
    print("RAG + Qwen server is running.")
    app.run(host="0.0.0.0", port=5000, debug=True)
