# app.py
import re
from itertools import count

from flask import Flask, request, jsonify, render_template

# ===== 你自己封装的模块 =====
from memory_module import reset_memories, add_memory_from_turn
from rag_module import (
    refine_question_with_qwen,
    is_course_selection_question,
    is_syllabus_question,
    is_course_comparison_question,
    retrieve_context,
    LAST_RETRIEVAL_DEBUG,
)
from advisor_module import (
    chat_with_memory_only,
    call_qwen_with_rag,
    answer_course_selection_question,
    answer_course_comparison_question,
)

# 如果你把 Qwen HTTP 调用封到 qwen_client 里，这里不用再管 URL/模型名
# from qwen_client import call_qwen  # 现在 app.py 不直接用 Qwen 了

app = Flask(__name__)

TURN_COUNTER = count(1)  # 1,2,3,...


# ============== Web 页 + 重置记忆 ==============

@app.route("/", methods=["GET"])
def index():
    """
    打开主页面时顺带清空记忆（你之前说“刷新就初始化”）。
    如果以后不想自动清空，把 reset_memories() 注释掉即可。
    """
    reset_memories()
    return render_template("index.html")


@app.route("/reset_memory", methods=["POST"])
def reset_memory_route():
    """
    提供一个显式重置记忆的接口，前端按钮可以调用这个。
    """
    reset_memories()
    return jsonify({"status": "ok"})


# ============== 主问答接口 ==============

@app.route("/ask", methods=["POST"])
def ask():
    # 1) 取问题
    if request.is_json:
        question = request.json.get("question", "")
    else:
        question = request.form.get("question", "")

    if not question or not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    try:
        refined = None

        # 2) 意图路由：比较 / 选课 / syllabus / 普通聊天
        if is_course_comparison_question(question):
            # A vs B / A 还是 B：课程对比，双 syllabus + memory
            refined = refine_question_with_qwen(question)
            answer = answer_course_comparison_question(
                original_question=question,
                retrieval_question=refined,
            )

        elif is_course_selection_question(question):
            # 普通选课咨询：在 COURSE_PROFILES 里挑一门 + syllabus + memory
            refined = refine_question_with_qwen(question)
            answer = answer_course_selection_question(
                original_question=question,
                retrieval_question=refined,
            )

        elif is_syllabus_question(question):
            # 课程细节 / syllabus 问题：syllabus RAG + memory
            refined = refine_question_with_qwen(question)
            answer = call_qwen_with_rag(
                original_question=question,
                retrieval_question=refined,
            )

        else:
            # 其它任何聊天 / 规划 / 生活问题：纯记忆聊天，不查 syllabus
            answer = chat_with_memory_only(question)
            # 保守点，防止模型乱提课号
            answer = re.sub(
                r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b",
                "this course",
                answer,
            )

        # 3) 每一轮都写入记忆（无论是 syllabus 还是纯聊天）
        turn_id = next(TURN_COUNTER)
        add_memory_from_turn(question, answer, source_turn=turn_id)

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {e}"}), 500

    # 4) 返回格式：JSON 模式 / 表单模式
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


# ============== RAG debug 接口 ==============

@app.route("/debug_retrieval", methods=["POST"])
def debug_retrieval():
    """
    调试 syllabus 检索结果：
    - 返回 refine 后的 query
    - 返回 retrieve_context 的上下文
    - 返回 rag_module 里的 LAST_RETRIEVAL_DEBUG
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    refined = refine_question_with_qwen(question)
    contexts = retrieve_context(
        question=refined,
        analysis_question=question,
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