# app.py
import re
from itertools import count

from flask import Flask, request, jsonify, render_template



from schedule_module import try_generate_schedule_from_dialog
# ===== 你自己封装的模块 =====
from memory_module import reset_memories, add_memory_from_turn
from rag_module import (
    
    retrieve_context,
    LAST_RETRIEVAL_DEBUG,
    analyze_request_with_qwen,

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
    # ---------------------------------------------------------
    # 1) 获取用户输入
    # ---------------------------------------------------------
    if request.is_json:
        question = request.json.get("question", "")
    else:
        question = request.form.get("question", "")

    if not question or not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    try:
        # =========================================================
        # [REMOVED] 删除了原先在这里的 "优先尝试课表生成" 代码块
        # 现在完全依赖 LLM 的意图识别来触发
        # =========================================================

        answer = ""
        
        # ---------------------------------------------------------
        # 2) 智能分析 (The Analyzer)
        #    现在 analyze_request_with_qwen 会返回 "SCHEDULE" 意图
        # ---------------------------------------------------------
        question_intent, refined_query = analyze_request_with_qwen(question)
        
        print(f"[DEBUG] User Question: {question}")
        print(f"[DEBUG] Intent: {question_intent} | Query: {refined_query}")

        # ---------------------------------------------------------
        # 3) 意图路由 (Route Logic)
        # ---------------------------------------------------------
        
        # === [NEW] 新增：处理排课意图 ===
        if question_intent == "SCHEDULE":
            # 调用排课状态机/逻辑
            try:
                handled, payload = try_generate_schedule_from_dialog(
                    question=question,
                    answer="", 
                    request_obj=request
                )
            except Exception as e:
                print(f"[Schedule Error] {e}")
                handled, payload = False, None

            if handled:
                # A) 如果生成了 HTML 课表 -> 直接返回结果
                if payload["type"] == "html":
                    note = "Here is your updated schedule:"
                    if request.is_json:
                        return jsonify({
                            "answer": note,
                            "intent": question_intent,
                            "schedule_html": payload["html"],
                            "conflicts": payload.get("conflicts", [])
                        })
                    else:
                        return payload["html"]

                # B) 如果需要确认 (Confirmation) -> 直接返回询问
                if payload["type"] == "ask_for_confirmation":
                    return jsonify(payload)
            
            else:
                # 识别到了排课意图，但没提取到课程号（例如用户只说了 "Add course"）
                answer = "I understood you want to manage your schedule, but I need the course code. Please specify it (e.g., 'Add 6143')."

        # === 比较模式 ===
        elif question_intent == "COMPARISON":
            answer = answer_course_comparison_question(
                original_question=question,
                retrieval_question=refined_query
            )
        
        # === 选课模式 ===
        elif question_intent == "SELECTION":
            answer = answer_course_selection_question(
                original_question=question,
                retrieval_question=refined_query
            )

        # === 大纲详情 ===
        elif question_intent == "SYLLABUS":
            answer = call_qwen_with_rag(
                original_question=question,
                retrieval_question=refined_query
            )

        # === 闲聊模式 ===
        else: 
            answer = chat_with_memory_only(question)
            # 防御性替换
            answer = re.sub(
                r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b",
                "this course",
                answer,
            )

        # ---------------------------------------------------------
        # 4) 写入长短期记忆
        # ---------------------------------------------------------
        turn_id = next(TURN_COUNTER) 
        add_memory_from_turn(question, answer, source_turn=turn_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

    # ---------------------------------------------------------
    # 5) 返回响应 (AI Chat Response)
    # ---------------------------------------------------------
    if request.is_json:
        return jsonify({
            "answer": answer, 
            "intent": question_intent,
            "refined_query": refined_query
        })
    else:
        # HTML 简易返回
        return f"""
        <html>
        <head><meta charset="utf-8"><title>Answer</title></head>
        <body>
          <h3>Intent: <span style="color:blue">{question_intent}</span></h3>
          <p><b>Query:</b> {refined_query}</p>
          <p><b>Question:</b> {question}</p>
          <hr>
          <pre style="white-space: pre-wrap;">{answer}</pre>
          <br>
          <a href="/">Back to main page</a>
        </body>
        </html>
        """


# ============== RAG debug 接口 (保持不变) ==============

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

    intent, refined = analyze_request_with_qwen(question)
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