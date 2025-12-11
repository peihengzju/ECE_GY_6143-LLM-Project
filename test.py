import importlib
import sys
import types

import pytest


@pytest.fixture(scope="module")
def stubbed_app():
    """
    Provide the Flask app with stubbed dependencies so tests do not call real models or IO.
    """
    mem_log = {"added": []}

    mem_mod = types.ModuleType("memory_module")
    mem_mod.reset_memories = lambda: None

    def add_memory_from_turn(question, answer, source_turn):
        mem_log["added"].append((question, answer, source_turn))

    mem_mod.add_memory_from_turn = add_memory_from_turn

    rag_mod = types.ModuleType("rag_module")
    rag_mod.LAST_RETRIEVAL_DEBUG = {}
    rag_mod.refine_question_with_qwen = lambda q: f"refined:{q}"
    rag_mod.is_course_comparison_question = lambda q: "compare" in q.lower()
    rag_mod.is_course_selection_question = lambda q: "select" in q.lower()
    rag_mod.is_syllabus_question = lambda q: "syllabus" in q.lower()

    def retrieve_context(question, analysis_question=None, forced_course=None):
        rag_mod.LAST_RETRIEVAL_DEBUG = {
            "question": question,
            "analysis_question": analysis_question,
            "forced_course": forced_course,
        }
        return ["ctx"] if question else []

    rag_mod.retrieve_context = retrieve_context

    advisor_mod = types.ModuleType("advisor_module")
    advisor_mod.chat_with_memory_only = lambda q: "memory chat"
    advisor_mod.call_qwen_with_rag = (
        lambda original_question, retrieval_question=None: "rag answer"
    )
    advisor_mod.answer_course_selection_question = (
        lambda original_question, retrieval_question=None: "selection answer"
    )
    advisor_mod.answer_course_comparison_question = (
        lambda original_question, retrieval_question=None: "comparison answer"
    )

    sys.modules.update(
        {
            "memory_module": mem_mod,
            "rag_module": rag_mod,
            "advisor_module": advisor_mod,
        }
    )

    app_mod = importlib.import_module("app")
    yield app_mod, mem_log, rag_mod

    for name in ["memory_module", "rag_module", "advisor_module"]:
        sys.modules.pop(name, None)


def test_ask_empty_question(stubbed_app):
    app_mod, mem_log, _ = stubbed_app
    client = app_mod.app.test_client()

    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Question must not be empty."
    assert not mem_log["added"]


def test_ask_selection_path(stubbed_app):
    app_mod, mem_log, _ = stubbed_app
    mem_log["added"].clear()
    client = app_mod.app.test_client()

    resp = client.post("/ask", json={"question": "please SELECT a course"})
    assert resp.status_code == 200
    assert resp.get_json()["answer"] == "selection answer"
    assert mem_log["added"]


def test_ask_comparison_path(stubbed_app):
    app_mod, mem_log, _ = stubbed_app
    mem_log["added"].clear()
    client = app_mod.app.test_client()

    resp = client.post("/ask", json={"question": "compare two courses"})
    assert resp.status_code == 200
    assert resp.get_json()["answer"] == "comparison answer"
    assert mem_log["added"]


def test_ask_syllabus_path(stubbed_app):
    app_mod, mem_log, _ = stubbed_app
    mem_log["added"].clear()
    client = app_mod.app.test_client()

    resp = client.post("/ask", json={"question": "syllabus question"})
    assert resp.status_code == 200
    assert resp.get_json()["answer"] == "rag answer"
    assert mem_log["added"]


def test_ask_memory_only_path(stubbed_app):
    app_mod, mem_log, _ = stubbed_app
    mem_log["added"].clear()
    client = app_mod.app.test_client()

    resp = client.post("/ask", json={"question": "just chat"})
    assert resp.status_code == 200
    assert resp.get_json()["answer"] == "memory chat"
    assert mem_log["added"]


def test_debug_retrieval(stubbed_app):
    app_mod, _, rag_mod = stubbed_app
    client = app_mod.app.test_client()

    resp = client.post("/debug_retrieval", json={"question": "check syllabus"})
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["refined"].startswith("refined:")
    assert data["num_contexts"] == len(data["contexts"])
    assert data["debug"]["question"] == "refined:check syllabus"
    assert rag_mod.LAST_RETRIEVAL_DEBUG["question"] == "refined:check syllabus"
