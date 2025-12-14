# schedule_module.py
"""
模块职责：
- detect_generate_schedule_intent(text)            -> bool
- extract_courses_from_text(text)                  -> List[{"code","name","meetings"}] or None
- try_generate_schedule_from_dialog(question, answer, request_obj)
    -> (handled: bool, payload: dict)
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from flask import render_template
from course_scheduler import build_schedule, format_conflicts, minutes_to_time_str, parse_meeting
from course_db import CourseDB

INTENT_KEYWORDS = [
    "生成课表", "生成时间表", "排课", "帮我排课", "帮我生成课表",
    "create schedule", "generate schedule", "timetable", "generate timetable"
]
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}-?GY?\s*\d{2,4}\b", re.I)
COURSE_SIMPLE_RE = re.compile(r"\b[A-Z]{2,6}\-?\d{2,4}\b", re.I)
LINE_PARSE_RE = re.compile(r"^\s*([^|,]+?)[\|,]\s*([^|,]+?)[\|,]\s*(.+)$")

# CourseDB singleton
_course_db_singleton = None
def get_course_db():
    global _course_db_singleton
    if _course_db_singleton is None:
        # Use the structured course list (config/courses.json by default)
        _course_db_singleton = CourseDB()
    return _course_db_singleton

def detect_generate_schedule_intent(text: str) -> bool:
    if not text:
        return False
    s = text.lower()
    for k in INTENT_KEYWORDS:
        if k.lower() in s:
            return True
    if re.search(r"(有没有.*冲突|帮我排.*课|把.*排成表)", text):
        return True
    # 若文本里包含两门及以上课程代码/管道分隔行，也视为排课意图（便于二次补充）
    code_hits = COURSE_CODE_RE.findall(text) or COURSE_SIMPLE_RE.findall(text)
    if len(set(code_hits)) >= 2:
        return True
    if "|" in text:
        for line in text.splitlines():
            if LINE_PARSE_RE.match(line):
                return True
    return False

def _line_to_course(line: str) -> Optional[Dict[str, str]]:
    line = line.strip()
    if not line:
        return None
    m = LINE_PARSE_RE.match(line)
    if m:
        return {"code": m.group(1).strip(), "name": m.group(2).strip(), "meetings": m.group(3).strip()}
    parts = re.split(r"\s+-\s+|\s+\|\s+|,", line)
    if len(parts) >= 2:
        return {"code": parts[0].strip(), "name": parts[1].strip(), "meetings": parts[2].strip() if len(parts) >= 3 else ""}
    c = COURSE_CODE_RE.search(line) or COURSE_SIMPLE_RE.search(line)
    if c:
        cc = c.group(0).strip()
        return {"code": cc, "name": cc, "meetings": ""}
    return None

def extract_courses_from_text(text: str) -> Optional[List[Dict[str, str]]]:
    if not text:
        return None
    # 将单行里用“和/and”并列的课程拆成多行，便于逐行解析
    raw_lines = text.splitlines()
    lines: List[str] = []
    for ln in raw_lines:
        if re.search(COURSE_CODE_RE, ln) and re.search(r"\s+(和|and)\s+", ln, re.I):
            parts = re.split(r"\s+(?:和|and)\s+", ln, flags=re.I)
            lines.extend([p for p in parts if p.strip()])
        else:
            lines.append(ln)
    candidates = []
    for line in lines:
        # 如果一行里出现多个课程号，拆成多个候选
        multi_codes = COURSE_CODE_RE.findall(line) or COURSE_SIMPLE_RE.findall(line)
        multi_codes = [c.strip() for c in multi_codes if c.strip()]
        if len(set(multi_codes)) >= 2:
            for mc in multi_codes:
                candidates.append({"code": mc, "name": mc, "meetings": ""})
            continue
        parsed = _line_to_course(line)
        if parsed:
            candidates.append(parsed)
    if candidates:
        return candidates
    codes = COURSE_CODE_RE.findall(text) or COURSE_SIMPLE_RE.findall(text)
    if codes:
        seen = set(); out = []
        for c in codes:
            cc = c.strip()
            if cc not in seen:
                seen.add(cc); out.append({"code": cc, "name": cc, "meetings": ""})
        return out
    m = re.search(r"课程[:：\s]+(.+)", text)
    if m:
        tail = m.group(1)
        parts = re.split(r"[，,;；\n]", tail)
        out = []
        for p in parts:
            p = p.strip()
            if not p: continue
            parsed = _line_to_course(p) or {"code": p, "name": p, "meetings": ""}
            out.append(parsed)
        if out:
            return out
    return None

def generate_schedule_html_from_courses(courses: List[Dict[str, str]], day_start=8*60, day_end=20*60, slot_minutes=30) -> Tuple[str, List[Dict]]:
    grid, conflicts = build_schedule(courses, day_start=day_start, day_end=day_end, slot_minutes=slot_minutes)
    n_slots = len(next(iter(grid.values())))
    # 显示用映射：code -> "code name"
    display_map = {}
    course_entries = []
    for c in courses:
        code = str(c.get("code", "")).strip()
        name = str(c.get("name", "")).strip()
        if code and name and name.lower() != code.lower():
            display = f"{code} {name}"
        elif code:
            display = code
        else:
            display = name
        display_map[code or name] = display
        course_entries.append({
            "code": code,
            "name": name,
            "meetings": c.get("meetings", ""),
            "room": c.get("room", "")
        })

    # 先记录课程起止 slot 与时间文本
    slot_range_map = {}
    label_slots = set()
    for c in courses:
        code_key = str(c.get("code", c.get("name", ""))).strip()
        meetings = parse_meeting(str(c.get("meetings", "") or ""))
        for day_idx, smin, emin in meetings:
            start_slot = max(0, (smin - day_start) // slot_minutes)
            end_slot = min(n_slots, max(0, math.ceil((emin - day_start) / slot_minutes)))
            slot_range_map[(code_key, day_idx, start_slot)] = (
                minutes_to_time_str(smin),
                minutes_to_time_str(emin),
                end_slot,
            )
            label_slots.add(start_slot)
            label_slots.add(end_slot)

    # 再计算 rowspan，尽量让同一课程时间段合并到一个单元格
    cell_info = {d: [None for _ in range(n_slots)] for d in range(7)}
    for d in range(7):
        slot = 0
        while slot < n_slots:
            cell_str = grid[d][slot][0]
            if not cell_str:
                cell_info[d][slot] = {"html": "", "multiple": False, "rowspan": 1}
                slot += 1
                continue
            span = 1
            while slot + span < n_slots and grid[d][slot + span][0] == cell_str:
                span += 1
            parts = [p.strip() for p in cell_str.split(",") if p.strip()]
            display_parts = [display_map.get(p, p) for p in parts]
            html = "<br>".join(display_parts)
            start_str = end_str = ""
            span_override = None
            # 尝试匹配课程开始/结束时间（按 code + day + start_slot）
            if parts:
                key = (parts[0], d, slot)
                if key in slot_range_map:
                    start_str, end_str, end_slot = slot_range_map[key]
                    span_override = max(1, end_slot - slot)
            use_span = span_override if span_override else span
            cell_info[d][slot] = {
                "html": html,
                "multiple": len(parts) > 1,
                "rowspan": use_span,
                "start_line": True,
                "end_line": True,
                "start_time": start_str,
                "end_time": end_str,
            }
            for s in range(1, use_span):
                cell_info[d][slot + s] = {"skip": True}
            slot += use_span

    rows = []
    for slot in range(n_slots):
        t0 = day_start + slot * slot_minutes
        t_str = minutes_to_time_str(t0)
        # 显示完整时间轴
        time_label = t_str
        cells = [cell_info[d][slot] for d in range(7)]
        row = {"time": t_str, "time_label": time_label, "cells": cells}
        rows.append(row)
    html = render_template(
        "schedule.html",
        rows=rows,
        day_names=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        conflicts=conflicts,
        course_entries=course_entries,
        slot_minutes=slot_minutes
    )
    return html, conflicts

def try_generate_schedule_from_dialog(question: str, answer: str, request_obj) -> Tuple[bool, Dict]:
    """
    主入口：
    - 尝试检测意图
    - 抽取候选课程
    - 若 meetings 为空则用 CourseDB 补全
    - 若补全完毕则直接生成 HTML 并返回
    - 否则返回 candidates 让前端确认/补全
    """
    combined = (question or "") + "\n" + (answer or "")
    if not detect_generate_schedule_intent(combined):
        return False, {}

    candidates = extract_courses_from_text(answer) or extract_courses_from_text(question)
    if candidates:
        # 用 DB 补全 meetings/room
        db = get_course_db()
        for c in candidates:
            if not c.get("meetings"):
                info = db.find_course_info(c.get("code", "") or c.get("name", ""))
                if info and info.get("meetings"):
                    c["meetings"] = "; ".join(info.get("meetings"))
                if not c.get("room") and info.get("rooms"):
                    c["room"] = "; ".join(info.get("rooms"))

        need_meetings = any(not c.get("meetings") for c in candidates)
        if not need_meetings:
            html, conflicts = generate_schedule_html_from_courses(candidates)
            return True, {"type": "html", "html": html, "conflicts": conflicts}
        else:
            return True, {"type": "ask_for_confirmation", "message": "检测到以下候选课程，请确认并补全上课时间后生成课表。", "candidates": candidates}

    msg = "我检测到你想生成课表，但没从你的问题/回答中识别出具体课程。请以每行 `CODE|NAME` 的格式提供课程，系统会尝试从已收集的 syllabus 文本中自动补全时间与地点。"
    return True, {"type": "ask_for_confirmation", "message": msg, "candidates": []}
