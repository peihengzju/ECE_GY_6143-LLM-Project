# course_scheduler.py
"""
课程表生成器与冲突检测器

接口说明：
- build_schedule(courses, day_start=8*60, day_end=20*60, slot_minutes=30)
    -> (grid, conflicts)
    grid: dict day_index -> list of slot cells，每个 cell 为字符串（课程 code 或 "" 或 "A,B" 表示冲突）
    conflicts: list of dict {course_a, course_b, day, overlap_start, overlap_end}

- minutes_to_time_str(minutes) -> "HH:MM"
- format_conflicts(conflicts) -> str （可用于文本输出）
"""

from typing import List, Dict, Tuple, Any
import re
import math

# 星期映射（支持常见中英文/缩写）
DAY_MAP = {
    "mon": 0, "monday": 0, "周一": 0, "一": 0, "mon.": 0,
    "tue": 1, "tues": 1, "tuesday": 1, "周二": 1, "二": 1,
    "wed": 2, "wednesday": 2, "周三": 2, "三": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3, "周四": 3, "四": 3,
    "fri": 4, "friday": 4, "周五": 4, "五": 4,
    "sat": 5, "saturday": 5, "周六": 5, "六": 5,
    "sun": 6, "sunday": 6, "周日": 6, "周天": 6, "日": 6, "七": 6,
}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 时间字符串转换
def time_str_to_minutes(t: str) -> int:
    t = t.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if not m:
        raise ValueError(f"非法时间格式: '{t}'")
    h = int(m.group(1)); mm = int(m.group(2))
    return h * 60 + mm

def minutes_to_time_str(m: int) -> str:
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

# 解析单个 meeting 字符串到 (day_index, start_min, end_min) 列表
def split_days_token(token: str) -> List[str]:
    token = token.strip()
    if not token:
        return []
    for sep in ["/", ",", "、", "&", ";", "|"]:
        if sep in token:
            return [p.strip() for p in token.split(sep) if p.strip()]
    # 中文周或单字
    if "周" in token or any(ch in token for ch in ["一","二","三","四","五","六","日","天"]):
        parts = re.findall(r"周[一二三四五六日天]|[一二三四五六日天]", token)
        return parts
    # 英文连写：按大写单词切分
    parts = re.findall(r"[A-Za-z]{2,6}\.?", token)
    if parts:
        return parts
    # fallback
    return [token]

def parse_meeting(meeting_str: str) -> List[Tuple[int,int,int]]:
    """
    支持格式示例：
      "Mon/Wed 09:00-10:15"
      "Tue 9:30-11:00; Thu 9:30-11:00"
      "周一 09:00-10:15; 周三 11:00-12:15"
      "Tuesdays 11:00-13:30"
    返回列表 (day_index, start_min, end_min)
    """
    results = []
    # split by ';' or '|' segments
    parts = [p.strip() for p in re.split(r"[;|]", meeting_str) if p.strip()]
    time_re = re.compile(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})")
    for part in parts:
        m = time_re.search(part)
        if not m:
            continue
        start_s, end_s = m.group(1), m.group(2)
        start_min = time_str_to_minutes(start_s)
        end_min = time_str_to_minutes(end_s)
        # day token is the text before time or after time
        day_token = part[:m.start()].strip() or part[m.end():].strip()
        if not day_token:
            continue
        day_items = split_days_token(day_token)
        for d in day_items:
            key = d.strip().lower().rstrip('.')
            mapped = None
            if key in DAY_MAP:
                mapped = DAY_MAP[key]
            else:
                key2 = re.sub(r"[^a-z]", "", key)
                if key2 in DAY_MAP:
                    mapped = DAY_MAP[key2]
                else:
                    k3 = d.strip()
                    if k3 in DAY_MAP:
                        mapped = DAY_MAP[k3]
            if mapped is None:
                k4 = d.strip().lower().replace(".", "")
                if k4 in DAY_MAP:
                    mapped = DAY_MAP[k4]
            if mapped is None:
                # 尝试英文首字母识别，例如 "tuesday" -> Tue
                klow = d.strip().lower()
                for name, idx in DAY_MAP.items():
                    if klow.startswith(name):
                        mapped = idx
                        break
            if mapped is None:
                continue
            results.append((mapped, start_min, end_min))
    return results

# 主函数：生成 grid 与冲突列表
def build_schedule(courses: List[Dict[str,Any]],
                   day_start: int = 8*60,
                   day_end: int = 20*60,
                   slot_minutes: int = 30) -> Tuple[Dict[int, List[List[str]]], List[Dict[str,Any]]]:
    """
    courses: 每项 dict 包含至少 {"code": "...", "meetings": "Mon 09:00-10:15; Wed 09:00-10:15", "name": "..."}。
    返回:
      grid: { day_index: [ [cell_str], [cell_str], ... ] }  每格目前放单槽字符串（空或课程code或 "A,B"）
      conflicts: [{course_a, course_b, day, overlap_start, overlap_end}, ...]
    """
    n_slots = math.ceil((day_end - day_start) / slot_minutes)
    # 初始化 grid：每格为列表 [cell_string]，保持兼容前端模板
    grid = {d: [[ "" ] for _ in range(n_slots)] for d in range(7)}

    meetings = []  # list of tuples (course_code, day, start_min, end_min)

    for c in courses:
        code = str(c.get("code", c.get("name", ""))).strip()
        meetings_str = str(c.get("meetings", "") or "").strip()
        if not meetings_str:
            # 如果没有 meetings 字段，跳过；上层应保证已补全
            continue
        items = parse_meeting(meetings_str)
        for (day, smin, emin) in items:
            if emin <= smin:
                continue
            meetings.append((code, day, smin, emin))

    # 按天组织
    conflicts = []
    meetings_by_day = {d: [] for d in range(7)}
    for m in meetings:
        meetings_by_day[m[1]].append(m)

    for d in range(7):
        ms = sorted(meetings_by_day[d], key=lambda x: x[2])
        # 填 grid
        for i, (code, day, smin, emin) in enumerate(ms):
            start_slot = max(0, (smin - day_start) // slot_minutes)
            end_slot = min(n_slots, math.ceil((emin - day_start) / slot_minutes))
            for slot in range(start_slot, end_slot):
                cur = grid[day][slot][0]
                if cur == "":
                    grid[day][slot][0] = code
                else:
                    existing = grid[day][slot][0]
                    # 避免重复添加同一个 code
                    parts = [p.strip() for p in existing.split(",") if p.strip()]
                    if code not in parts:
                        parts.append(code)
                        grid[day][slot][0] = ",".join(parts)
            # 检测冲突（与后续会议）
            for j in range(i+1, len(ms)):
                other = ms[j]
                _, _, os, oe = other
                if os < emin:
                    overlap_start = max(smin, os)
                    overlap_end = min(emin, oe)
                    conflicts.append({
                        "course_a": code,
                        "course_b": other[0],
                        "day": d,
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end
                    })
                else:
                    break

    return grid, conflicts

# 格式化冲突为可读文本
def format_conflicts(conflicts: List[Dict[str,Any]]) -> str:
    if not conflicts:
        return "没有检测到冲突。"
    lines = []
    for c in conflicts:
        dayname = DAY_NAMES[c["day"]]
        s = minutes_to_time_str(c["overlap_start"])
        e = minutes_to_time_str(c["overlap_end"])
        lines.append(f"{c['course_a']} 与 {c['course_b']} 在 {dayname} {s}-{e} 冲突")
    return "\n".join(lines)

# 方便命令行测试（可选）
if __name__ == "__main__":
    sample_courses = [
        {"code": "ECE-GY6143", "name": "Advanced ML", "meetings": "Tue 11:00-13:30; Thu 11:00-13:30"},
        {"code": "MATH200", "name": "Linear Algebra", "meetings": "Tue 12:30-14:00"},
        {"code": "PHYS1", "name": "Physics", "meetings": "Wed 09:00-11:00"}
    ]
    g, conf = build_schedule(sample_courses)
    print("Grid sample (Tue slots):")
    print(g[1][:10])
    print("Conflicts:")
    print(format_conflicts(conf))
