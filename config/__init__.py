# config/__init__.py
import json
import os
from .paths import PROJECT_ROOT

_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "courses.json")

# 1. 安全读取 JSON
try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _COURSE_DATA = json.load(f)
except FileNotFoundError:
    # 简单的容错，防止文件不存在时直接崩掉
    print(f"Warning: Configuration file not found at {_CONFIG_PATH}")
    _COURSE_DATA = {"courses": [], "global_context": {}}

# 2. 导出全局上下文 (供其他模块使用)
GLOBAL_CONTEXT = _COURSE_DATA.get("global_context", {})

# 3. 生成 COURSE_FILE_HINTS (供 RAG 路由使用)
# 结构: {"ECE-GY 6143": ["ml", "statistics"...]}
COURSE_FILE_HINTS = {
    c["code"]: c.get("keywords", [])
    for c in _COURSE_DATA.get("courses", [])
}

# 4. 生成 COURSE_PROFILES (供 Advisor 选课逻辑使用)
# 关键修复：这里手动构建 'focus' 字段，把 JSON 里的 description 和 blurb 拼起来
COURSE_PROFILES = []

for c in _COURSE_DATA.get("courses", []):
    # 动态合成 focus 文本，防止 KeyError
    # 逻辑：优先用 description，没有就用 keywords 拼凑
    desc = c.get("description", "")
    blurb = c.get("recommendation_blurb", "")
    
    # 拼成一个给大模型看的完整 focus 描述
    combined_focus = f"{desc} {blurb}".strip()
    
    if not combined_focus:
        # 最后的兜底，防止空字符串
        combined_focus = ", ".join(c.get("keywords", []))

    profile = {
        "code": c["code"],
        "name": c["name"],
        "focus": combined_focus,  # <--- 这样 Advisor 模块就不会报错了！
        "keywords": c.get("keywords", []), # 多传一些数据以备后用
        "id": c.get("id", "")
    }
    COURSE_PROFILES.append(profile)

# 5. 生成 ID 映射表 (供 extract_course_codes 使用)
# 自动生成: "6143" -> "ECE-GY 6143"
ID_TO_FULLCODE = {}
ALIAS_TO_FULLCODE = {}

for c in COURSE_PROFILES:
    full_code = c["code"] # ECE-GY 6143
    
    # 1. 数字 ID 映射 (取最后一段)
    # 假设 code 是 "ECE-GY 6143"，split后取 "6143"
    try:
        num_id = full_code.split()[-1] 
        ID_TO_FULLCODE[num_id] = full_code
    except IndexError:
        pass
    
    # 2. 关键词映射 (简单的别名支持)
    # 如果 keywords 里有像 "ml", "dl", "risc-v" 这种短词，也可以作为别名
    for kw in c.get("keywords", []):
        # 只把短的关键词当作别名（比如长度小于10），太长的句子就算了
        if len(kw) < 10:
            ALIAS_TO_FULLCODE[kw.upper()] = full_code

__all__ = [
    "COURSE_FILE_HINTS", 
    "COURSE_PROFILES", 
    "GLOBAL_CONTEXT",
    "ID_TO_FULLCODE",
    "ALIAS_TO_FULLCODE"
]