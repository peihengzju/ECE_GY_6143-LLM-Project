import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

# Default to the structured course list in config/courses.json
DEFAULT_COURSE_PATH = Path(__file__).resolve().parent / "config" / "courses.json"


class CourseDB:
    def __init__(self, course_json_path: str | Path | None = None):
        """
        Lightweight in-memory course catalog.
        Automatically loads `config/courses.json` unless a path is provided.
        """
        self.path = Path(course_json_path) if course_json_path else DEFAULT_COURSE_PATH
        self.courses_map: Dict[str, Dict] = {}
        self._load_data()

    def _load_data(self):
        """Load structured course data and build an index."""
        if not self.path.exists():
            print(f"[CourseDB] Error: File not found at {self.path}")
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[CourseDB] Failed to parse JSON at {self.path}: {e}")
            return

        # Accept a few common layouts:
        # 1) {"courses": [...]} (what config/courses.json uses)
        # 2) [...]              (list of course dicts)
        # 3) {"ECE-GY 6143": {...}, ...} (code keyed map)
        if isinstance(data, dict):
            if "courses" in data:
                course_list = data.get("courses", [])
            elif all(isinstance(v, dict) for v in data.values()):
                course_list = [{"code": k, **v} for k, v in data.items()]
            else:
                print(f"[CourseDB] Error: unexpected JSON dict layout in {self.path}")
                return
        elif isinstance(data, list):
            course_list = data
        else:
            print(f"[CourseDB] Error: unexpected JSON root type: {type(data)}")
            return

        # Flatten list-of-lists if present
        if course_list and isinstance(course_list[0], list):
            course_list = [item for sub in course_list for item in (sub if isinstance(sub, list) else [sub])]

        print(f"[CourseDB] Loading {len(course_list)} courses from {self.path}...")

        loaded = 0
        self.courses_map.clear()

        for course in course_list:
            if not isinstance(course, dict):
                continue

            raw_code = (course.get("code") or "").strip()
            # Some generators may only provide code inside meta fields
            if not raw_code and isinstance(course.get("meta"), dict):
                raw_code = (course["meta"].get("course") or "").strip()
            if not raw_code:
                continue

            norm_code = self._normalize_code(raw_code)
            if not norm_code:
                continue

            # Normalize repeated fields to unique lists
            meetings = self._normalize_list(course.get("meetings") or course.get("schedule"))
            rooms = self._normalize_list(course.get("rooms") or course.get("location"))
            instructors = self._normalize_list(course.get("instructors") or course.get("instructor"))

            self.courses_map[norm_code] = {
                "code": raw_code,
                "name": (course.get("name") or "").strip(),
                "meetings": meetings,
                "rooms": rooms,
                "instructors": instructors,
                "description": (course.get("description") or "").strip(),
            }
            loaded += 1

        print(f"[CourseDB] Successfully loaded {loaded} courses. Index size={len(self.courses_map)}")

    def _normalize_list(self, value: Iterable | None) -> List[str]:
        """Coerce a scalar or iterable into a de-duplicated list of stripped strings."""
        if value is None:
            return []
        if isinstance(value, str):
            value_list = [value]
        elif isinstance(value, Iterable):
            value_list = list(value)
        else:
            return []
        cleaned = []
        seen = set()
        for item in value_list:
            s = str(item).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        return cleaned

    def _normalize_code(self, code: str) -> str:
        """
        Normalize a course code so different spellings map to the same key.
        Examples:
        - "ECE 6143"
        - "ECE-GY 6143"
        - "ECEGY6143"
        - "EL5613"
        Returns "ECE-GY6143" or the pure number ("6143") when only digits are given.
        """
        if not code:
            return ""

        s = code.upper().strip()

        # Fast path: pure number queries like "6143"
        if re.fullmatch(r"\d{3,5}", s):
            return s

        # Keep only letters and digits, drop separators like spaces/hyphens
        s = re.sub(r"[^A-Z0-9]", "", s)

        # Extract dept + optional GY + number
        m = re.match(r"^([A-Z]{2,6})(?:GY)?(\d{3,5})", s)
        if not m:
            return ""

        dept, num = m.group(1), m.group(2)
        return f"{dept}-GY{num}"

    def find_course_info(self, query_code: str) -> Dict:
        """
        Retrieve a course by code or partial numeric id.
        Supports:
        - Full code: "ECE-GY 6143"
        - Compact code: "ECEGY6143"
        - Missing GY: "ECE 6143", "EL5613"
        - Numbers only: "6143"
        """
        norm_query = self._normalize_code(query_code)

        if not norm_query:
            return self._unknown(query_code)

        # Exact lookup
        if norm_query in self.courses_map:
            return self.courses_map[norm_query]

        # Numeric fuzzy lookup: "6143" should match "*6143"
        if re.fullmatch(r"\d{3,5}", norm_query):
            hits = [info for db_code, info in self.courses_map.items() if db_code.endswith(norm_query)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                candidates = [c["code"] for c in hits[:10]]
                print(f"[CourseDB] Warning: multiple matches for {query_code} -> {candidates} (showing up to 10)")
                return hits[0]

        # Not found
        return self._unknown(query_code)

    def _unknown(self, query_code: str) -> Dict:
        return {
            "code": query_code,
            "name": "Unknown Course",
            "meetings": [],
            "rooms": [],
            "instructors": [],
            "description": ""
        }
