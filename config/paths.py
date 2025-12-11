# config/paths.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Memory
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory_store")
MEMORY_INDEX_PATH = os.path.join(MEMORY_DIR, "mem_index.faiss")
MEMORY_TEXTS_PATH = os.path.join(MEMORY_DIR, "memories.json")

# Syllabus vector store
INDEX_DIR = os.path.join(PROJECT_ROOT, "vector_store")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

# Models
E5_MODEL_NAME = "intfloat/multilingual-e5-large"
QWEN_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
QWEN_MAX_TOKENS = 2048