# qwen_client.py
import requests
from typing import List, Dict, Any
from config.paths import QWEN_API_URL, QWEN_MODEL_NAME

def call_qwen(messages: List[Dict[str, str]],
              max_tokens: int = 512,
              temperature: float = 0.2,
              top_p: float = 0.8) -> str:
    payload: Dict[str, Any] = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = requests.post(QWEN_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()