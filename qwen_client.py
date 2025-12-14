# qwen_client.py
import json
import requests
from typing import List, Dict, Any, Optional
from config.paths import QWEN_API_URL, QWEN_MODEL_NAME


def _safe_preview(text: str, n: int = 400) -> str:
    text = text or ""
    text = text.replace("\n", "\\n")
    return text[:n]


def call_qwen(
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> str:
    """
    Robust wrapper for calling Qwen backend.
    Guarantees a non-empty return string for easier debugging / UI rendering.
    """
    

    payload: Dict[str, Any] = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "stream": False,

        # Some backends expect top-level params (OpenAI-like)
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,

        # Some backends (Ollama-like) accept options
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    try:
        # Split timeout: (connect_timeout, read_timeout)
        resp = requests.post(QWEN_API_URL, json=payload, timeout=(10, 120))
        resp.raise_for_status()

        # Sometimes servers return non-json on error even with 200
        try:
            data = resp.json()
        except Exception:
            preview = _safe_preview(resp.text)
            return f"Error: Model returned non-JSON response. Preview: {preview}"
        

        # 1) Ollama /api/chat style
        if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
            content = (data["message"].get("content") or "").strip()
            if content:
                return content
            return "Error: Model returned an empty message content."

        # 2) OpenAI /v1/chat/completions style
        if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            content = (msg.get("content") or "").strip()
            if content:
                return content
            return "Error: Model returned empty content in choices[0].message.content."

        # 3) Some backends return {"response": "..."}
        if isinstance(data, dict) and "response" in data:
            content = (data.get("response") or "").strip()
            if content:
                return content
            return "Error: Model returned empty 'response' field."

        # Unknown format
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        preview = _safe_preview(json.dumps(data, ensure_ascii=False))
        return f"Error: Unknown response format from model. Keys/Type: {keys}. Preview: {preview}"

    except requests.exceptions.Timeout:
        return "Error: Model request timed out. Please retry."
    except requests.exceptions.RequestException as e:
        # Includes connection errors + non-2xx
        status = getattr(getattr(e, "response", None), "status_code", None)
        text = getattr(getattr(e, "response", None), "text", "") or ""
        preview = _safe_preview(text)
        if status:
            return f"Error: Model HTTP {status}. Preview: {preview}"
        return f"Error: Could not connect to model. {e}"
    except Exception as e:
        # Any unexpected parsing error
        return f"Error: Unexpected failure calling model. {e}"
