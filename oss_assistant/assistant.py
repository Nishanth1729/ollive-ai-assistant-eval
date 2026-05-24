"""
OSS Assistant — Qwen2.5-0.5B-Instruct via HuggingFace Inference API.

Supports:
  - Multi-turn conversation with ConversationMemory
  - Input/output guardrails
  - Observability tracing
  - Tool use: web_search (via DuckDuckGo), calculator

Deployment options (set in .env):
  Option A: HF_SPACE_URL  — your own HF Spaces Gradio endpoint
  Option B: HF_MODEL_ID   — HF Inference API (free tier or PRO)
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Generator, Optional

import requests
from dotenv import load_dotenv

from utils.memory import ConversationMemory
from utils.guardrails import check_input, check_output, GuardResult
from utils.observability import tracer

load_dotenv()

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant powered by Qwen2.5.
You answer questions clearly and concisely. If you are unsure about a fact, say so.
You do not make up information. You can use available tools when needed.
Current date: {date}"""

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")

# ── Tool definitions ─────────────────────────────────────────────────────────

def _calculator(expression: str) -> str:
    """Safe math expression evaluator."""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression, {"__builtins__": {}, "math": math})  # noqa: S307
        return str(round(float(result), 8))
    except Exception as e:
        return f"Error: {e}"


def _web_search(query: str) -> str:
    """DuckDuckGo instant answers (no API key required)."""
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=5,
        )
        data = resp.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract[:500]
        related = data.get("RelatedTopics", [])
        if related and isinstance(related[0], dict):
            return related[0].get("Text", "No results found.")[:500]
        return "No results found."
    except Exception as e:
        return f"Search unavailable: {e}"


TOOLS = {
    "calculator": _calculator,
    "web_search": _web_search,
}

TOOL_DESCRIPTIONS = """
Available tools (call with JSON):
- calculator: evaluate math expressions  {"tool":"calculator","expression":"2+2"}
- web_search: search the web             {"tool":"web_search","query":"your query"}
To use a tool, output ONLY the JSON line. The result will be appended and you continue.
"""


# ── OSS Assistant class ──────────────────────────────────────────────────────

class OSSAssistant:
    def __init__(self):
        from datetime import date
        system = (SYSTEM_PROMPT.format(date=date.today().isoformat()) + TOOL_DESCRIPTIONS)
        self.memory = ConversationMemory(system_prompt=system, max_tokens=2000)
        self._session_id = f"oss-{int(time.time())}"

    # ── inference ────────────────────────────────────────────────────────────

    def _call_hf_inference_api(self, messages: list[dict]) -> str:
        """HuggingFace serverless Inference API."""
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "model": HF_MODEL_ID,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_hf_space(self, messages: list[dict]) -> str:
        """Gradio Space endpoint (if self-hosted)."""
        url = f"{HF_SPACE_URL.rstrip('/')}/api/predict"
        payload = {"data": [messages]}
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["data"][0]

    def _call_model(self, messages: list[dict]) -> str:
        if HF_SPACE_URL:
            return self._call_hf_space(messages)
        return self._call_hf_inference_api(messages)

    # ── tool-use loop ────────────────────────────────────────────────────────

    def _run_with_tools(self, messages: list[dict], max_steps: int = 3) -> str:
        for _ in range(max_steps):
            response = self._call_model(messages)
            # detect tool call
            stripped = response.strip()
            if stripped.startswith("{") and '"tool"' in stripped:
                try:
                    tool_call = json.loads(stripped)
                    tool_name = tool_call.get("tool")
                    if tool_name in TOOLS:
                        arg_key = "expression" if tool_name == "calculator" else "query"
                        result = TOOLS[tool_name](tool_call.get(arg_key, ""))
                        # append tool result and continue
                        messages.append({"role": "assistant", "content": stripped})
                        messages.append({"role": "user", "content": f"Tool result: {result}"})
                        continue
                except json.JSONDecodeError:
                    pass
            return response
        return response  # max steps reached

    # ── public API ───────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:
        """
        Returns:
            {
              "reply": str,
              "guard_input": GuardResult,
              "guard_output": GuardResult,
              "latency_ms": float,
              "model": str,
            }
        """
        # 1. Input guardrail
        guard_in = check_input(user_message)
        if not guard_in.is_safe:
            return {
                "reply": f"⚠️ {guard_in.reason}. I can't respond to that.",
                "guard_input": guard_in,
                "guard_output": None,
                "latency_ms": 0,
                "model": HF_MODEL_ID,
                "blocked": True,
            }

        self.memory.add_user(guard_in.sanitized)
        messages = self.memory.get_messages()

        with tracer.trace("oss_chat", model=HF_MODEL_ID, session_id=self._session_id) as span:
            span.set_input(user_message)
            try:
                reply = self._run_with_tools(messages)
            except Exception as e:
                reply = f"[Model error: {e}]"
            span.set_output(reply)
            latency = span.latency_ms()

        # 2. Output guardrail
        guard_out = check_output(reply)
        final_reply = guard_out.sanitized

        self.memory.add_assistant(final_reply)

        return {
            "reply": final_reply,
            "guard_input": guard_in,
            "guard_output": guard_out,
            "latency_ms": latency,
            "model": HF_MODEL_ID,
            "blocked": False,
        }

    def reset(self) -> None:
        self.memory.clear()
        self._session_id = f"oss-{int(time.time())}"

    @property
    def history(self) -> list[dict]:
        return self.memory.get_messages()
