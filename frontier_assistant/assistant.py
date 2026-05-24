"""
Frontier Assistant - Groq API.

Supports:
  - Multi-turn conversation with ConversationMemory
  - Input/output guardrails
  - Observability tracing
  - OpenAI-compatible tool calling: web_search, calculator
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from groq import Groq

from utils.memory import ConversationMemory
from utils.guardrails import check_input, check_output
from utils.observability import tracer

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant powered by Groq.
You give clear, accurate, and thoughtful responses. When unsure, you say so honestly.
You do not fabricate facts. You use tools when they would genuinely help answer a question.
Current date: {date}"""


def _run_calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: invalid expression"
        result = eval(expression, {"__builtins__": {}, "math": math})  # noqa: S307
        return str(round(float(result), 8))
    except Exception as e:
        return f"Error: {e}"


def _run_web_search(query: str) -> str:
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract[:600]
        topics = data.get("RelatedTopics", [])
        if topics and isinstance(topics[0], dict):
            return topics[0].get("Text", "No result.")[:600]
        return "No result found."
    except Exception as e:
        return f"Search error: {e}"


TOOL_REGISTRY = {
    "calculator": _run_calculator,
    "web_search": _run_web_search,
}

GROQ_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe math expression, e.g. '(3+5)*2'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


class FrontierAssistant:
    def __init__(self):
        from datetime import date

        system = SYSTEM_PROMPT.format(date=date.today().isoformat())
        self.memory = ConversationMemory(system_prompt=system, max_tokens=3000)
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        self._session_id = f"frontier-{int(time.time())}"

    def _call_with_tools(self, messages: list[dict]) -> tuple[str, int]:
        """
        Runs an OpenAI-compatible tool-use loop through Groq.

        Returns (reply_text, total_prompt_tokens).
        """
        total_tokens = 0
        current_messages = list(messages)
        last_message = None

        for _ in range(5):
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=current_messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
                max_tokens=GROQ_MAX_TOKENS,
                temperature=0.4,
            )
            choice = response.choices[0]
            message = choice.message
            last_message = message
            if response.usage:
                total_tokens += response.usage.prompt_tokens or 0

            tool_calls = message.tool_calls or []
            if not tool_calls:
                return message.content or "", total_tokens

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            current_messages.append(assistant_message)

            for tool_call in tool_calls:
                name = tool_call.function.name
                fn = TOOL_REGISTRY.get(name)
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = fn(**args) if fn else f"Unknown tool: {name}"
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        return (last_message.content if last_message else "") or "[No response]", total_tokens

    def chat(self, user_message: str) -> dict:
        guard_in = check_input(user_message)
        if not guard_in.is_safe:
            return {
                "reply": f"Warning: {guard_in.reason}. I can't respond to that.",
                "guard_input": guard_in,
                "guard_output": None,
                "latency_ms": 0,
                "input_tokens": 0,
                "model": GROQ_MODEL,
                "blocked": True,
            }

        self.memory.add_user(guard_in.sanitized)
        messages = self.memory.get_messages()

        with tracer.trace("frontier_chat", model=GROQ_MODEL, session_id=self._session_id) as span:
            span.set_input(user_message)
            try:
                reply, tokens = self._call_with_tools(messages)
            except Exception as e:
                reply = f"[Groq API error: {e}]"
                tokens = 0
            span.set_output(reply)
            span.set_metrics(input_tokens=tokens)
            latency = span.latency_ms()

        guard_out = check_output(reply)
        final_reply = guard_out.sanitized
        self.memory.add_assistant(final_reply)

        return {
            "reply": final_reply,
            "guard_input": guard_in,
            "guard_output": guard_out,
            "latency_ms": latency,
            "input_tokens": tokens,
            "model": GROQ_MODEL,
            "blocked": False,
        }

    def reset(self) -> None:
        self.memory.clear()
        self._session_id = f"frontier-{int(time.time())}"

    @property
    def history(self) -> list[dict]:
        return self.memory.get_messages()
