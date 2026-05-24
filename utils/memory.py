"""
Conversation memory management.
Supports sliding-window buffer memory with token-aware truncation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import tiktoken

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Sliding-window conversation buffer.

    Keeps the system prompt pinned at index-0 and trims oldest
    user/assistant turns when the token budget is exceeded.
    """

    def __init__(
        self,
        system_prompt: str,
        max_tokens: int = 3000,
        model: str = "gpt-3.5-turbo",   # tiktoken encoder proxy
    ):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._turns: list[Message] = []

    # ── public API ──────────────────────────────────────────────────────────

    def add_user(self, text: str) -> None:
        self._turns.append(Message("user", text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._turns.append(Message("assistant", text))

    def get_messages(self) -> list[dict]:
        """Return full history as list-of-dicts (OpenAI / Anthropic format)."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(t.to_dict() for t in self._turns)
        return msgs

    def get_provider_messages(self) -> tuple[str, list[dict]]:
        """
        Returns (system_prompt, messages) for providers that keep system text separate.
        """
        return self.system_prompt, [t.to_dict() for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns) // 2

    # ── internals ───────────────────────────────────────────────────────────

    def _token_count(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += len(self._enc.encode(m["content"])) + 4  # overhead
        return total

    def _trim(self) -> None:
        """Drop oldest user+assistant pairs until under budget."""
        while True:
            msgs = self.get_messages()
            if self._token_count(msgs) <= self.max_tokens:
                break
            # need at least 1 turn to make progress
            if len(self._turns) < 2:
                break
            self._turns.pop(0)   # drop oldest user msg
            if self._turns and self._turns[0].role == "assistant":
                self._turns.pop(0)  # drop its paired assistant reply
