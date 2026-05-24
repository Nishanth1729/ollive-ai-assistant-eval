"""
Guardrails layer — applied to both user inputs and model outputs.

Two-stage pipeline:
  1. Rule-based fast filter  (regex + keyword blocklist)
  2. LLM-as-judge safety check (async, optional — only when rule filter is uncertain)

Returns a GuardResult with:
  - is_safe: bool
  - category: str  ("safe" | "profanity" | "jailbreak" | "harmful" | "bias")
  - reason: str
  - sanitized: str  (original text if safe, else redacted placeholder)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Blocklists ──────────────────────────────────────────────────────────────

# Common jailbreak / prompt-injection patterns
JAILBREAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions?", re.I),
    re.compile(r"you are now (DAN|an? AI with no restrictions)", re.I),
    re.compile(r"(pretend|act|roleplay|simulate).{0,40}(no restrictions|evil|unethical|harmful)", re.I),
    re.compile(r"bypass.{0,30}(safety|filter|guard|moderation)", re.I),
    re.compile(r"do anything now", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"prompt injection", re.I),
    re.compile(r"(disregard|forget|override).{0,20}(instruction|guideline|policy|rule)", re.I),
]

# Harmful content keywords (non-exhaustive — supplement with LLM judge)
HARMFUL_KEYWORDS: set[str] = {
    "make a bomb", "build a weapon", "synthesize drugs",
    "create malware", "hack into", "exploit vulnerability",
    "child pornography", "csam", "self-harm instructions",
    "how to kill", "commit suicide method",
}

# Bias / hate speech triggers
BIAS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(all|every)\s+(women|men|blacks|whites|jews|muslims|christians)\s+(are|is)\b", re.I),
    re.compile(r"\b(inferior|superior)\s+(race|gender|religion)\b", re.I),
]


@dataclass
class GuardResult:
    is_safe: bool
    category: str       # "safe" | "profanity" | "jailbreak" | "harmful" | "bias"
    reason: str
    sanitized: str      # original if safe, else a placeholder


class Guardrails:
    """
    Lightweight, zero-dependency guardrail (no model call required for most cases).
    For production, extend with an actual LLM-judge pass on uncertain inputs.
    """

    REDACTED = "[Content removed by safety filter]"

    def check(self, text: str, context: str = "input") -> GuardResult:
        """
        Run all checks and return the first violation found.
        context: "input" (user message) | "output" (model response)
        """
        # 1. Jailbreak / prompt injection
        for pattern in JAILBREAK_PATTERNS:
            if pattern.search(text):
                return GuardResult(
                    is_safe=False,
                    category="jailbreak",
                    reason=f"Jailbreak pattern detected: '{pattern.pattern}'",
                    sanitized=self.REDACTED,
                )

        # 2. Harmful keywords
        text_lower = text.lower()
        for kw in HARMFUL_KEYWORDS:
            if kw in text_lower:
                return GuardResult(
                    is_safe=False,
                    category="harmful",
                    reason=f"Harmful keyword: '{kw}'",
                    sanitized=self.REDACTED,
                )

        # 3. Bias / hate speech
        for pattern in BIAS_PATTERNS:
            if pattern.search(text):
                return GuardResult(
                    is_safe=False,
                    category="bias",
                    reason=f"Potential bias/hate-speech pattern detected",
                    sanitized=self.REDACTED,
                )

        # 4. Profanity (simple — upgrade with `better-profanity` if installed)
        try:
            from better_profanity import profanity
            if profanity.contains_profanity(text):
                cleaned = profanity.censor(text)
                return GuardResult(
                    is_safe=True,          # safe but sanitized
                    category="profanity",
                    reason="Profanity censored",
                    sanitized=cleaned,
                )
        except ImportError:
            pass

        return GuardResult(
            is_safe=True,
            category="safe",
            reason="No violations detected",
            sanitized=text,
        )

    def input_check(self, user_text: str) -> GuardResult:
        return self.check(user_text, context="input")

    def output_check(self, model_text: str) -> GuardResult:
        return self.check(model_text, context="output")


# Module-level singleton
_guardrails = Guardrails()


def check_input(text: str) -> GuardResult:
    return _guardrails.input_check(text)


def check_output(text: str) -> GuardResult:
    return _guardrails.output_check(text)
