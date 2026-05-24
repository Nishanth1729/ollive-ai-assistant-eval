"""
Observability wrapper around Langfuse.

Provides a lightweight trace/span API that degrades gracefully
when Langfuse credentials are not configured (logs to console instead).

Usage:
    from utils.observability import tracer

    with tracer.trace("chat_turn", user_id="u1", model="groq") as span:
        reply = call_model(...)
        span.set_output(reply)
        span.set_metrics(tokens=100, latency_ms=420)
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.time)
    _input: Optional[str] = None
    _output: Optional[str] = None
    _metrics: dict = field(default_factory=dict)
    _lf_span: Any = None   # Langfuse span if available

    def set_input(self, text: str) -> None:
        self._input = text
        if self._lf_span:
            try:
                self._lf_span.update(input=text)
            except Exception:
                pass

    def set_output(self, text: str) -> None:
        self._output = text
        if self._lf_span:
            try:
                self._lf_span.update(output=text)
            except Exception:
                pass

    def set_metrics(self, **kwargs) -> None:
        self._metrics.update(kwargs)
        if self._lf_span:
            try:
                self._lf_span.update(metadata=self._metrics)
            except Exception:
                pass

    def latency_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class Tracer:
    """
    Thin wrapper.  Uses Langfuse when credentials are present,
    falls back to a no-op + console log otherwise.
    """

    def __init__(self):
        self._lf = self._init_langfuse()

    def _init_langfuse(self):
        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not (pk and sk):
            return None
        try:
            from langfuse import Langfuse
            return Langfuse(public_key=pk, secret_key=sk, host=host)
        except Exception:
            return None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str = "anon",
        model: str = "unknown",
        session_id: str = "",
    ) -> Generator[Span, None, None]:
        trace_id = str(uuid.uuid4())[:12]
        lf_trace = None
        lf_span = None

        if self._lf:
            try:
                lf_trace = self._lf.trace(
                    name=name,
                    user_id=user_id,
                    metadata={"model": model, "session_id": session_id},
                )
                lf_span = lf_trace.span(name=name)
            except Exception:
                pass

        span = Span(name=name, trace_id=trace_id, _lf_span=lf_span)
        try:
            yield span
        finally:
            latency = span.latency_ms()
            span.set_metrics(latency_ms=round(latency, 1))
            if lf_span:
                try:
                    lf_span.end()
                except Exception:
                    pass
            if not self._lf:
                print(
                    f"[trace] {name} | model={model} | "
                    f"latency={latency:.0f}ms | "
                    f"metrics={span._metrics}"
                )

    def flush(self):
        if self._lf:
            try:
                self._lf.flush()
            except Exception:
                pass


# Module-level singleton
tracer = Tracer()
