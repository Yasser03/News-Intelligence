"""LLM client abstraction (Groq / Ollama via LangChain) with JSON parsing, rate limiting and retries."""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections import deque
from typing import Any, Optional, Protocol

from news_radar.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Anything that turns a prompt into text. Tests use a fake implementation."""

    model_name: str

    def complete(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating <think> blocks, code fences and surrounding prose.

    Raises ``ValueError`` if no JSON object/array can be recovered.
    """
    if not isinstance(text, str):
        raise ValueError("LLM returned non-text output")
    cleaned = _FENCE.sub("", _THINK.sub("", text)).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM output: {cleaned[:200]!r}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TokenBudget:
    """Thread-safe sliding-window tokens-per-minute limiter."""

    def __init__(self, tpm_limit: int, window_s: float = 60.0):
        self.tpm_limit = max(1, tpm_limit)
        self.window_s = window_s
        self._events: deque = deque()
        self._used = 0
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window_s:
            _, tokens = self._events.popleft()
            self._used -= tokens

    def acquire(self, tokens: int) -> None:
        tokens = min(max(1, tokens), self.tpm_limit)  # a single oversized request must still pass
        while True:
            with self._lock:
                now = time.monotonic()
                self._expire(now)
                if self._used + tokens <= self.tpm_limit:
                    self._events.append((now, tokens))
                    self._used += tokens
                    return
            time.sleep(0.25)


def estimate_tokens(text: str) -> int:
    return max(300, len(text) // 4 + 600)  # prompt + expected completion (incl. reasoning)


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in ("rate limit", "rate_limit", "429", "too many requests", "tpm", "rpm"))


# ---------------------------------------------------------------------------
# LangChain-backed client
# ---------------------------------------------------------------------------

class LangChainLLM:
    """Groq or Ollama chat model in JSON mode with retry/backoff and a TPM budget."""

    def __init__(self, cfg: LLMConfig, model: Optional[str] = None, budget: Optional[TokenBudget] = None):
        self.cfg = cfg
        self.model = model or cfg.model_name
        self.model_name = f"{cfg.backend}:{self.model}"
        # One budget per client: Groq enforces tokens-per-minute separately for each model.
        self.budget = budget or (TokenBudget(cfg.tpm_limit) if cfg.backend == "groq" else None)
        self._llm = self._build()

    def _build(self):
        if self.cfg.backend == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = dict(model=self.model, temperature=self.cfg.temperature, format="json")
            if self.cfg.ollama_base_url:
                kwargs["base_url"] = self.cfg.ollama_base_url
            return ChatOllama(**kwargs)
        if self.cfg.backend == "groq":
            from langchain_groq import ChatGroq

            extra = {}
            if self.model.startswith("openai/gpt-oss") and self.cfg.reasoning_effort:
                extra["reasoning_effort"] = self.cfg.reasoning_effort
            return ChatGroq(
                model=self.model,
                temperature=self.cfg.temperature,
                model_kwargs={"response_format": {"type": "json_object"}},
                max_retries=0,  # we handle retries ourselves
                **extra,
            )
        raise ValueError(f"Unknown LLM backend: {self.cfg.backend!r} (use 'groq' or 'ollama')")

    def complete(self, prompt: str) -> str:
        if self.budget:
            self.budget.acquire(estimate_tokens(prompt))
        last: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                msg = self._llm.invoke(prompt)
                return msg.content if hasattr(msg, "content") else str(msg)
            except Exception as e:  # network / provider errors
                last = e
                base = 2.0 if _is_rate_limit(e) else 0.75
                delay = min(30.0, base * (2 ** attempt)) + random.uniform(0, 0.5)
                logger.warning("LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                               attempt + 1, self.cfg.max_retries, e, delay)
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} attempts: {last}")


def build_llm(cfg: LLMConfig, triage: bool = False) -> LLMClient:
    """Analysis client by default; ``triage=True`` gives the (cheaper) triage model on Groq."""
    if triage and cfg.backend == "groq" and cfg.groq_triage_model:
        return LangChainLLM(cfg, model=cfg.groq_triage_model)
    return LangChainLLM(cfg)
