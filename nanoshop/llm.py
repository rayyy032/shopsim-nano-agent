"""LLM layer - adapted from nano-cc (CoreCoder)'s ``corecoder/llm.py``.

Kept from the original: OpenAI-compatible single client, retry with
backoff, per-model token accounting. Dropped for this domain harness:
streaming, tool-call plumbing and the LiteLLM backend - our pipeline
issues structured-chat calls only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


@dataclass
class LLMResponse:
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    """Thin wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**params)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
                ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                self.n_calls += 1
                return LLMResponse(content=content, prompt_tokens=pt, completion_tokens=ct)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
            except APIError:
                status = getattr(APIError, "status_code", None)
                if status and status >= 500 and attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("unreachable")

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Structured output: ask for JSON, parse defensively."""
        raw = self.chat(messages).content
        return parse_json_loose(raw)


def parse_json_loose(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM reply (handles ```json fences,
    stray prose, trailing commas)."""
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    # last resort: trim trailing commas
    cleaned = text[start:].rstrip()
    for tail in ("},", "}。", "},\n"):
        if cleaned.endswith(tail):
            cleaned = cleaned[: -len(tail)] + "}"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
