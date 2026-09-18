"""Turn/Stage two-level PipelineTrace.

Every pipeline run records one TraceTurn; each stage inside appends a
TraceStage with its input digest, output digest, tool calls, latency and
token usage. Traces are what the component-level evaluation reads:
intent accuracy, slot F1, clarify-trigger precision, verify coverage,
per-stage latency and token overhead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceStage:
    name: str            # guardrails / nlu / retrieve / verify / reflect / generate / memory
    input_digest: str = ""
    output_digest: str = ""
    tool_calls: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def finish(self, output: str = "", **detail) -> "TraceStage":
        self.latency_ms = (time.time() - self.started_at) * 1000
        self.output_digest = _digest(output)
        self.detail.update(detail)
        return self


@dataclass
class TraceTurn:
    turn: int
    role: str = ""                    # shopper / env
    trigger: str = ""                 # what started this turn
    stages: List[TraceStage] = field(default_factory=list)
    action_type: str = ""             # ask_user / interact_with_env
    action_content: str = ""
    latency_ms: float = 0.0
    started_at: float = field(default_factory=time.time)

    def stage(self, name: str, input_text: str = "") -> TraceStage:
        s = TraceStage(name=name, input_digest=_digest(input_text))
        self.stages.append(s)
        return s

    def finish(self, action_type: str, action_content: str) -> None:
        self.action_type = action_type
        self.action_content = action_content
        self.latency_ms = (time.time() - self.started_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "role": self.role,
            "trigger": self.trigger[:200],
            "action_type": self.action_type,
            "action_content": self.action_content[:300],
            "latency_ms": round(self.latency_ms, 1),
            "stages": [
                {
                    "name": s.name,
                    "input": s.input_digest,
                    "output": s.output_digest,
                    "tools": s.tool_calls,
                    "latency_ms": round(s.latency_ms, 1),
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    **({"detail": s.detail} if s.detail else {}),
                }
                for s in self.stages
            ],
        }


class PipelineTrace:
    """Accumulates TraceTurns for one task."""

    def __init__(self, task_id: int, mode: str = "multi"):
        self.task_id = task_id
        self.mode = mode
        self.turns: List[TraceTurn] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def new_turn(self, turn: int, role: str, trigger: str) -> TraceTurn:
        t = TraceTurn(turn=turn, role=role, trigger=trigger)
        self.turns.append(t)
        return t

    def add_tokens(self, pt: int, ct: int) -> None:
        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "turns": [t.to_dict() for t in self.turns],
        }

    # -- component-level metric helpers ----------------------------------

    @property
    def n_ask_user(self) -> int:
        return sum(1 for t in self.turns if t.action_type == "ask_user")

    @property
    def n_interact(self) -> int:
        return sum(1 for t in self.turns if t.action_type == "interact_with_env")

    def stage_views(self, name: str) -> List[TraceStage]:
        return [s for t in self.turns for s in t.stages if s.name == name]


def _digest(text: str, limit: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")
