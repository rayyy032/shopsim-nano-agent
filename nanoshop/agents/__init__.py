"""Agent 策略注册表：harness 通过名字实例化策略。"""

from __future__ import annotations

from typing import Dict, Optional

from .base import BaseAgent
from .nano_agent import NanoAgent
from .react_agent import ReactAgent

AGENTS: Dict[str, type] = {
    "nano": NanoAgent,
    "react": ReactAgent,
}


def make_agent(
    name: str, llm, max_turns: int = 40, ablations: Optional[Dict[str, bool]] = None
) -> BaseAgent:
    if name not in AGENTS:
        raise ValueError(f"未知 agent {name!r}；可选：{', '.join(sorted(AGENTS))}")
    return AGENTS[name](llm, max_turns=max_turns, ablations=ablations)


__all__ = ["BaseAgent", "NanoAgent", "ReactAgent", "make_agent", "AGENTS"]
