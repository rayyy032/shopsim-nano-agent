"""NanoAgent：把七阶段 pipeline 包成 BaseAgent 策略。

pipeline 本身不动，这里只做接口适配，保证旧评测结果可复现。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..pipeline import ShoppingPipeline
from ..state import AgentState
from ..trace import PipelineTrace
from .base import BaseAgent


class NanoAgent(BaseAgent):
    name = "nano"
    needs_env = True  # Verify 阶段用 env 做 asin<->title 匹配

    def __init__(self, llm, max_turns: int = 40, ablations: Optional[Dict[str, bool]] = None):
        # pipeline 先建好，BaseAgent.__init__ 里对 state/trace 的赋值
        # 才能通过 property 转写到 pipeline 上
        self.pipeline = ShoppingPipeline(llm=llm, max_turns=max_turns, ablations=ablations or {})
        super().__init__(llm, max_turns, ablations)

    @property
    def state(self) -> AgentState:
        return self.pipeline.state

    @state.setter
    def state(self, value: AgentState) -> None:
        self.pipeline.state = value

    @property
    def trace(self) -> PipelineTrace:
        return self.pipeline.trace

    @trace.setter
    def trace(self, value: PipelineTrace) -> None:
        self.pipeline.trace = value

    @property
    def env(self) -> Any:
        return getattr(self.pipeline, "env", None)

    @env.setter
    def env(self, value: Any) -> None:
        self.pipeline.env = value

    def start_task(
        self,
        task_id: int,
        instruction: str,
        user_persona: Optional[Dict[str, Any]] = None,
        mode: str = "multi",
    ) -> None:
        self.pipeline.start_task(task_id, instruction, user_persona, mode=mode)

    def step(self, turn: int, role: str, message: str) -> Tuple[str, str]:
        return self.pipeline.step(turn, role, message)
