"""统一 Agent 策略接口：harness 与策略解耦的关键。

任何策略（七阶段 pipeline、ReAct 基线、后续训练出的 RL policy）只要实现
BaseAgent，就能被同一个 env-loop harness 驱动、用同一套评测代码打分。
环境交互、用户模拟、结果落盘全部由 harness（nanoshop.agent.ShoppingAgent）
负责，策略只做一件事：给定本轮观察，返回下一个动作。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ..state import AgentState
from ..trace import PipelineTrace


class BaseAgent(ABC):
    """一个任务一个实例生命周期：start_task -> step* (每轮一次)。

    子类必须维护 ``self.state``（AgentState，供 harness 的守卫逻辑与
    组件级指标读取）和 ``self.trace``（PipelineTrace，供评测挖掘）。
    """

    name: str = "base"

    #: 策略是否需要访问底层环境（asin<->title 匹配等）；由 harness 注入。
    needs_env: bool = False

    def __init__(self, llm, max_turns: int = 40, ablations: Optional[Dict[str, bool]] = None):
        self.llm = llm
        self.max_turns = max_turns
        self.ablations = ablations or {}
        self.state = AgentState()
        self.trace = PipelineTrace(task_id=-1)
        self.mode = "multi"
        self.env: Any = None  # harness 在 needs_env 时注入

    @abstractmethod
    def start_task(
        self,
        task_id: int,
        instruction: str,
        user_persona: Optional[Dict[str, Any]] = None,
        mode: str = "multi",
    ) -> None:
        """任务开始：重置 state / trace / 内部记忆。"""

    @abstractmethod
    def step(self, turn: int, role: str, message: str) -> Tuple[str, str]:
        """处理一条观察（shopper 消息或环境观察），返回 (action_type, action_content)。

        action_type: "ask_user" | "interact_with_env"
        action_content: ask_user 时为对用户说的话；interact 时为环境动作
        （search[...] / click[...]）。
        """
