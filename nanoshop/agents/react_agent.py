"""ReactAgent：单 prompt 的 ReAct 基线，作为七阶段结构的对照组。

与 NanoAgent 的唯一差异是"没有结构"：
- 无 NLU 槽位抽取、无证据状态、无 Memory 压缩——全部信息就是对话 +
  观察原文塞进一个 prompt；
- 每轮一次 LLM 调用，输出 Thought/Action。
组件级指标（slot F1、verify coverage 等）在 ReAct 上自然缺失或归零，
这正是对比要展示的东西：harness 能把"结构带来的收益"逐组件量化。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm import LLM
from ..state import AgentState
from ..trace import PipelineTrace
from .base import BaseAgent

REACT_SYSTEM = """你是一个购物助手Agent，正在帮用户在电商环境中购买最合适的商品。

每轮你会收到一条观察（用户消息或环境页面）。请输出：
Thought: 简要思考
Action: 以下之一
- ask_user[想问用户的问题]
- search[关键词]
- click[按钮或商品asin或规格]
- click[buy now]（购买当前商品，必须已确认需求并选好规格）

规则：
- 需求不明就先问用户；信息够了就搜索、进详情页核验、选规格、确认后购买
- click 的值必须来自当前可点击按钮列表
- 严禁重复执行完全相同的动作
- 共 {max_turns} 轮，快结束时果断购买最合适商品
"""

ACTION_RE = re.compile(
    r"Action:\s*(ask_user|search|click|buy_now|buy now)\s*\[([^\]]*)\]", re.S
)


class ReactAgent(BaseAgent):
    name = "react"

    def __init__(self, llm: LLM, max_turns: int = 40, ablations: Optional[Dict[str, bool]] = None):
        super().__init__(llm, max_turns, ablations)
        self.history: List[Dict[str, str]] = []

    def start_task(
        self,
        task_id: int,
        instruction: str,
        user_persona: Optional[Dict[str, Any]] = None,
        mode: str = "multi",
    ) -> None:
        self.state = AgentState(task_id=task_id, instruction=instruction, user_persona=user_persona)
        self.trace = PipelineTrace(task_id=task_id, mode=mode)
        self.mode = mode
        self.history = []

    def step(self, turn: int, role: str, message: str) -> Tuple[str, str]:
        state = self.state
        state.turn = turn
        t = self.trace.new_turn(turn, role, message)

        s = t.stage("llm", message)
        if role == "Shopper" or "Shopper" in message[:20]:
            self.history.append({"role": "user", "content": message})
        else:
            self.history.append({"role": "user", "content": f"[环境观察] {message}"})

        action_type, action_content, pt, ct = self._decide(turn)
        s.finish(
            output=action_content[:160],
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        self.trace.add_tokens(pt, ct)
        self.history.append({"role": "assistant", "content": f"Thought/Action done ({action_content[:60]})"})

        if action_type == "ask_user":
            state.dialogue_turns += 1
        t.finish(action_type, action_content)
        return action_type, action_content

    def _decide(self, turn: int) -> Tuple[str, str, int, int]:
        # 保留最近的历史；老消息截断，不做任何结构化压缩
        history_text = self._render_history()
        sys_prompt = REACT_SYSTEM.format(max_turns=self.max_turns)
        if self.mode == "single":
            sys_prompt += "\n当前为单轮模式（无用户交互），禁止 ask_user，直接根据任务指令搜索并购买。"
        if self.state.user_persona:
            import json as _json

            persona = _json.dumps(self.state.user_persona, ensure_ascii=False)[:800]
            sys_prompt += f"\n\n用户个人画像（参考）：{persona}"

        user_prompt = f"[历史对话与环境观察]\n{history_text}\n\n请输出本轮 Thought 与 Action。"

        resp = self.llm.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        action_type, action_content = self._parse(resp.content)
        # 记录搜索历史，供组件指标 / 死循环守卫使用
        if action_type == "interact_with_env" and action_content.startswith("search["):
            q = action_content[len("search[") : action_content.rfind("]")]
            if q and q not in self.state.search_history:
                self.state.search_history.append(q)
        return action_type, action_content, resp.prompt_tokens, resp.completion_tokens

    def _parse(self, text: str) -> Tuple[str, str]:
        m = ACTION_RE.search(text)
        if not m:
            # 宽松兜底：直接把 Action: 后面整行当环境动作
            m2 = re.search(r"Action:\s*(.+)", text)
            if m2:
                content = m2.group(1).strip()
                if content.startswith("ask_user"):
                    return "ask_user", content[len("ask_user") :].strip("[] ：:")
                return "interact_with_env", content
            return "interact_with_env", "search[乳胶枕]"
        verb, arg = m.group(1).lower(), m.group(2).strip()
        if verb == "ask_user":
            return "ask_user", arg
        return "interact_with_env", f"{verb}[{arg}]"

    def _render_history(self, max_messages: int = 16, per_msg: int = 400) -> str:
        lines = []
        for msg in self.history[-max_messages:]:
            content = msg["content"][:per_msg]
            lines.append(f"{msg['role']}: {content}")
        return "\n".join(lines)
