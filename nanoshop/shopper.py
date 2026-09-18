"""LLM-shopper simulator, adapted from the official multi_eval/shopper.py.

The shopper hides a concrete purchase goal and only reveals details when
asked - exactly the multi-turn challenge ShopSimulator is designed for.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .llm import LLM

SYSTEM_PROMPT = """你是一个模拟的购物者（shopper），正在尝试通过与客服代理（agent）的对话完成一次商品购买任务。你的任务是：

1. 你有一个**具体的购买目标**（只有你知道），但你不会一开始就透露它。你需要从一个模糊的购买目标出发，与 agent 进行自然的多轮对话。
2. 在对话中，你主要作用是回答 agent 的提问，帮助它逐渐了解你的购买需求，最终让 agent 帮你找到并推荐出你实际想买的商品。
3. 注意：**不要主动提供详细信息，而是等待agent向你询问。**
4. 注意：**在对话结束前，也就是agent准备要购买之前，确保agent已经得到了具体购买目标中的所有特征和属性，不可以遗漏任何信息。**
5. 若有信息未告知agent，请拒绝agent的购买行为，并提供一个简单的理由，等待agent的更多询问。

请保持语言自然、口语化，像一个真实的人在购物时那样思考和表达。"""


class ShopperSimulator:
    def __init__(self, llm: LLM, system_prompt: str = ""):
        self.llm = llm
        self.base_system_prompt = system_prompt or SYSTEM_PROMPT
        self.messages: List[Dict[str, str]] = []

    def reset(self, instruction: str, goal_options: List[str]) -> None:
        self.messages = []
        options = json.dumps(goal_options, ensure_ascii=False)
        task_info = (
            f"\n\n你本轮的购买任务是: {instruction}\n你的规格选择是: {options}"
        )
        self.messages.append(
            {"role": "system", "content": self.base_system_prompt + task_info}
        )

    def step(self, agent_response: str) -> str:
        self.messages.append({"role": "user", "content": agent_response})
        resp = self.llm.chat(self.messages)
        content = resp.content.strip()
        if not content:
            content = "我再想想。"
        self.messages.append({"role": "assistant", "content": content})
        return content
