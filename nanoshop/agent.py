"""Shopping agent: wires the seven-stage pipeline to the (local) ShopSimulator
environment and, in multi-turn mode, to the LLM-shopper simulator.

Compatible with the official eval protocol: one JSON per task containing
reward / reward_detail / goal / purchase / conversation, plus our
pipeline trace and token accounting.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Dict, Optional

from .agents.base import BaseAgent
from .env_bridge import LocalShopEnv
from .llm import LLM
from .pipeline import ROLE_ENV, ROLE_SHOPPER
from .shopper import ShopperSimulator

logger = logging.getLogger(__name__)


class ShoppingAgent:
    """env-loop harness：驱动任意 BaseAgent 策略跑完一个任务。

    环境交互、用户模拟、守卫逻辑（死循环 / 规格选择 / 购买确认）、
    结果落盘都在这里；策略只负责"看观察、出动作"。评测代码与策略
    无关，换 nano / react / 训练出的 policy 都走同一条路。
    """

    def __init__(
        self,
        env: LocalShopEnv,
        llm: LLM,
        mode: str = "multi",
        max_turns: int = 40,
        ablations: Optional[Dict[str, bool]] = None,
        shopper: Optional[ShopperSimulator] = None,
        policy: Optional[BaseAgent] = None,
    ):
        self.env = env
        self.llm = llm
        self.mode = mode  # single | multi
        self.max_turns = max_turns
        self.ablations = ablations or {}
        self.shopper = shopper
        self.policy = policy
        if self.policy is None:
            # 向后兼容：不传 policy 时默认七阶段 nano
            from .agents import make_agent

            self.policy = make_agent("nano", llm, max_turns=max_turns, ablations=self.ablations)
        if getattr(self.policy, "needs_env", False):
            self.policy.env = env

    def run_task(self, task_id: int, output_dir: str, model_tag: str = "") -> Dict[str, Any]:
        """Run one task end-to-end; writes {output_dir}/{model_tag}/{task_id}.json
        (or directly {output_dir}/{task_id}.json when model_tag is empty)."""
        result: Dict[str, Any] = {"task_id": task_id}
        self._tok_base = (
            self.llm.total_prompt_tokens,
            self.llm.total_completion_tokens,
        )
        t0 = time.time()
        try:
            result = self._run(task_id)
        except Exception as e:  # noqa: BLE001 - keep batch eval alive
            logger.exception("task %s failed", task_id)
            result = {
                "task_id": task_id,
                "reward": 0,
                "reward_detail": {},
                "goal": {},
                "purchase": {},
                "conversation": [],
                "error": f"{e}: {traceback.format_exc()[-800:]}",
            }
            result["tokens"] = self._token_usage()
        finally:
            self.env.release()
        result["latency_s"] = round(time.time() - t0, 1)
        result["ablations"] = self.ablations
        result["mode"] = self.mode
        result["agent"] = getattr(self.policy, "name", "unknown")

        out = os.path.join(output_dir, model_tag) if model_tag else output_dir
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{task_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # ------------------------------------------------------------------

    def _run(self, task_id: int) -> Dict[str, Any]:
        env = self.env
        reset_info = env.reset(task_id)

        if env.if_persona:
            instruction = reset_info["instruction_simple"]
        else:
            instruction = reset_info["instruction"]

        persona = reset_info.get("user_persona")
        self.policy.start_task(task_id, instruction, persona, mode=self.mode)

        conversation: list = []
        done = False
        turn = 0
        observation = instruction + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
        shopper_msg: Optional[str] = None

        if self.mode == "multi":
            assert self.shopper is not None
            # The shopper always holds the concrete goal (official multi_eval
            # semantics: shopper.reset(env_result["instruction"], ...)); in
            # persona mode only the agent side is downgraded to
            # instruction_simple + persona document.
            self.shopper.reset(reset_info["instruction"], reset_info.get("goal_options", []))
            shopper_msg = self.shopper.step("请提供您的购物需求。")
            conversation.append({"shopper": shopper_msg})

        role = ROLE_SHOPPER if self.mode == "multi" else ROLE_ENV
        message = shopper_msg if self.mode == "multi" else observation

        while not done and turn < self.max_turns:
            turn += 1
            action_type, action_content = self.policy.step(turn, role, message)
            conversation.append({"agent_action": f"{action_type}: {action_content}"})

            # --- state bookkeeping for guards ---
            state = self.policy.state
            state.last_actions.append(f"{action_type}:{action_content}")

            if action_type == "ask_user":
                if self.mode != "multi" or self.shopper is None:
                    # single-turn mode has no one to ask: coerce to env action
                    action_type, action_content = "interact_with_env", "search[乳胶枕]"
                else:
                    message = self.shopper.step(action_content)
                    conversation.append({"shopper": message})
                    # shopper agreement (in reply to a purchase confirmation) marks confirmation
                    if any(k in message for k in ("同意", "可以", "好的", "没错", "就这个", "确认", "嗯", "行")):
                        state.confirmed_with_user = True
                    role = ROLE_SHOPPER
                    continue

            # interact_with_env: track option selection
            if action_content.startswith("click[") and action_content.endswith("]"):
                clicked = action_content[6:-1]
                nav = {"back to search", "< prev", "next >", "buy now", "description", "features", "reviews", "attributes", "search"}
                if clicked not in nav and not clicked.isdigit():
                    state.selected_option = clicked
                    # attach to the current product view if known
                    # (nano 策略把 _last_item_asin 记在 pipeline 上)
                    policy_impl = getattr(self.policy, "pipeline", self.policy)
                    for v in state.viewed_products.values():
                        if v.asin == getattr(policy_impl, "_last_item_asin", None):
                            v.selected_option = clicked

            env_resp = self.env.interact(f"Thought: pipeline\nAction: {action_content}")
            if "error" in env_resp and env_resp["error"]:
                message = f"环境错误：{env_resp['error']}"
                role = ROLE_ENV
                continue

            observation = env_resp["instruction"]
            message = observation
            role = ROLE_ENV

            if env_resp.get("done") or env_resp.get("over"):
                return self._finalize(env_resp, conversation)

        # turns exhausted: force-record with zero reward (official protocol)
        return self._finalize(
            {"reward": 0, "reward_detail": {}, "goal": {}, "purchase": {}},
            conversation,
        )

    def _token_usage(self) -> Dict[str, int]:
        """Per-task tokens: difference of LLM lifetime counters since run start."""
        base_p, base_c = getattr(self, "_tok_base", (0, 0))
        return {
            "prompt": self.llm.total_prompt_tokens - base_p,
            "completion": self.llm.total_completion_tokens - base_c,
            "n_llm_calls": self.llm.n_calls,
        }

    def _finalize(self, env_resp: Dict[str, Any], conversation: list) -> Dict[str, Any]:
        return {
            "task_id": self.policy.state.task_id,
            "reward": env_resp.get("reward", 0),
            "reward_detail": env_resp.get("reward_detail", {}),
            "goal": env_resp.get("goal", {}),
            "purchase": env_resp.get("purchase", {}),
            "conversation": conversation,
            "trace": self.policy.trace.to_dict(),
            "tokens": self._token_usage(),
            "state": self.policy.state.snapshot(),
        }
