"""Dry-run smoke test with a scripted LLM (no API needed).

Replays a hand-written policy for task 0: search -> click gold item ->
select option -> buy. Verifies that the agent loop, trace, output format
and metrics all work end-to-end before spending API credits.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoshop.agent import ShoppingAgent
from nanoshop.env_bridge import LocalShopEnv
from nanoshop.llm import LLM, LLMResponse


class ScriptedLLM(LLM):
    """Returns a fixed action sequence regardless of input."""

    def __init__(self, scripted: list):
        super().__init__(model="scripted", api_key="x", base_url="http://localhost")
        self.scripted = scripted
        self.i = 0
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        # NLU calls get a canned extraction; decision calls get next action
        is_nlu = any("信息抽取引擎" in m.get("content", "") for m in messages)
        if is_nlu:
            self.total_prompt_tokens += 50
            self.total_completion_tokens += 80
            self.n_calls += 1
            content = json.dumps(
                {
                    "intent": "search",
                    "slots": {
                        "category": "乳胶枕",
                        "attributes": ["泰国", "进口", "天然乳胶", "满天星"],
                        "option": "【推荐4-6岁】塔拉蕾乳胶二阶枕：满天星",
                        "budget": "1000元以下",
                        "scenario": "5岁儿童",
                    },
                    "rewritten_query": "泰国乳胶枕 儿童 满天星",
                    "new_facts": ["用户给5岁孩子买枕头"],
                },
                ensure_ascii=False,
            )
            return LLMResponse(content=content, prompt_tokens=50, completion_tokens=80)
        action = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 30
        self.n_calls += 1
        content = json.dumps(
            {"thought": "t", "action_type": "interact_with_env", "action_content": action},
            ensure_ascii=False,
        )
        return LLMResponse(content=content, prompt_tokens=100, completion_tokens=30)


def main():
    env = LocalShopEnv()
    llm = ScriptedLLM(
        [
            "search[泰国乳胶枕 儿童 满天星]",
            "click[747848614498]",
            "click[【推荐4-6岁】塔拉蕾乳胶二阶枕：满天星]",
            "click[buy now]",
        ]
    )
    agent = ShoppingAgent(env=env, llm=llm, mode="single", max_turns=10)
    out_dir = "/tmp/nanoshop_smoke"
    res = agent.run_task(0, out_dir, "scripted")

    print("reward:", res["reward"])
    print("purchase asin:", res["purchase"].get("asin"))
    print("n turns:", len(res["trace"]["turns"]))
    print("tokens:", res["tokens"])
    print("slots:", json.dumps(res["state"]["slots"], ensure_ascii=False))
    assert res["reward"] == 1.0, f"expected reward 1.0, got {res['reward']}"
    assert res["purchase"]["asin"] == "747848614498"

    # check trace structure: 7 stages present
    stages = {s["name"] for t in res["trace"]["turns"] for s in t["stages"]}
    expected = {"guardrails", "nlu", "retrieve", "verify", "reflect", "generate", "memory"}
    assert expected <= stages, f"missing stages: {expected - stages}"
    print("stages recorded:", sorted(stages))

    # metrics smoke
    from nanoshop.eval.metrics import summarize_dir

    summary = summarize_dir(f"{out_dir}/scripted")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert summary["end_to_end"]["r_success"] == 1.0
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
