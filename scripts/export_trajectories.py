"""导出评测轨迹为训练可用的 JSONL。

两类输出（--out 控制目录，默认 exports/）：

1. shopsim_tasks.jsonl —— 任务级记录，字段对齐 Search-R1 的
   datasets/ecommerce/train.jsonl（id / question / answers / data_source），
   外加完整 gold goal，供 RL rollout / 奖励计算直接消费。

2. shopsim_trajectories.jsonl —— 轨迹级记录：多轮 messages
   （shopper=user，agent=assistant，环境动作挂在 assistant 消息的
   env_action 字段），可用 --min-reward 做拒绝采样（只保留高奖励轨迹），
   作为 SFT 冷启动或 IGPO 伪序列前向的输入。

Usage:
    python scripts/export_trajectories.py --setting multi_standard --tag main \
        [--min-reward 0.9] [--out exports]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def conversation_to_messages(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 harness 的 conversation 列表转成标准多轮消息。

    shopper 消息 -> user；agent 动作 -> assistant。
    ask_user 动作的 content 是对用户说的话；环境动作记在 env_action 字段
    （无用户可见文本）。一条 agent 轮只对应其中一种动作。
    """
    messages = []
    for entry in conversation:
        if not isinstance(entry, dict):
            continue
        if "shopper" in entry:
            messages.append({"role": "user", "content": entry["shopper"]})
        elif "agent_action" in entry:
            raw = entry["agent_action"]
            action_type, _, content = raw.partition(": ")
            if action_type == "ask_user":
                messages.append({"role": "assistant", "content": content, "env_action": None})
            else:
                messages.append(
                    {"role": "assistant", "content": "", "env_action": content}
                )
    return messages


def export(setting: str, tag: str, min_reward: float, out_dir: str) -> Dict[str, int]:
    src = PROJECT_ROOT / "outputs" / setting / tag
    files = sorted(glob.glob(str(src / "*.json")))
    n_task = n_traj = 0
    os.makedirs(out_dir, exist_ok=True)

    tasks_path = os.path.join(out_dir, f"{setting}_{tag}_tasks.jsonl".replace("+", "_"))
    traj_path = os.path.join(out_dir, f"{setting}_{tag}_trajectories.jsonl".replace("+", "_"))

    with open(tasks_path, "w", encoding="utf-8") as ft, open(traj_path, "w", encoding="utf-8") as fr:
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if "task_id" not in rec:
                continue  # metrics.json 等
            goal = rec.get("goal") or {}
            if not goal:
                continue

            # 1) 任务级记录（Search-R1 兼容字段 + 完整 gold）
            task_rec = {
                "id": f"shopsim_{setting}_{rec['task_id']}",
                "question": goal.get("instruction_text") or goal.get("instruction_simple", ""),
                "answers": [goal["name"]] if goal.get("name") else [],
                "data_source": f"shopsim_{setting}",
                "gold": goal,
            }
            ft.write(json.dumps(task_rec, ensure_ascii=False) + "\n")
            n_task += 1

            # 2) 轨迹级记录（可拒绝采样）
            reward = rec.get("reward", 0) or 0
            if reward >= min_reward:
                traj_rec = {
                    "id": f"shopsim_{setting}_{rec['task_id']}",
                    "task_id": rec["task_id"],
                    "reward": reward,
                    "reward_detail": rec.get("reward_detail", {}),
                    "agent": rec.get("agent", ""),
                    "n_turns": rec.get("state", {}).get("turn", 0),
                    "messages": conversation_to_messages(rec.get("conversation", [])),
                }
                fr.write(json.dumps(traj_rec, ensure_ascii=False) + "\n")
                n_traj += 1

    return {"tasks": n_task, "trajectories": n_traj,
            "tasks_file": tasks_path, "trajectories_file": traj_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="multi_standard",
                        help="outputs/ 下的 setting 目录名")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--min-reward", type=float, default=0.9,
                        help="轨迹保留阈值；1.0=只留全对轨迹（拒绝采样）")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "exports"))
    args = parser.parse_args()

    stats = export(args.setting, args.tag, args.min_reward, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
