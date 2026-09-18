"""Evaluation runner: samples tasks from a ShopSimulator setting and runs the
seven-stage agent over them, checkpointing one JSON per task.

Settings: (single|multi) x (standard|persona). Task IDs follow the official
protocol (index into the goals list). Supports ablations via --ablate.

Usage:
    python -m nanoshop.eval.run_eval --mode multi --persona false \
        --n 30 --tag mytag [--ablate no_nlu] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(dotenv_path: Optional[str] = None) -> None:
    path = Path(dotenv_path or PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def get_llm_config(role: str) -> Dict[str, str]:
    return {
        "api_key": os.environ.get(f"{role}_API_KEY", os.environ.get("LLM_API_KEY", "")),
        "base_url": os.environ.get(f"{role}_BASE_URL", os.environ.get("LLM_BASE_URL", "")),
        "model": os.environ.get(f"{role}_MODEL", os.environ.get("LLM_MODEL", "deepseek-chat")),
    }


def sample_task_ids(n: int, seed: int = 42, total_hint: Optional[int] = None) -> List[int]:
    rng = random.Random(seed)
    if total_hint:
        pool = list(range(total_hint))
    else:
        pool = list(range(1459))  # official multi/standard test slice
    n = min(n, len(pool))
    return sorted(rng.sample(pool, n))


def run_one(args_tuple) -> str:
    """Worker entry: runs a list of task IDs in one process (own env copy)."""
    (
        mode,
        persona,
        task_ids,
        output_dir,
        tag,
        ablations,
        max_turns,
        worker_idx,
    ) = args_tuple

    logging.basicConfig(level=logging.WARNING)
    from ..agent import ShoppingAgent
    from ..env_bridge import LocalShopEnv
    from ..llm import LLM
    from ..shopper import ShopperSimulator

    agent_cfg = get_llm_config("AGENT")
    shopper_cfg = get_llm_config("SHOPPER")

    env = LocalShopEnv(if_persona=persona)
    agent_llm = LLM(
        model=agent_cfg["model"],
        api_key=agent_cfg["api_key"],
        base_url=agent_cfg["base_url"],
        temperature=0.0,
        max_tokens=1024,
    )
    shopper_llm = LLM(
        model=shopper_cfg["model"],
        api_key=shopper_cfg["api_key"],
        base_url=shopper_cfg["base_url"],
        temperature=0.7,
        max_tokens=512,
    )
    shopper = ShopperSimulator(shopper_llm) if mode == "multi" else None

    agent = ShoppingAgent(
        env=env,
        llm=agent_llm,
        mode=mode,
        max_turns=max_turns,
        ablations=ablations,
        shopper=shopper,
    )

    setting = f"{mode}_{'persona' if persona else 'standard'}"
    out_dir = os.path.join(output_dir, setting, tag)
    os.makedirs(out_dir, exist_ok=True)
    done_ids = {
        int(f.split(".")[0])
        for f in os.listdir(out_dir)
        if f.endswith(".json") and f.split(".")[0].isdigit()
    }

    for tid in task_ids:
        if tid in done_ids:
            continue
        try:
            res = agent.run_task(tid, out_dir, model_tag="")
            logger.warning(
                "[w%d] task %s reward=%.3f turns=%d tokens=%d",
                worker_idx,
                tid,
                res.get("reward", 0),
                res.get("state", {}).get("turn", 0),
                res.get("tokens", {}).get("prompt", 0) + res.get("tokens", {}).get("completion", 0),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("task %s crashed: %s", tid, e)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--persona", default="false")
    parser.add_argument("--n", type=int, default=30, help="number of sampled tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--ablate", nargs="*", default=[], help="no_nlu no_verify no_reflect no_memory")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1, help="processes (each loads its own env)")
    parser.add_argument("--task-ids", default=None, help="explicit comma list, overrides --n")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    load_env()
    persona = args.persona.lower() in ("true", "1", "yes")
    ablations = {k: True for k in args.ablate}
    if ablations:
        args.tag = args.tag + "+" + "+".join(sorted(ablations))

    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = sample_task_ids(args.n, args.seed)

    print(f"setting={args.mode}_{'persona' if persona else 'standard'} tag={args.tag} "
          f"tasks={len(task_ids)} ablations={ablations} workers={args.workers}")

    if args.workers <= 1:
        run_one((args.mode, persona, task_ids, args.output, args.tag, ablations, args.max_turns, 0))
    else:
        import multiprocessing as mp

        chunks = [task_ids[i::args.workers] for i in range(args.workers)]
        jobs = [
            (args.mode, persona, chunk, args.output, args.tag, ablations, args.max_turns, i)
            for i, chunk in enumerate(chunks)
            if chunk
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(jobs)) as pool:
            pool.map(run_one, jobs)

    print("done ->", os.path.join(args.output, f"{args.mode}_{'persona' if persona else 'standard'}", args.tag))


if __name__ == "__main__":
    main()
