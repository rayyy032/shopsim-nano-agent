"""Metrics: end-to-end scores (official 4-dim reward) + component-level
metrics mined from the pipeline traces (NLU slot F1, query rewrite gold-hit,
clarify rate, verify coverage, token overhead).

Usage:
    python -m nanoshop.eval.metrics --dir outputs/multi_standard/main
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List


# ---------------------------------------------------------------- end-to-end

def end_to_end(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    def dim_rate(key: str) -> float:
        vals = []
        for r in records:
            rd = r.get("reward_detail") or {}
            v = rd.get(key)
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    rewards = [r.get("reward", 0) or 0 for r in records]
    right = [
        1.0
        if (r.get("purchase", {}).get("asin") and r["purchase"]["asin"] == r.get("goal", {}).get("asin"))
        else 0.0
        for r in records
    ]
    # full success: all four dims perfect
    full = []
    for r in records:
        rd = r.get("reward_detail") or {}
        ok_att = rd.get("r_att", 0) == 1
        ok_opt = rd.get("r_option", 0) == 1
        ok_price = bool(rd.get("r_price"))
        ok_type = rd.get("r_type", 0) == 1
        full.append(1.0 if (ok_att and ok_opt and ok_price and ok_type) else 0.0)

    return {
        "n": n,
        "r_loose": round(sum(rewards) / n, 4),
        "r_success": round(sum(full) / n, 4),
        "right_product": round(sum(right) / n, 4),
        "r_type": round(dim_rate("r_type"), 4),
        "r_att": round(dim_rate("r_att"), 4),
        "r_option": round(dim_rate("r_option"), 4),
        "r_price": round(dim_rate("r_price"), 4),
    }


# ------------------------------------------------------------- component

def _fuzz_match(a: str, b: str) -> bool:
    try:
        from thefuzz import fuzz

        return fuzz.token_set_ratio(a, b) > 85
    except Exception:
        return a.strip() == b.strip()


def slot_prf(state: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, float]:
    """NLU slot extraction vs gold attributes (attribute-level P/R/F1)."""
    pred = state.get("slots", {}).get("attributes", []) or []
    gold = goal.get("attributes", []) or []
    if not gold:
        return {}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for g in gold if any(_fuzz_match(g, p) for p in pred))
    fp = len(pred) - sum(1 for p in pred if any(_fuzz_match(p, g) for g in gold))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def component_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    slot_scores: List[Dict[str, float]] = []
    gold_hits: List[float] = []
    clarify_rates: List[float] = []
    verify_cov: List[float] = []
    early_clarify: List[float] = []
    tokens: List[int] = []
    turns_list: List[int] = []
    search_counts: List[int] = []
    latency: List[float] = []

    for r in records:
        state = r.get("state", {}) or {}
        goal = r.get("goal", {}) or {}
        trace = r.get("trace", {}) or {}

        s = slot_prf(state, goal)
        if s:
            slot_scores.append(s)

        # gold product hit among the candidates the agent actually surfaced
        gold_asin = goal.get("asin")
        candidates = set(state.get("candidate_asins", []) or [])
        viewed = set((state.get("viewed_products") or {}).keys())
        if gold_asin:
            gold_hits.append(1.0 if (gold_asin in candidates or gold_asin in viewed) else 0.0)

        # clarify behaviour: ask_user share of agent turns + first-3-turn clarify
        n_ask = sum(1 for t in trace.get("turns", []) if t.get("action_type") == "ask_user")
        n_agent_turns = len(trace.get("turns", []))
        if n_agent_turns:
            clarify_rates.append(n_ask / n_agent_turns)
            early = sum(
                1
                for t in trace.get("turns", [])[:3]
                if t.get("action_type") == "ask_user"
            )
            early_clarify.append(1.0 if early else 0.0)

        # verify coverage before purchase: the agent opened a product-detail
        # sub-page (attributes / features) at least once before deciding —
        # mined from the action history so it works retrospectively on
        # already-written JSONs
        actions = [
            c.get("agent_action", "")
            for c in r.get("conversation", [])
            if isinstance(c, dict)
        ]
        clicked_attrs = any(
            ("click[attributes]" in a) or ("click[features]" in a) for a in actions
        )
        views = state.get("viewed_products", {}) or {}
        attr_checked = any(bool(v.get("attributes_checked")) for v in views.values())
        if r.get("purchase") or views:
            verify_cov.append(1.0 if (clicked_attrs or attr_checked) else 0.0)

        tk = r.get("tokens", {}) or {}
        tokens.append(tk.get("prompt", 0) + tk.get("completion", 0))
        turns_list.append(state.get("turn", 0) or len(trace.get("turns", [])))
        search_counts.append(state.get("n_searches", 0))
        latency.append(r.get("latency_s", 0) or 0)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    out: Dict[str, Any] = {
        "n": n,
        "slot_f1": mean([s["f1"] for s in slot_scores]),
        "slot_precision": mean([s["precision"] for s in slot_scores]),
        "slot_recall": mean([s["recall"] for s in slot_scores]),
        "gold_candidate_hit": mean(gold_hits),
        "clarify_rate": mean(clarify_rates),
        "early_clarify_rate": mean(early_clarify),
        "verify_coverage": mean(verify_cov),
        "avg_turns": mean(turns_list),
        "avg_searches": mean(search_counts),
        "avg_latency_s": mean(latency),
        # NOTE: tokens are LLM-instance lifetime cumulatives (see agent.py);
        # use only for rough cost comparisons across runs of the same size.
        "avg_cum_tokens_per_task": mean(tokens),
        "total_cum_tokens_last": tokens[-1] if tokens else 0,
    }
    if slot_scores:
        out["slot_n"] = len(slot_scores)
    return out


def summarize_dir(d: str) -> Dict[str, Any]:
    records = []
    for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(fp, encoding="utf-8") as f:
                records.append(json.load(f))
        except json.JSONDecodeError:
            continue
    if not records:
        return {}
    # metrics.json may live in the same dir - exclude it
    records = [r for r in records if "task_id" in r]
    summary = {
        "end_to_end": end_to_end(records),
        "component": component_metrics(records),
    }
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="dir with per-task JSONs")
    args = parser.parse_args()
    summary = summarize_dir(args.dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
