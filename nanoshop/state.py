"""AgentState: cross-turn structured memory for the shopping agent.

Extends the nano-cc session concept with domain slots: the seven-stage
pipeline reads/writes this state every turn; the Memory stage persists
confirmed slots across turns and compresses dialogue history past a
threshold (mirroring nano-cc's layered context compaction).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProductView:
    """A product the agent has opened and (optionally) verified."""

    asin: str
    title: str = ""
    price: str = ""
    options: List[str] = field(default_factory=list)
    attributes_checked: bool = False
    attribute_text: str = ""
    selected_option: Optional[str] = None
    verified: bool = False
    verdict: str = ""  # match / mismatch / partial, set by Verify


@dataclass
class AgentState:
    """Everything the pipeline needs to know about the current task."""

    task_id: int = -1
    instruction: str = ""
    user_persona: Optional[Dict[str, Any]] = None

    # --- NLU stage output ---
    intent: str = "explore"  # clarify / search / compare / verify / recommend / buy
    slots: Dict[str, Any] = field(
        default_factory=lambda: {
            "category": "",      # e.g. 乳胶枕
            "attributes": [],    # e.g. ["泰国进口", "天然乳胶"]
            "option": "",        # e.g. 满天星
            "budget": "",        # e.g. 1000元以下
            "scenario": "",      # e.g. 5岁儿童
        }
    )
    rewritten_query: str = ""

    # --- Retrieve / Verify stage output ---
    search_history: List[str] = field(default_factory=list)
    candidate_asins: List[str] = field(default_factory=list)
    viewed_products: Dict[str, ProductView] = field(default_factory=dict)
    current_page: str = "search"  # search / results / item / sub

    # --- Reflect stage output ---
    evidence_sufficient: bool = False
    missing_info: List[str] = field(default_factory=list)
    reflect_decision: str = ""  # requery / clarify / verify / recommend / buy

    # --- Memory stage ---
    dialogue_turns: int = 0
    compressed_summary: str = ""
    known_user_facts: List[str] = field(default_factory=list)
    last_memory_compress_turn: int = 0

    # --- accounting ---
    turn: int = 0
    started_at: float = field(default_factory=time.time)
    last_actions: List[str] = field(default_factory=list)
    selected_option: Optional[str] = None
    confirmed_with_user: bool = False

    def update_slots(self, new_slots: Dict[str, Any]) -> List[str]:
        """Merge newly-extracted slots into state; returns changed keys."""
        changed = []
        for key, value in new_slots.items():
            if not value:
                continue
            if key == "attributes":
                merged = list(dict.fromkeys(self.slots.get("attributes", []) + list(value)))
                if merged != self.slots.get("attributes"):
                    self.slots["attributes"] = merged
                    changed.append(key)
            elif isinstance(value, str) and value != self.slots.get(key):
                self.slots[key] = value
                changed.append(key)
        return changed

    def add_user_fact(self, fact: str) -> None:
        if fact and fact not in self.known_user_facts:
            self.known_user_facts.append(fact)

    def snapshot(self) -> Dict[str, Any]:
        """Serializable view of the state (goes into the trace / output)."""
        return {
            "task_id": self.task_id,
            "turn": self.turn,
            "intent": self.intent,
            "slots": self.slots,
            "rewritten_query": self.rewritten_query,
            "search_history": self.search_history,
            "candidate_asins": self.candidate_asins[:10],
            "viewed_products": {
                asin: {
                    "title": v.title,
                    "price": v.price,
                    "attributes_checked": v.attributes_checked,
                    "selected_option": v.selected_option,
                    "verified": v.verified,
                    "verdict": v.verdict,
                }
                for asin, v in self.viewed_products.items()
            },
            "evidence_sufficient": self.evidence_sufficient,
            "missing_info": self.missing_info,
            "reflect_decision": self.reflect_decision,
            "known_user_facts": self.known_user_facts,
            "compressed_summary": self.compressed_summary[:300],
            "n_searches": len(self.search_history),
            "n_views": len(self.viewed_products),
        }
