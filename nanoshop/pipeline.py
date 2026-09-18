"""Seven-stage orchestration pipeline for the shopping agent.

    Guardrails -> NLU -> Retrieve -> Verify -> Reflect -> Generate -> Memory

Execution model: the pipeline runs once per decision step. Guardrails
parses the raw observation (page state, candidates, buttons) into
``AgentState``; NLU extracts intent/slots/rewritten-query from the latest
shopper message (LLM); Retrieve/Verify are environment-side stages that
record searches and per-product attribute checks; Reflect+Generate is a
single fused LLM call that decides the next action *and* produces the
user-facing text; Memory maintains cross-turn facts and nano-cc-style
dialogue compaction.

Ablation switches (for the消融 experiments): ``no_nlu``, ``no_verify``,
``no_reflect``, ``no_memory``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLM, parse_json_loose
from .state import AgentState, ProductView
from .trace import PipelineTrace, TraceTurn

ROLE_ENV = "Env"
ROLE_SHOPPER = "Shopper"

MAX_TURNS_DEFAULT = 40
MEMORY_COMPRESS_AFTER_TURNS = 12  # nano-cc style thresholded compaction

NLU_SYSTEM = """你是电商导购场景的信息抽取引擎。根据对话中用户的最新消息，抽取结构化购物需求。
只输出 JSON，不要输出其他内容。字段：
{"intent": "clarify|search|compare|verify|recommend|buy|chitchat",
 "slots": {"category": "商品类目，如 乳胶枕", "attributes": ["属性关键词列表"], "option": "规格/款式偏好，如 满天星", "budget": "预算，如 1000元以下", "scenario": "使用场景，如 5岁儿童"},
 "rewritten_query": "把用户需求改写成最适合电商搜索的查询词（仅关键词，去掉口语）",
 "new_facts": ["本轮用户透露的新事实（可空）"]}"""

DECISION_SYSTEM = """你是一个智能购物助手Agent，正在帮用户在电商环境中购买最合适的商品。

## 你的决策信息（系统维护）
- 用户需求槽位：{slots}
- 已搜索关键词：{searches}
- 已浏览核验的商品：{views}
- 证据状态：{evidence}
- 已知用户事实：{facts}

## 动作格式（只输出 JSON）
{{"thought": "简要思考",
 "action_type": "ask_user" 或 "interact_with_env",
 "action_content": "ask_user时为对用户说的话；interact时为环境动作"}}

环境动作格式：
- search[关键词]：搜索商品（仅搜索页可用）
- click[值]：点击按钮/商品asin/规格选项，值必须来自当前可点击按钮列表
- click[buy now]：购买当前商品（必须已选规格）

## 决策原则
1. 需求信息缺口大（类目/属性/预算不明）→ ask_user 澄清；模糊需求通常要问2-4次才能拼齐
2. 信息基本明确 → 搜索；候选不足或属性不符 → 换关键词重搜
3. 看到高潜力商品 → 进详情页核验属性（点 asin，再看 attributes）
4. 核验通过且规格匹配 → 选规格 → 购买前用 ask_user 向用户确认商品和规格 → 用户同意后 click[buy now]
5. 购买前确信：商品属性覆盖用户所有需求、价格在预算内、规格已选对
6. 轮次有限（{max_turns}轮），不要过度澄清；对话快结束时果断购买最合适商品
7. 严禁重复执行完全相同的动作；若环境无变化说明动作无效，必须换动作
8. 已选规格后不要再点规格，直接进入购买确认流程"""

MEMORY_COMPRESS_SYSTEM = """压缩以下购物对话历史，保留：用户需求要点、已确认信息、已看商品结论、当前进展。输出不超过150字的事实清单（分号分隔），不要输出其他内容。"""


class ShoppingPipeline:
    """One instance per task; drives the seven stages each decision step."""

    def __init__(
        self,
        llm: LLM,
        max_turns: int = MAX_TURNS_DEFAULT,
        ablations: Optional[Dict[str, bool]] = None,
    ):
        self.llm = llm
        self.max_turns = max_turns
        self.ablations = ablations or {}
        self.state = AgentState()
        self.trace = PipelineTrace(task_id=-1)
        # rolling dialogue used for NLU/decision prompts (may be compressed)
        self.dialogue: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # turn entry point
    # ------------------------------------------------------------------

    def start_task(
        self,
        task_id: int,
        instruction: str,
        user_persona: Optional[Dict[str, Any]] = None,
        mode: str = "multi",
    ) -> None:
        self.state = AgentState(task_id=task_id, instruction=instruction, user_persona=user_persona)
        self.trace = PipelineTrace(task_id=task_id, mode=mode)
        self.dialogue = []
        self.mode = mode

    def step(self, turn: int, role: str, message: str) -> Tuple[str, str]:
        """Run one pipeline pass. Returns (action_type, action_content)."""
        state = self.state
        state.turn = turn
        t = self.trace.new_turn(turn, role, message)
        trigger = message

        # 1) Guardrails: parse env observation / validate input
        s = t.stage("guardrails", trigger)
        page_info = self._parse_observation(message)
        state.current_page = page_info["page"]
        if page_info["candidates"]:
            state.candidate_asins = page_info["candidates"]
        s.finish(
            output=f"page={page_info['page']} candidates={len(page_info['candidates'])}",
            page=page_info["page"],
        )

        # 2) NLU: slot filling + query rewriting.
        #    - shopper messages: extract from the dialogue
        #    - turn 1 of single mode: extract from the instruction itself
        nlu_input = None
        if role == ROLE_SHOPPER:
            self.dialogue.append({"role": "user", "content": message})
            nlu_input = message
        elif turn == 1 and state.dialogue_turns == 0:
            nlu_input = state.instruction
        s = t.stage("nlu", nlu_input or "-")
        if nlu_input and not self.ablations.get("no_nlu"):
            state.dialogue_turns += 1
            nlu_out = self._run_nlu(nlu_input)
            state.intent = nlu_out.get("intent", state.intent)
            state.update_slots(nlu_out.get("slots", {}))
            if nlu_out.get("rewritten_query"):
                state.rewritten_query = nlu_out["rewritten_query"]
            for fact in nlu_out.get("new_facts", []) or []:
                state.add_user_fact(str(fact))
            s.finish(
                output=json.dumps(nlu_out, ensure_ascii=False)[:200],
                intent=state.intent,
                n_slots=len(state.slots.get("attributes", [])),
                prompt_tokens=0,
            )
        else:
            s.finish(output="skipped")

        # 3) Retrieve bookkeeping happens when the action executes (below);
        #    here we just decide whether a (re)search is warranted.
        s = t.stage("retrieve", state.rewritten_query)
        s.finish(output=f"searches={len(state.search_history)}", n_searches=len(state.search_history))

        # 4) Verify: parse product/attribute pages seen in this observation
        s = t.stage("verify", page_info["item_text"][:160])
        if not self.ablations.get("no_verify"):
            self._run_verify(page_info)
            # an attributes sub-page after click[attributes] counts as an
            # attribute check for the product we last opened
            if page_info.get("page") == "sub" and "attribute" in page_info.get("item_text", "")[:200].lower():
                last_view = state.viewed_products.get(getattr(self, "_last_item_asin", "") or "")
                if last_view is not None:
                    last_view.attributes_checked = True
            n_verified = sum(1 for v in state.viewed_products.values() if v.attributes_checked)
            s.finish(output=f"viewed={len(state.viewed_products)} attr_checked={n_verified}")
        else:
            s.finish(output="ablated")

        # 5+6) Reflect + Generate: fused decision LLM call
        s = t.stage("reflect", f"intent={state.intent} evidence={state.evidence_sufficient}")
        action_type, action_content, pt, ct = self._run_decision(trigger, page_info)
        s.finish(
            output=action_content[:160],
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        self.trace.add_tokens(pt, ct)
        t.stage("generate", action_content[:160]).finish(
            output=action_content[:160],
            fused_with="reflect",
        )

        # 7) Memory: facts already updated in NLU; thresholded compaction here
        s = t.stage("memory", f"turns={len(self.dialogue)}")
        if not self.ablations.get("no_memory"):
            compressed = self._maybe_compress(turn)
            s.finish(output=compressed or "no-op", compressed=bool(compressed))
        else:
            s.finish(output="ablated")

        if action_type == "ask_user":
            self.dialogue.append({"role": "assistant", "content": action_content})

        t.finish(action_type, action_content)
        return action_type, action_content

    # ------------------------------------------------------------------
    # stage implementations
    # ------------------------------------------------------------------

    def _run_nlu(self, message: str) -> Dict[str, Any]:
        history = self._dialogue_text(max_messages=8)
        user = (f"对话历史：\n{history}\n\n最新用户消息：{message}" if history else f"用户消息：{message}")
        if self.state.user_persona:
            persona = json.dumps(self.state.user_persona, ensure_ascii=False)[:800]
            user = f"用户画像（参考，勿直接当成已说出的需求）：{persona}\n\n{user}"
        resp = self.llm.chat(
            [
                {"role": "system", "content": NLU_SYSTEM},
                {"role": "user", "content": user},
            ]
        )
        self.trace.add_tokens(resp.prompt_tokens, resp.completion_tokens)
        return parse_json_loose(resp.content)

    def _run_decision(self, trigger: str, page_info: Dict[str, Any]) -> Tuple[str, str, int, int]:
        state = self.state
        if self.ablations.get("no_reflect"):
            evidence_line = "（消融模式：不提供证据状态）"
            views_line = "（消融模式：不提供浏览记录）"
        else:
            evidence_line = (
                f"充分={state.evidence_sufficient}, 缺失={state.missing_info or '无'}"
            )
            views_line = self._views_text()

        sys_prompt = DECISION_SYSTEM.format(
            slots=json.dumps(state.slots, ensure_ascii=False),
            searches=", ".join(state.search_history[-5:]) or "无",
            views=views_line,
            evidence=evidence_line,
            facts="; ".join(state.known_user_facts[-8:]) or "无",
            max_turns=self.max_turns,
        )
        if state.selected_option:
            sys_prompt += f"\n- 已选规格：{state.selected_option}（不要重复选择）"
        if state.confirmed_with_user:
            sys_prompt += "\n- 用户已确认购买（可直接 click[buy now]）"
        if getattr(self, "mode", "multi") == "single":
            if state.user_persona:
                sys_prompt += (
                    "\n\n注意：当前为单轮模式（无用户交互），任务指令较简略，"
                    "请结合用户画像推断完整需求，禁止 ask_user，直接搜索并购买。"
                )
            else:
                sys_prompt += (
                    "\n\n注意：当前为单轮模式，用户需求已在任务指令中完整给出，"
                    "禁止 ask_user，直接根据指令搜索并购买。"
                )
        # dead-loop guard: surface recent repeated actions to the LLM
        recent = state.last_actions[-3:]
        if len(recent) >= 2 and len(set(recent)) == 1:
            sys_prompt += (
                f"\n- 警告：动作「{recent[-1]}」已连续重复{len(recent)}次且无效，"
                "你必须改变策略（换动作/换商品/向用户澄清/直接购买）"
            )
        if state.user_persona and not self.ablations.get("no_memory"):
            persona = json.dumps(state.user_persona, ensure_ascii=False)[:800]
            sys_prompt += f"\n\n用户个人画像（用于个性化推荐）：{persona}"

        if state.compressed_summary:
            sys_prompt += f"\n\n历史对话摘要：{state.compressed_summary}"

        user_prompt = f"[{ 'Shopper' if 'Shopper' in trigger[:20] else 'Env' }] 观察/消息：\n{trigger[:2500]}"

        resp = self.llm.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = parse_json_loose(resp.content)
        action_type = parsed.get("action_type", "interact_with_env")
        action_content = parsed.get("action_content", "")
        if action_type not in ("ask_user", "interact_with_env") or not action_content:
            # fall back to regex on raw text
            m = re.search(r"(ask_user|interact_with_env)[:：]?\s*(.+)", resp.content, re.S)
            if m:
                action_type = "ask_user" if "ask" in m.group(1) else "interact_with_env"
                action_content = m.group(2).strip()
            else:
                action_type, action_content = "interact_with_env", "search[乳胶枕]"

        # bookkeeping for Retrieve stage stats
        if action_type == "interact_with_env" and action_content.startswith("search["):
            q = action_content[len("search[") : action_content.rfind("]")]
            if q and q not in self.state.search_history:
                self.state.search_history.append(q)
        return action_type, action_content, resp.prompt_tokens, resp.completion_tokens

    def _run_verify(self, page_info: Dict[str, Any]) -> None:
        """Parse item/attribute page content into ProductView records."""
        state = self.state
        if page_info["page"] != "item":
            return
        item_text = page_info["item_text"]
        # match viewed item to a candidate asin via title lookup
        title = page_info.get("title", "")
        asin = page_info.get("asin") or self._match_asin_by_title(title)
        if not asin:
            return
        view = state.viewed_products.get(asin) or ProductView(asin=asin)
        self._last_item_asin = asin
        view.title = title or view.title
        if page_info.get("price"):
            view.price = page_info["price"]
        view.options = page_info.get("options", view.options)
        view.attributes_checked = "Attributes" in item_text or "属性" in item_text
        view.attribute_text = item_text[:600]
        state.viewed_products[asin] = view
        # simple verdict: required attributes present in page text?
        wanted = state.slots.get("attributes", []) or []
        if wanted:
            hits = sum(1 for a in wanted if a and a in item_text)
            view.verdict = "match" if hits >= len(wanted) * 0.6 else ("partial" if hits else "mismatch")
            view.verified = True

    def _maybe_compress(self, turn: int) -> str:
        if len(self.dialogue) < MEMORY_COMPRESS_AFTER_TURNS:
            return ""
        if turn - self.state.last_memory_compress_turn < 6:
            return ""
        convo = "\n".join(f"{m['role']}: {m['content'][:120]}" for m in self.dialogue[:-6])
        if not convo.strip():
            return ""
        resp = self.llm.chat(
            [
                {"role": "system", "content": MEMORY_COMPRESS_SYSTEM},
                {"role": "user", "content": convo[:4000]},
            ]
        )
        self.trace.add_tokens(resp.prompt_tokens, resp.completion_tokens)
        summary = resp.content.strip()
        if summary:
            self.state.compressed_summary = summary
            # keep only the most recent exchanges; older ones are now summarised
            self.dialogue = self.dialogue[-6:]
            self.state.last_memory_compress_turn = turn
        return summary

    # ------------------------------------------------------------------
    # observation parsing (Guardrails helper)
    # ------------------------------------------------------------------

    def _parse_observation(self, message: str) -> Dict[str, Any]:
        """Extract page type, candidate asins, buttons, item info from the
        text observation returned by the environment."""
        info: Dict[str, Any] = {
            "page": "search",
            "candidates": [],
            "buttons": [],
            "item_text": "",
            "title": "",
            "price": "",
            "options": [],
            "asin": None,
        }
        if "可点击的按钮:" in message:
            btn_part = message.split("可点击的按钮:")[-1].strip()
            try:
                info["buttons"] = json.loads(btn_part)
            except json.JSONDecodeError:
                info["buttons"] = []
        body = message.split("搜索功能是否可用:")[0]

        if "Total results" in body or "[SEP] Next >" in body.split("Instruction:")[-1][:400]:
            info["page"] = "results"
            asins = re.findall(r"\[SEP\] (\d{6,16}) \[SEP\]", body)
            info["candidates"] = list(dict.fromkeys(asins))
        elif "Buy Now" in body or "buy now" in info["buttons"]:
            info["page"] = "item"
            info["item_text"] = body
            m = re.search(r"\[SEP\] ([^[\]]{6,80}) \[SEP\] 价格: ([\d. to]+)", body)
            if m:
                info["title"], info["price"] = m.group(1).strip(), m.group(2).strip()
            # options are the radio-like buttons that are not navigation
            nav = {"back to search", "< prev", "next >", "buy now", "description", "features", "reviews", "attributes"}
            info["options"] = [b for b in info["buttons"] if b not in nav]
            if not info["asin"]:
                info["asin"] = self._match_asin_by_title(info["title"])
        elif "Attributes" in body or "attribute" in body[:200]:
            info["page"] = "sub"
            info["item_text"] = body
        return info

    def _match_asin_by_title(self, title: str) -> Optional[str]:
        if not title:
            return None
        env = getattr(self, "env", None)
        if env is None:
            return None
        for asin, prod in env._shared["product_item_dict"].items():
            if prod.get("Title", "") == title:
                return asin
        return None

    # ------------------------------------------------------------------

    def _views_text(self) -> str:
        parts = []
        for v in list(self.state.viewed_products.values())[-5:]:
            parts.append(
                f"{v.asin}|{v.title[:20]}|价格{v.price}|核验:{v.verdict or '未'}"
            )
        return "; ".join(parts) or "无"

    def _dialogue_text(self, max_messages: int = 8) -> str:
        recent = self.dialogue[-max_messages:]
        return "\n".join(f"{m['role']}: {m['content'][:150]}" for m in recent)
