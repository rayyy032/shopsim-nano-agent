"""Local in-process bridge to the official ShopSimulator environment.

The official release serves the environment over HTTP (Flask, 20 pre-built
gym envs) and relies on ``pyserini.LuceneSearcher`` (requires JDK 21) plus
legacy torch/gym pins. On a laptop we instead:

1. stub out ``pyserini`` / ``torch`` before importing ``web_agent_site``;
2. swap the Lucene index for a jieba-tokenised BM25 index (rank_bm25),
   keeping the exact ``search``/``doc`` interface used by ``engine.py``;
3. drive a single shared ``WebAgentTextEnv`` in-process, which avoids the
   20x memory blow-up of the Flask deployment.

The public API mirrors the official HTTP protocol (reset / interact /
release) so the agent code above is protocol-compatible with the original
single_eval / multi_eval harnesses.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SHOPSIM_REPO = os.environ.get(
    "SHOPSIM_REPO",
    os.path.join(os.path.dirname(__file__), "..", "..", "ShopSimulator"),
)
SHOP_ENV_ROOT = os.path.join(SHOPSIM_REPO, "shop_env")
DATA_FILE = os.path.join(SHOP_ENV_ROOT, "data", "fine_items_eval_train_all.json")

MAX_HISTORY_LENGTH = 42  # mirrors shop_agent.MAX_HISTORY_LENGTH


# ---------------------------------------------------------------------------
# 1. Dependency stubs: pyserini / torch are unavailable (and unneeded) here.
# ---------------------------------------------------------------------------

class _StubTorch:
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        if name == "zeros":
            return lambda *a, **k: None
        raise AttributeError(f"torch.{name} is stubbed out in local mode")


def _install_stubs() -> None:
    """Inject lightweight stubs for pyserini and torch before web_agent_site
    is imported (its modules import both at top level)."""
    pyserini = sys.modules.setdefault("pyserini", type(sys)("pyserini"))
    search_pkg = sys.modules.setdefault(
        "pyserini.search", type(sys)("pyserini.search")
    )
    lucene_pkg = sys.modules.setdefault(
        "pyserini.search.lucene", type(sys)("pyserini.search.lucene")
    )

    class _UnavailableLuceneSearcher:  # pragma: no cover - never called
        def __init__(self, *a, **k):
            raise RuntimeError(
                "LuceneSearcher is stubbed; BM25Searcher is used instead"
            )

    lucene_pkg.LuceneSearcher = _UnavailableLuceneSearcher
    search_pkg.lucene = lucene_pkg
    pyserini.search = search_pkg
    sys.modules.setdefault("torch", _StubTorch())


def _import_text_env_module():
    """Load ``web_agent_text_env`` from its file path.

    Importing ``web_agent_site.envs.web_agent_text_env`` normally executes
    ``envs/__init__.py``, which drags in selenium + the flask site env that
    we never use in text mode; loading the single module avoids all that.
    """
    import importlib.util

    mod_path = os.path.join(
        SHOP_ENV_ROOT, "web_agent_site", "envs", "web_agent_text_env.py"
    )
    spec = importlib.util.spec_from_file_location(
        "web_agent_site.envs.web_agent_text_env_local", mod_path
    )
    text_env_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = text_env_mod
    spec.loader.exec_module(text_env_mod)
    return text_env_mod


def _load_text_env_class():
    if "WebAgentTextEnv" not in _SHARED:
        _SHARED["WebAgentTextEnv"] = _import_text_env_module().WebAgentTextEnv
    return _SHARED["WebAgentTextEnv"]


# ---------------------------------------------------------------------------
# 2. BM25 search engine with a LuceneSearcher-compatible interface.
# ---------------------------------------------------------------------------

class _Doc:
    """Minimal stand-in for a pyserini DocumentObject."""

    __slots__ = ("_id",)

    def __init__(self, doc_id: str):
        self._id = doc_id

    def raw(self) -> str:
        return json.dumps({"id": self._id}, ensure_ascii=False)


class _Hit:
    __slots__ = ("docid", "score")

    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score


class BM25Searcher:
    """jieba-tokenised BM25 index over product text fields.

    Exposes ``search(query, k)`` / ``doc(docid)`` so that
    ``web_agent_site.engine.engine.get_top_n_product_from_keywords`` works
    unmodified.
    """

    def __init__(self, products: List[Dict[str, Any]]):
        import jieba
        from rank_bm25 import BM25Okapi

        jieba.initialize()
        self._docids: List[str] = []
        corpus: List[List[str]] = []
        for p in products:
            asin = p["asin"]
            text = " ".join(
                str(p.get(field, "") or "")
                for field in ("Title", "category", "query", "BulletPoints")
            )
            attr_text = " ".join(p.get("Attributes", []) or [])
            option_text = " ".join(
                v for opts in (p.get("options", {}) or {}).values() for v in opts
            )
            full = f"{text} {attr_text} {option_text}"
            self._docids.append(asin)
            corpus.append(self._tokenize(full))
        self._index = BM25Okapi(corpus)
        logger.info("BM25 index built over %d products", len(self._docids))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import jieba

        return [t.strip().lower() for t in jieba.cut(text) if t.strip()]

    def search(self, query: str, k: int = 150) -> List[_Hit]:
        scores = self._index.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= k:
                break
            hits.append(_Hit(self._docids[i], float(scores[i])))
        # fall back to head products when nothing matches
        while len(hits) < min(k, len(self._docids)) and not hits:
            hits.append(_Hit(self._docids[len(hits)], 0.0))
        return hits

    def doc(self, docid: str) -> _Doc:
        return _Doc(docid)


# ---------------------------------------------------------------------------
# 3. Patched engine module + shared environment construction.
# ---------------------------------------------------------------------------

_ENGINE_PATCHED = False
_SHARED: Dict[str, Any] = {}


def _patch_engine() -> None:
    """Point ``init_search_engine`` at our BM25 index."""
    global _ENGINE_PATCHED
    if _ENGINE_PATCHED:
        return
    import web_agent_site.engine.engine as engine_mod

    def init_search_engine_bm25(num_products=None):  # noqa: ANN001
        searcher = _SHARED.get("bm25")
        if searcher is None:
            raise RuntimeError("BM25 index not initialised yet")
        return searcher

    engine_mod.init_search_engine = init_search_engine_bm25
    _ENGINE_PATCHED = True


def _ensure_paths() -> None:
    root = os.path.abspath(SHOP_ENV_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def get_shared_state(if_persona: bool = False) -> Dict[str, Any]:
    """Build (once per persona mode) the shared SimServer-side objects:
    products / prices / goals / BM25 index. Loading 23k products takes a few
    seconds and ~1.5GB RAM; doing it once keeps evaluation cheap."""
    key = "persona" if if_persona else "standard"
    if key in _SHARED:
        return _SHARED

    _install_stubs()
    _ensure_paths()
    _patch_engine()

    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    logger.info("Loading products from %s ...", DATA_FILE)
    t0 = time.time()
    all_products, product_item_dict, product_prices, _ = load_products(
        filepath=DATA_FILE, num_products=None, human_goals=0
    )
    logger.info("Loaded %d products in %.1fs", len(all_products), time.time() - t0)

    _SHARED["bm25"] = BM25Searcher(all_products)
    _SHARED["products"] = all_products
    _SHARED["product_item_dict"] = product_item_dict
    _SHARED["product_prices"] = product_prices
    if if_persona:
        # Persona-mode adaptations (the released data file lacks the fields
        # the official persona path expects, see engine/goal.py):
        # 1. keep only items that actually carry a user_persona document --
        #    otherwise the agent would face a vague instruction with no way
        #    to infer the hidden requirement (~80% of items have none);
        # 2. inject `instruction_sample` = full instruction as a fallback,
        #    so the goal's instruction_text stays the concrete requirement.
        #    The shopper simulator receives it (official multi_eval
        #    semantics) while the agent only ever sees instruction_simple
        #    plus the persona document.
        persona_products = [it for it in all_products if it.get("user_persona")]
        for item in persona_products:
            for product in item["instructions"]:
                if "instruction_sample" not in product:
                    product["instruction_sample"] = product["instruction"]
        _SHARED[f"goals_{key}"] = get_goals(
            persona_products, product_prices, if_persona=True
        )
    else:
        _SHARED[f"goals_{key}"] = get_goals(
            all_products, product_prices, if_persona=False
        )
    logger.info("Built %d goals (%s mode)", len(_SHARED[f"goals_{key}"]), key)
    _SHARED[key] = True
    if "server" not in _SHARED:
        _build_shared_server()
    return _SHARED


def _build_shared_server() -> None:
    """Assemble one SimServer instance that reuses the already-loaded
    products / goals / BM25 index, instead of letting each WebAgentTextEnv
    reload everything from disk."""
    import numpy as np

    text_env_mod = _import_text_env_module()
    SimServer = text_env_mod.SimServer

    server = SimServer.__new__(SimServer)
    server.base_url = "http://127.0.0.1:3000"
    server.all_products = _SHARED["products"]
    server.product_item_dict = _SHARED["product_item_dict"]
    server.product_prices = _SHARED["product_prices"]
    server.search_engine = _SHARED["bm25"]
    server.existed_goals = True
    if "goals_standard" in _SHARED:
        server.goals = _SHARED["goals_standard"]
    else:  # persona goals were built first
        server.goals = _SHARED["goals_persona"]
    server.show_attrs = False
    server.shuffle_goals = False
    server.shuffle_num = 20
    server.shift_goals = False
    server.weights = [goal["weight"] for goal in server.goals]
    server.cum_weights = [0] + np.cumsum(server.weights).tolist()
    server.user_sessions = {}
    server.search_time = 0
    server.render_time = 0
    server.sample_time = 0
    server.assigned_instruction_text = None
    _SHARED["server"] = server
    _SHARED["server_mode"] = (
        "standard" if "goals_standard" in _SHARED else "persona"
    )


class LocalShopEnv:
    """In-process equivalent of the official HTTP ``/api/shop_agent`` service.

    ``reset`` / ``interact`` / ``release`` return the same payload shape as
    ``shop_env/shop_agent.py`` so downstream agent code is unchanged.
    """

    def __init__(self, if_persona: bool = False):
        self.if_persona = if_persona
        key = "persona" if if_persona else "standard"
        shared = get_shared_state(if_persona)
        self._shared = shared

        _patch_engine()
        WebAgentTextEnv = _load_text_env_class()

        # Swap the shared server onto the right goal set for this mode.
        if _SHARED.get("server_mode") != key:
            import numpy as np

            _SHARED["server"].goals = _SHARED[f"goals_{key}"]
            _SHARED["server"].weights = [
                g["weight"] for g in _SHARED["server"].goals
            ]
            _SHARED["server"].cum_weights = [
                0
            ] + np.cumsum(_SHARED["server"].weights).tolist()
            _SHARED["server_mode"] = key

        # A single env instance can serve tasks sequentially: each reset(idx)
        # creates a fresh session inside the shared SimServer.
        self._env = WebAgentTextEnv(
            observation_mode="text",
            server=_SHARED["server"],
            split="train",
            num_products=None,
            if_persona=if_persona,
        )
        self._active = False
        self.instruction: Optional[str] = None
        self.task_id: Optional[int] = None

    # -- official protocol ---------------------------------------------------

    def reset(self, task_id: int) -> Dict[str, Any]:
        self._env.reset(idx=task_id)
        self.task_id = task_id
        self._active = True
        info: Dict[str, Any] = {
            "instruction": self._env.instruction_text,
            "instruction_simple": self._env.instruction_simple,
            "goal_options": self._env.goal_options,
            "message": f"Task {task_id} started",
            "env_idx": 0,
            "idx": task_id,
        }
        if getattr(self._env, "user_persona", None) is not None:
            info["user_persona"] = self._env.user_persona
            info["reason_key"] = self._env.reason_key
        if self.if_persona:
            self.instruction = info["instruction_simple"]
        return info

    def interact(self, action_text: str) -> Dict[str, Any]:
        if not self._active:
            raise RuntimeError("environment not initialised; call reset() first")

        normalized = action_text.replace("\\n", "\n")
        self._env.history.append({"role": "assistant", "content": normalized})
        action_str = (
            normalized.split("\nAction: ")[1]
            if "\nAction: " in normalized
            else normalized
        )
        observation, status, _ = self._env.step(action_str)
        done = status["done"]

        available = self._env.get_available_actions()
        clickables = [c for c in available["clickables"] if c != "search"]
        observation = (
            observation
            + f"\n\n搜索功能是否可用: {available['has_search_bar']}"
            + f"\n\n可点击的按钮: {json.dumps(clickables, ensure_ascii=False)}"
        )

        result: Dict[str, Any] = {
            "done": done,
            "reward": status.get("reward", 0),
            "instruction": observation,
            "message": "Continue interaction",
            "env_idx": 0,
            "idx": self._env.session,
            "reward_detail": status.get("reward_detail", {}) if done else {},
            "purchase": status.get("purchase", {}) if done else {},
            "goal": status.get("goal", {}) if done else {},
            "over": len(self._env.history) > MAX_HISTORY_LENGTH or done,
        }
        if self.if_persona and done:
            pass  # observation already reflects the simple instruction
        return result

    def release(self) -> Dict[str, Any]:
        self._active = False
        self.task_id = None
        return {"message": "released"}

    # -- helpers used by component-level evaluation --------------------------

    @property
    def goals(self) -> List[Dict[str, Any]]:
        key = "persona" if self.if_persona else "standard"
        return self._shared[f"goals_{key}"]

    def current_goal(self) -> Dict[str, Any]:
        return self.goals[self.task_id]
