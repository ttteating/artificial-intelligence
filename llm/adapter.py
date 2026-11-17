"""真实 LLM 适配层（OpenAI 兼容接口）+ 磁盘缓存 + 调用计数。

⚠️ 归属说明：分工上属于 B 同学。A 同学提供这份实现，是为了让 `--use-llm real`
这条路径可用（而不是把 A2/H 永远锁死在 mock 上）。未配置 `LLM_API_KEY` 时自动
回退到 `MockLLMEvaluator`，并在结果里标记 `llm_mode="mock"`，报告不会把两者混为一谈。

三条硬约束（对应文档 §3.6）：
  1. **禁止输出合规标签**：prompt 明确要求只回 JSON 分数，解析时若出现标签词直接丢弃
  2. **磁盘缓存**：key = sha1(prompt)，同一 (state, action) 不重复付费
  3. **只在语义复杂的边上调用**：EXCEPTIONS / CONFLICTS / DEPENDS_ON + 例外节点
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from data.clause_graph import E_CONFLICTS, E_DEPENDS_ON, E_EXCEPTIONS, ROLE_EXC
from llm.env import load_project_env
from state.core import Action, State

LABEL_WORDS = ("compliant", "non-compliant", "not-applicable", "not applicable")

VALIDATOR_SYS = (
    "You are a legal-reasoning search assistant. You do NOT decide compliance labels. "
    "Given a contract scenario and a candidate move in a clause graph, judge only whether "
    "this move is a legitimate next step of legal reading. "
    'Reply with JSON only: {"valid": true|false, "confidence": 0.0-1.0, "why": "<=15 words"}. '
    "Never output a compliance label such as Compliant / Non-Compliant / Not-Applicable."
)

REWARD_SYS = (
    "You are a legal-reasoning search assistant. You do NOT decide compliance labels. "
    "Score how much this reasoning path contributes to resolving the question, 0.0-1.0. "
    'Reply with JSON only: {"reward": 0.0-1.0, "why": "<=20 words"}. '
    "Never output a compliance label."
)


@dataclass
class LLMClient:
    model: str = "gpt-4o-mini"
    cache_dir: Path = Path("results/cache/llm")
    timeout: float = 20.0
    max_retries: int = 2
    calls: int = 0
    cache_hits: int = 0
    api_key: str | None = None
    base_url: str | None = None
    _client: object | None = field(default=None, repr=False)

    def __post_init__(self):
        load_project_env()
        self.api_key = self.api_key or os.environ.get("LLM_API_KEY")
        self.base_url = self.base_url or os.environ.get("LLM_BASE_URL")
        self.model = os.environ.get("LLM_MODEL", self.model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client_obj(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------ 调用
    def json_call(self, system: str, user: str, tag: str = "") -> dict:
        key = hashlib.sha1(f"{self.model}|{system}|{user}".encode()).hexdigest()
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text(encoding="utf-8"))

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                r = self._client_obj().chat.completions.create(
                    model=self.model, temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                txt = r.choices[0].message.content or "{}"
                obj = json.loads(txt)
                obj = self._sanitize(obj)
                obj["_tag"] = tag
                f.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
                return obj
            except Exception as exc:      # noqa: BLE001 - 网络/配额都可能失败
                last_err = exc
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"LLM call failed: {last_err}")

    @staticmethod
    def _sanitize(obj: dict) -> dict:
        """若模型违规输出标签词，把分数压到最低并标记，避免污染搜索。"""
        s = json.dumps(obj).lower()
        if any(w in s for w in LABEL_WORDS):
            for k in ("confidence", "reward"):
                if k in obj:
                    obj[k] = 0.0
            obj["violation"] = True
        return obj


class LLMEvaluator:
    """把 LLMClient 包装成文档 2.4 的 Evaluator 协议。

    只对语义复杂的边调用（省 token），其余用本地启发式兜底。real-mock 的分界
    在结果里通过 `mode` 字段暴露，实验表会分别标注。
    """

    mode = "real"
    COMPLEX_EDGES = (E_EXCEPTIONS, E_CONFLICTS, E_DEPENDS_ON)

    def __init__(self, client: LLMClient, problem, fallback=None):
        self.client = client
        self.problem = problem
        self.fallback = fallback
        self.calls = 0
        self.rejects = 0

    def _need_llm(self, action: Action) -> bool:
        role = self.problem.graph.node(action.dst).role
        return action.edge_type in self.COMPLEX_EDGES or role == ROLE_EXC

    def _ctx(self, action: Action) -> str:
        g = self.problem.graph
        src = "ROOT" if action.src == "ROOT" else g.node(action.src).text[:400]
        return (f"SCENARIO:\n{self.problem.text[:1200]}\n\n"
                f"FROM:\n{src}\n\nMOVE: {action.edge_type}\n"
                f"TO:\n{g.node(action.dst).text[:400]}")

    def action_valid(self, state: State, action: Action, question: str = "") -> tuple[bool, float]:
        if not self._need_llm(action):
            if self.fallback:
                return self.fallback.action_valid(state, action, question)
            return True, 0.6
        self.calls += 1
        obj = self.client.json_call(VALIDATOR_SYS, self._ctx(action), tag="valid")
        v = bool(obj.get("valid", True))
        c = float(obj.get("confidence", 0.5))
        if not v:
            self.rejects += 1
        return v, c

    def path_reward(self, path: list[Action], question: str = "") -> float:
        if not path:
            return 0.0
        last = path[-1]
        if not self._need_llm(last):
            if self.fallback:
                return self.fallback.path_reward(path, question)
            return 0.5
        self.calls += 1
        obj = self.client.json_call(REWARD_SYS, self._ctx(last), tag="reward")
        return float(obj.get("reward", 0.5))

    def success_prob(self, state: State, question: str = "") -> float:
        if self.fallback:
            return self.fallback.success_prob(state, question)
        return 0.5


def build_evaluator(problem, use_llm: str = "off", hallucination_rate: float = 0.0,
                    seed: int = 0):
    """use_llm: off | mock | real。返回 (evaluator|None, mode_str)。"""
    from llm.mock_evaluator import MockLLMEvaluator

    if use_llm == "off":
        return None, "off"
    if use_llm == "real":
        cli = LLMClient()
        if cli.available:
            return LLMEvaluator(cli, problem), "real"
        print("[llm] 未检测到 LLM_API_KEY，回退 mock")
    return MockLLMEvaluator(problem=problem, hallucination_rate=hallucination_rate,
                            seed=seed), "mock"
