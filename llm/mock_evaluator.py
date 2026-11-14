"""LLM 评估器的 mock 实现（A 侧离线跑 A2 / H 用）。

用途：在 B 同学的真实 `llm_client` 就位前，让"LLM 参与搜索"这条链路的
**接口、开销、失效模式**都可运行、可测量。它严格实现文档 2.4 的 Evaluator 协议：

    action_valid(state, action, question) -> (bool, float)
    path_reward(path, question)          -> float
    success_prob(state, question)        -> float

它**从不输出合规标签**（协议里就没有这个出口），只给节点/路径打分 —— 与真实 LLM
的职责边界一致。

`hallucination_rate` 是刻意留出的**受控扰动**：以概率 p 用一个"看似合理但错误"的
奖励替换真实奖励（例如对语义相近却不适用的条款给高分）。主表实验用 p=0；报告里
单独做 p ∈ {0, 0.1, 0.2, 0.3} 的敏感性曲线，用来量化"LLM 幻觉如何让搜索失效"。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from data.clause_graph import E_CONFLICTS, E_DEPENDS_ON, E_EXCEPTIONS, ROLE_EXC
from state.core import Action, State


@dataclass
class MockLLMEvaluator:
    problem: object = None
    hallucination_rate: float = 0.0
    seed: int = 0
    calls: int = 0
    rejects: int = 0
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------ 动作合法性
    def action_valid(self, state: State, action: Action, question: str = "") -> tuple[bool, float]:
        self.calls += 1
        p = self.problem
        role = p.graph.node(action.dst).role
        dst_sim = p.sim.get(p.graph.node(action.dst).clause, 0.0)

        conf = 0.5
        valid = True
        if role == ROLE_EXC and dst_sim >= 0.15:
            conf = 0.85                      # "去看例外"几乎总是合法
        elif action.edge_type in (E_EXCEPTIONS, E_CONFLICTS):
            conf = 0.75
        elif action.edge_type == E_DEPENDS_ON and state.depth >= 3:
            conf = 0.35                      # 引用链过深处开始可疑
            valid = conf >= 0.4
        elif dst_sim < 0.08:
            conf = 0.25
            valid = False
        elif dst_sim > 0.35:
            conf = 0.8

        if self._rng.random() < self.hallucination_rate:
            conf = 1.0 - conf * 0.3          # 幻觉：把判断反转成自信
        if not valid:
            self.rejects += 1
        return valid, conf

    # ------------------------------------------------------------ 路径奖励
    def path_reward(self, path: list[Action], question: str = "") -> float:
        self.calls += 1
        p = self.problem
        if not path:
            return 0.0
        base = 0.0
        for a in path:
            n = p.graph.node(a.dst)
            s = p.sim.get(n.clause, 0.0)
            w = 1.2 if n.role == ROLE_EXC else 1.0
            base += w * s
        r = base / max(1, len(path))
        if any(a.edge_type in (E_EXCEPTIONS, E_CONFLICTS) for a in path):
            r += 0.12
        r = max(0.0, min(1.0, r * 2.2))
        if self._rng.random() < self.hallucination_rate:
            r = 1.0 - r * 0.4
        return round(r, 4)

    # ------------------------------------------------------------ 成功概率
    def success_prob(self, state: State, question: str = "") -> float:
        self.calls += 1
        p = self.problem
        label, conf = p.evaluate(state)
        cov = len(state.clauses) / max(1, p.n_clauses)
        s = 0.65 * conf + 0.25 * cov + (0.1 if p.exception_ok(state) else 0.0)
        if self._rng.random() < self.hallucination_rate:
            s = 1.0 - s * 0.5
        return round(max(0.0, min(1.0, s)), 4)
