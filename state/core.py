"""状态空间建模：State / Action / 转移规则 / 终止条件。

建模要点（对应文档 2.4 接口契约 + 报告"搜索建模"章节）
-----------------------------------------------------
* 节点 ≠ 条款。ACE 每个场景只给 1~4 条 evidence，若以条款为状态，深度上限 3，
  多跳无从谈起。因此状态在 **facet（条款内语义单元）** 上累积：
  OBL(义务) / EXC(例外) / COND(条件) / DEF(定义) / SCOPE(引用) / REMEDY(救济)。
* 状态 = 已审查条款集合 + 已使用边（带类型）+ 深度 + 标签假设。
* 动作 = 沿一条**带类型的边**走到下一个 facet（法律推理的每一步都有类型：
  "去看例外" "去看定义" "去看被引用的条款"）。
* 转移：新状态并入目标 facet 所属条款，边记入 used_edges，深度 +1。
* 终止：满足"证据充分性"三条（见 `is_goal`）或触顶/无后继。

为什么终止条件不是"置信度够高就停"：单条款场景里模型往往在第一跳就很自信，
但那只是读了主句、没看例外。证据充分性把"必须解开例外"写进目标测试，才让
2~4 跳的搜索真正必要 —— 这同时也解释了后续 BFS 在浅层场景里为何不落后。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from data.clause_graph import (
    E_CONFLICTS, E_EXCEPTIONS, ROLE_COND, ROLE_EXC, ScenarioGraph,
)
from data.features import BREACH_CUES_CJK, SCENARIO_BREACH_CUES

UNDETERMINED = "undetermined"
LABELS = ("Compliant", "Non-Compliant", "Not-Applicable")


@dataclass(frozen=True)
class Action:
    src: str
    edge_type: str
    dst: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.src, self.edge_type, self.dst)

    def to_dict(self) -> dict:
        return {"src": self.src, "edge_type": self.edge_type, "dst": self.dst}


@dataclass(frozen=True)
class State:
    clauses: frozenset[str]
    used_edges: tuple[tuple[str, str, str], ...] = ()
    depth: int = 0
    label_hypothesis: str = UNDETERMINED
    nodes: tuple[str, ...] = ()
    g: float = 0.0

    # ---------------------------------------------------------- 抽象状态键
    def key(self) -> tuple:
        """closed set 的键：条款集合 + 边类型多重集（不是条款 ID 本身）。

        只按条款集合去重会把"经由例外边到达"与"经由补充边到达"判为同一状态，
        丢失掉法律推理最关键的区别，因此把边类型多重集并入键。
        """
        types = tuple(sorted(e[1] for e in self.used_edges))
        return (self.clauses, types)

    def with_action(self, act: Action, dst_clause: str, cost: float = 1.0) -> "State":
        return State(
            clauses=self.clauses | {dst_clause},
            used_edges=self.used_edges + (act.as_tuple(),),
            depth=self.depth + 1,
            label_hypothesis=self.label_hypothesis,
            nodes=self.nodes + (act.dst,),
            g=self.g + cost,
        )

    def path(self) -> list[Action]:
        return [Action(*e) for e in self.used_edges]

    def edge_type_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, t, _ in self.used_edges:
            out[t] = out.get(t, 0) + 1
        return out


@dataclass
class SearchResult:
    answer: str
    path: list[Action]
    expanded_nodes: int
    tree_root: object | None = None
    trace: list[dict] = field(default_factory=list)
    # 扩展字段（不改变文档契约，只增不减）
    sid: str = ""
    score: float = 0.0
    confidence: float = 0.0
    reached_goal: bool = False
    n_clauses_visited: int = 0
    llm_calls: int = 0
    pruned_by_rule: int = 0
    llm_rejected: int = 0
    llm_suspicious: int = 0
    margins: dict = field(default_factory=dict)
    # 推理自然语言解释（llm/explainer.py）。默认 None —— 搜索本身不依赖它，
    # 只有显式调用 explain_result() / API 的 explain=1 时才回填。
    explanation: dict | None = None

    @property
    def steps(self) -> int:
        return len(self.path)


@dataclass
class ProblemConfig:
    max_depth: int = 4
    max_expansions: int = 16
    tau_conf: float = 0.60          # 目标测试的置信度门槛
    min_clause_cover: float = 0.6   # 覆盖率门槛：与最优条款相似度 ≥ 60% 的条款都要看
    sim_floor: float = 0.08
    require_exception_resolution: bool = True


class SearchProblem:
    """把"一个 ACE 场景 + 其条款图"包装成标准搜索问题。"""

    def __init__(self, graph: ScenarioGraph, scenario_text: str,
                 clause_sim: dict[str, float],
                 decision: Callable[[State], tuple[str, float]] | None = None,
                 h_fn: Callable[[State, "SearchProblem"], float] | None = None,
                 rules=None, cfg: ProblemConfig | None = None, sid: str = "",
                 facet_sim: dict[str, float] | None = None):
        self.graph = graph
        self.text = scenario_text
        self.sim = dict(clause_sim)
        self.facet_sim = dict(facet_sim or {})
        self.decision = decision or (lambda s: ("Not-Applicable", 0.34))
        self.h_fn = h_fn or (lambda s, p: 0.0)
        self.rules = rules
        self.cfg = cfg or ProblemConfig()
        self.sid = sid or graph.sid

        self.n_clauses = max(1, graph.n_clauses)
        self.max_sim = max(self.sim.values()) if self.sim else 0.0
        self.relevant = {c for c, v in self.sim.items()
                         if v >= max(self.cfg.sim_floor, self.cfg.min_clause_cover * self.max_sim)}
        if not self.relevant:
            self.relevant = set(self.graph.clause_ids)

        # 例外是否必须被解开：场景出现违约线索 & 图里存在例外/冲突节点
        self.exc_nodes = {n.nid for n in graph.nodes.values()
                          if n.role in (ROLE_EXC, ROLE_COND)}
        self.has_conflict = any(e.etype == E_CONFLICTS
                                for es in graph.out.values() for e in es)
        self.breach_flagged = bool(SCENARIO_BREACH_CUES.search(scenario_text)
                                   or BREACH_CUES_CJK.search(scenario_text))
        self.needs_exc = (self.cfg.require_exception_resolution and self.breach_flagged
                          and bool(self.exc_nodes or self.has_conflict))

        self._cache: dict[tuple, list] = {}

    # ------------------------------------------------------------ 基础
    def initial_state(self) -> State:
        return State(clauses=frozenset(), used_edges=(), depth=0,
                     label_hypothesis=UNDETERMINED, nodes=(), g=0.0)

    def succ_actions(self, state: State) -> list[Action]:
        """当前状态下所有几何可行动作（尚未过规则掩码）。

        注意：**不禁止回到已访问过的条款**。理由与 closed set 的设计一致——
        同一组条款经由"例外边"与经由"补充边"到达，是两条不同的推理路径，
        法律含义不同（前者是"义务被豁免"，后者是"义务被补充"）。因此只有
        立即自环被剪掉，真正的去重交给 State.key()（含边类型多重集）。
        这一条决定了状态空间的真实规模：若按"节点不可重访"剪枝，多条款场景
        的可达状态常常不足 10 个，BFS 与 A* 会退化成同一件事。
        """
        if state.depth >= self.cfg.max_depth:
            return []
        if not state.nodes:
            return [Action(src="ROOT", edge_type="ENTRY", dst=nid)
                    for nid in sorted(self.graph.entry)]
        last = state.nodes[-1]
        acts: list[Action] = []
        seen: set[tuple[str, str, str]] = set()
        for nid in state.nodes:
            for e in self.graph.actions(nid):
                t = (e.src, e.etype, e.dst)
                if t in seen:
                    continue
                if e.dst == last and e.src == nid:      # 立即回退：无信息增益
                    continue
                seen.add(t)
                acts.append(Action(e.src, e.etype, e.dst))
        return acts

    def allowed_actions(self, state: State) -> tuple[list[Action], list[Action]]:
        """返回 (允许, 被规则剪掉)。规则引擎为 None 时不做剪枝。"""
        acts = self.succ_actions(state)
        if self.rules is None:
            return acts, []
        return self.rules.mask(state, acts, self)

    def apply(self, state: State, act: Action) -> State:
        if act.src == "ROOT":
            n = self.graph.node(act.dst)
            return State(clauses=frozenset({n.clause}),
                         used_edges=(("ROOT", "ENTRY", act.dst),),
                         depth=1,
                         label_hypothesis=state.label_hypothesis,
                         nodes=(act.dst,),
                         g=1.0)
        n = self.graph.node(act.dst)
        return state.with_action(act, n.clause, cost=1.0)

    def expand(self, state: State) -> tuple[list[tuple[Action, State]], list[Action]]:
        """一次掩码，同时返回（可行后继, 被剪动作）。

        为什么要有这个入口：可视化的交付要求"节点颜色区分成功路径 / 冗余 /
        LLM 误判 / 规则剪枝"，而被剪的分支根本不会进入后继列表。若在 trace 之外
        再调一次 `allowed_actions()` 去取被剪动作，规则引擎的 `pruned` 计数会翻倍
        （主表的剪枝列直接失真）。所以只在这里调用一次，两个消费者共用结果。
        """
        acts, pruned = self.allowed_actions(state)
        return [(a, self.apply(state, a)) for a in acts], pruned

    def successors(self, state: State) -> list[tuple[Action, State]]:
        return self.expand(state)[0]

    # ------------------------------------------------------------ 目标 / 终止
    def coverage_ok(self, state: State) -> bool:
        return self.relevant.issubset(state.clauses)

    def exception_ok(self, state: State) -> bool:
        if not self.needs_exc:
            return True
        return bool(self.exc_nodes & set(state.nodes))

    def evaluate(self, state: State) -> tuple[str, float]:
        return self.decision(state)

    def is_goal(self, state: State) -> bool:
        """证据充分性：覆盖了所有强相关条款 + 例外已解 + 置信度过门槛。"""
        if state.depth == 0:
            return False
        if not (self.coverage_ok(state) and self.exception_ok(state)):
            return False
        _, conf = self.evaluate(state)
        return conf >= self.cfg.tau_conf

    def is_terminal(self, state: State) -> bool:
        return (state.depth >= self.cfg.max_depth
                or self.is_goal(state)
                or not self.succ_actions(state))

    def terminal_score(self, state: State) -> float:
        """终局打分：置信度 + 证据覆盖奖励。MCTS 的 reward 与"最优终局"选择都用它。"""
        label, conf = self.evaluate(state)
        cov = len(state.clauses) / self.n_clauses
        bonus = 0.10 if self.exception_ok(state) else 0.0
        return float(min(1.0, 0.75 * conf + 0.15 * cov + bonus))

    def heuristic(self, state: State) -> float:
        return float(self.h_fn(state, self))


def trace_row(state: State, act: Action | None, g: float, h: float,
              label: str, conf: float, extra: dict | None = None) -> dict:
    """一行搜索轨迹。字段对齐文档 §2.4：

    `g` / `h` / `f` / `llm_reward` / `clf_score` / `rule_mask`。

    前三个在这里写；后三个由内核通过 `extra` 注入（`search.common.aux_fields`），
    因为只有内核手里才有融合器与规则引擎。
    """
    row = {
        "action": None if act is None else act.to_dict(),
        "depth": state.depth,
        "g": round(g, 4), "h": round(h, 4), "f": round(g + h, 4),
        "label": label, "conf": round(conf, 4),
        "clauses": sorted(state.clauses),
    }
    if extra:
        row.update(extra)
    return row
