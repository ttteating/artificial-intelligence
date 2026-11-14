"""轻量规则引擎：硬约束掩码 + 风险分。

⚠️ 归属说明：分工上属于 B 同学（"规则专家系统"）。A 同学在这里给的是一份
**可运行的规则骨架**，只为让 B1 / H 条件可跑、并支撑"规则过约束"的失败分析。
接口与文档 2.4 一致：`mask(state, actions) -> list[Action]` 与 `risk(state, action) -> float`。

规则清单（都来自合同审查的领域常识）
----------------------------------
R1 无关扩展阻断：目标条款与场景相似度 < floor 且已覆盖强相关条款 -> 剪
R2 深链截断：DEPENDS_ON 连续链 > max_ref_chain 跳 -> 剪（法律引用很少无限套嵌）
R3 冲突边双证据：CONFLICTS 边要求 src 与 dst 都已在状态里 -> 否则延后
R4 例外未解不下结论：needs_exc 且例外未解时，禁止把动作花在"再去看义务条款"上
   （strict 开关；关掉后只降权不剪枝 —— 这正是"规则过约束"的对照组）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.clause_graph import E_CONFLICTS, E_DEPENDS_ON, ROLE_EXC
from state.core import Action, State


@dataclass
class RuleEngine:
    sim_floor: float = 0.12
    max_ref_chain: int = 2
    strict_exception: bool = True
    enabled: tuple[str, ...] = ("R1", "R2", "R3", "R4")
    # 计数器（每场景新建一个引擎，所以直接累加即可）
    pruned: int = 0
    hits: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 掩码
    def mask(self, state: State, actions: list[Action], problem=None) -> tuple[list[Action], list[Action]]:
        allowed, pruned = [], []
        for a in actions:
            why = self._blocked_by(state, a, problem)
            if why:
                pruned.append(a)
                self.hits[why] = self.hits.get(why, 0) + 1
            else:
                allowed.append(a)
        # 若规则把动作全剪光，退化为不剪（保证搜索不会因规则而空转）
        if not allowed and pruned:
            return pruned, []
        self.pruned += len(pruned)
        return allowed, pruned

    def _blocked(self, state: State, a: Action, problem) -> bool:
        return bool(self.reasons(state, a, problem))

    def _blocked_by(self, state: State, a: Action, problem) -> str | None:
        """返回首个触发规则名（"R1".."R4"），未触发返回 None。用于失败归因统计。"""
        hit = self.reasons(state, a, problem)
        return hit[0] if hit else None

    def reasons(self, state: State, a: Action, problem) -> tuple[str, ...]:
        """返回**全部**命中的规则名（可能多条）。

        只读：不写 `hits` / `pruned`。trace 的 `rule_mask` 走这条路径，
        否则每记一行 trace 就把同一个动作的剪枝再数一次，主表的剪枝量会翻倍。
        """
        if problem is None:
            return ()
        out: list[str] = []
        dst_sim = problem.sim.get(self._clause_of(a.dst, problem), 0.0)

        # R1
        if "R1" in self.enabled:
            if (state.depth >= 1 and dst_sim < self.sim_floor
                    and problem.coverage_ok(state) is False and len(state.clauses) >= 2):
                out.append("R1")

        # R2
        if "R2" in self.enabled:
            chain = 0
            for _, t, _ in reversed(state.used_edges):
                if t != E_DEPENDS_ON:
                    break
                chain += 1
            if a.edge_type == E_DEPENDS_ON and chain >= self.max_ref_chain:
                out.append("R2")

        # R3
        if "R3" in self.enabled and a.edge_type == E_CONFLICTS:
            src_clause = self._clause_of(a.src, problem)
            if src_clause not in state.clauses and state.depth >= 1:
                out.append("R3")

        # R4
        if "R4" in self.enabled and self.strict_exception and problem.needs_exc:
            if not problem.exception_ok(state):
                dst_node = problem.graph.node(a.dst)
                if dst_node.role != ROLE_EXC and a.edge_type != E_CONFLICTS:
                    # 只允许朝着"能解开例外"的方向走
                    if dst_sim <= problem.max_sim * 0.95:
                        out.append("R4")
        return tuple(out)

    # ------------------------------------------------------------ 风险分
    def risk(self, state: State, a: Action, problem=None) -> float:
        """0~1：越高表示这条分支越可能是无效推理（用于 H 的 −λ4·risk 项）。"""
        if problem is None:
            return 0.0
        r = 0.0
        dst_sim = problem.sim.get(self._clause_of(a.dst, problem), 0.0)
        if dst_sim < 0.12:
            r += 0.45
        elif dst_sim < 0.22:
            r += 0.2
        if a.edge_type == E_CONFLICTS:
            r += 0.25
        if a.edge_type == E_DEPENDS_ON:
            r += 0.1 * sum(1 for _, t, _ in state.used_edges if t == E_DEPENDS_ON)
        if state.depth + 1 >= problem.cfg.max_depth:
            r += 0.15
        if problem.needs_exc and not problem.exception_ok(state):
            r += 0.1
        return min(1.0, r)

    @staticmethod
    def _clause_of(nid: str, problem) -> str:
        if nid == "ROOT":
            return ""
        try:
            return problem.graph.node(nid).clause
        except KeyError:
            return ""


class NoRules:
    """B1/H 的对照：返回全部动作、风险恒 0。"""

    pruned = 0
    rejected = 0
    hits: dict = {}

    def mask(self, state, actions, problem=None):
        return list(actions), []

    def reasons(self, state, action, problem=None) -> tuple[str, ...]:
        return ()

    def risk(self, state, action, problem=None) -> float:
        return 0.0
