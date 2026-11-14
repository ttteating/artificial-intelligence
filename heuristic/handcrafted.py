"""参数化启发函数 h_θ(n)。

设计（4 维，与文档 Day3 完全对应）
--------------------------------
    h_θ(n) = D · [ w1·(1-ŝim) + w2·(1-prior) + w3·(1-exc) + w4·depth/D ]
             ────────────────────────────────────────────────────────
                            w1 + w2 + w3 + w4

* ŝim  ：当前状态已审查条款与场景的语义相似度（按该场景可达最大值归一）
* prior：当前**前沿 facet** 的条款类型先验（该角色对标签的判别力，统计自 train_rest）
* exc  ：例外信号（是否已经走过 EXCEPTIONS/CONFLICTS 边或到达 EXC/COND 节点）
* depth：深度惩罚（已走跳数 / 深度预算）

h 被缩放到 [0, D]（D = max_depth），与 g（每跳成本 1）同量纲，因此 f = g + w·h
中的 w 可以直接理解为"启发权重"，w=1 时是标准 A*，w>1 是加权 A*。
h 的物理含义：**"从当前状态到证据充分，还差多少跳"**。ŝim 高说明找到了对口条款、
例外已解说明最关键的判断依据已经拿到，二者都让 h 变小（更接近目标）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from data.clause_graph import (
    E_CONFLICTS, E_EXCEPTIONS, ROLE_COND, ROLE_DEF, ROLE_EXC, ROLE_OBL, ROLE_REMEDY,
    ROLE_SCOPE, ROLES,
)
from state.core import State

HPARAM_NAMES = ("w_sim", "w_prior", "w_exc", "w_depth")
HPARAM_BOUNDS = ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0))

# 手工基线（A0 用）；GA 从这组值附近出发但允许自由搜索
DEFAULT_PARAMS = (0.55, 0.20, 0.15, 0.10)

FALLBACK_ROLE_PRIOR = {
    ROLE_OBL: 0.45, ROLE_EXC: 0.80, ROLE_COND: 0.65,
    ROLE_DEF: 0.50, ROLE_SCOPE: 0.40, ROLE_REMEDY: 0.35,
}


@dataclass(frozen=True)
class HParams:
    w_sim: float = DEFAULT_PARAMS[0]
    w_prior: float = DEFAULT_PARAMS[1]
    w_exc: float = DEFAULT_PARAMS[2]
    w_depth: float = DEFAULT_PARAMS[3]

    def vec(self) -> tuple[float, ...]:
        return (self.w_sim, self.w_prior, self.w_exc, self.w_depth)

    @staticmethod
    def from_vec(v) -> "HParams":
        v = [_clip(float(x), lo, hi) for x, (lo, hi) in zip(v, HPARAM_BOUNDS)]
        return HParams(*v)

    def __str__(self) -> str:
        return "{" + ", ".join(f"{k}={x:.3f}" for k, x in zip(HPARAM_NAMES, self.vec())) + "}"


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------- 角色先验
def compute_role_prior(corpus, out_path: str | Path | None = None) -> dict[str, float]:
    """角色的标签判别力：prior(r) = 1 - H(label | role) / log 3。

    只在 train_rest 上统计；比"角色出现频率"更贴近 h 需要的语义（越能决定标签的角色，越值得优先展开）。
    """
    from collections import defaultdict

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sid in corpus.ids("train_rest"):
        sc = corpus.scen(sid)
        roles = {n.role for n in corpus.graph(sid).nodes.values()}
        for r in roles:
            counts[r][sc.label] += 1
    prior: dict[str, float] = {}
    for r in ROLES:
        c = counts.get(r, {})
        tot = sum(c.values())
        if tot < 20:
            prior[r] = FALLBACK_ROLE_PRIOR.get(r, 0.4)
            continue
        h = 0.0
        for lab in ("Compliant", "Non-Compliant", "Not-Applicable"):
            p = c.get(lab, 0) / tot
            if p > 0:
                h -= p * math.log(p)
        prior[r] = round(max(0.0, min(1.0, 1.0 - h / math.log(3))), 4)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(prior, indent=2), encoding="utf-8")
    return prior


def load_role_prior(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return dict(FALLBACK_ROLE_PRIOR)


# ---------------------------------------------------------------- h 本体
class HandcraftedH:
    """把 HParams 编译成可调用的 h(state, problem)。"""

    def __init__(self, params: HParams | None = None,
                 role_prior: dict[str, float] | None = None):
        self.p = params or HParams()
        self.role_prior = dict(role_prior or FALLBACK_ROLE_PRIOR)
        s = sum(self.p.vec())
        self._norm = s if s > 1e-9 else 1.0

    def with_params(self, params: HParams) -> "HandcraftedH":
        return HandcraftedH(params, self.role_prior)

    # ---------------------------------------------------------- 分量
    @staticmethod
    def sim_component(state: State, problem) -> float:
        """语义对齐度：优先用 facet 级相似度（区分"读了主句"与"读了例外句"）。"""
        if not state.nodes:
            return 0.0
        fs = getattr(problem, "facet_sim", None)
        if fs:
            ms = max((fs.get(n, 0.0) for n in state.nodes), default=0.0)
            base = max(fs.values()) if fs else 1.0
        else:
            if not state.clauses:
                return 0.0
            ms = max((problem.sim.get(c, 0.0) for c in state.clauses), default=0.0)
            base = problem.max_sim if problem.max_sim > 1e-6 else 1.0
        return _clip(ms / base if base > 1e-6 else 0.0, 0.0, 1.0)

    def prior_component(self, state: State, problem) -> float:
        if not state.nodes:
            return 0.0
        n = problem.graph.node(state.nodes[-1])
        return _clip(self.role_prior.get(n.role, 0.4), 0.0, 1.0)

    @staticmethod
    def exc_component(state: State, problem) -> float:
        if not state.nodes:
            return 0.0
        etc = state.edge_type_counts()
        if etc.get(E_EXCEPTIONS, 0) or etc.get(E_CONFLICTS, 0):
            return 1.0
        for nid in state.nodes:
            if problem.graph.node(nid).role in (ROLE_EXC, ROLE_COND):
                return 0.75
        return 0.15

    # ---------------------------------------------------------- 主函数
    def __call__(self, state: State, problem) -> float:
        D = max(1, problem.cfg.max_depth)
        p = self.p
        sim = self.sim_component(state, problem)
        pri = self.prior_component(state, problem)
        exc = self.exc_component(state, problem)
        dep = _clip(state.depth / D, 0.0, 1.0)
        cost = (p.w_sim * (1.0 - sim) + p.w_prior * (1.0 - pri)
                + p.w_exc * (1.0 - exc) + p.w_depth * dep)
        return float(D * cost / self._norm)


def make_h(params: HParams | None = None, role_prior: dict[str, float] | None = None):
    h = HandcraftedH(params, role_prior)
    return h, (lambda s, p: h(s, p))


class ZeroH:
    """h ≡ 0：验证 A*(w·h=0) 与 BFS 结果一致（Day1 单测要求）。"""

    def __call__(self, state: State, problem) -> float:
        return 0.0
