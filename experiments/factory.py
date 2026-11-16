"""条件装配：把 (场景, 条件名) 装配成一个可直接跑的 SearchProblem。

八个条件的唯一差别都收敛到这里，避免每个实验脚本各写一份 if/else：

  条件  内核    规则   h         LLM        分类器  说明
  B0    BFS     -      0         -          -       盲目基准
  B1    BFS     硬约束 0         -          -       规则 + 盲目
  A0    A*      -      手工 h    -          -       固定手工启发
  A1    A*      -      GA 优化 h -          -       遗传算法优化
  A2    A*      -      GA 优化 h mock/real  -       LLM 参与动作校验 + reward
  M0    MCTS    -      0         -          -       蒙特卡洛基准
  M1    MCTS    -      0         mock/real  有       MCTS + LLM + 分类器
  H     A*      硬约束 GA 优化 h mock/real  有       规则+学习+启发混合
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable

from data.clause_graph import E_CONFLICTS, E_EXCEPTIONS
from heuristic.handcrafted import HParams, HandcraftedH
from hybrid.rule_engine import NoRules, RuleEngine
from llm.adapter import build_evaluator
from search.decision import ScenarioDecision
from state.core import Action, ProblemConfig, SearchProblem, State

CONDITIONS = {
    "B0": dict(kernel="bfs",   rules=False, h="zero", llm="off"),
    "B1": dict(kernel="bfs",   rules=True,  h="zero", llm="off"),
    "A0": dict(kernel="astar", rules=False, h="hand", llm="off"),
    "A1": dict(kernel="astar", rules=False, h="ga",   llm="off"),
    "A2": dict(kernel="astar", rules=False, h="ga",   llm="mock"),
    "M0": dict(kernel="mcts",  rules=False, h="zero", llm="off"),
    "M1": dict(kernel="mcts",  rules=True,  h="zero", llm="mock", clf=True),
    "H":  dict(kernel="astar", rules=True,  h="ga",   llm="mock", clf=True),
}

# dev 上 18 组网格搜索（experiments/run_lambda_grid.py）的最优组合。
# 网格里 F1 恒为 0.6437、扩展节点只在 4.57~4.58 之间抖动，说明融合权重不是瓶颈；
# 这里取 score 最高的 (0.15, 0.30, 0.40)，同时保留可追溯的 logs/lambda_grid.json。
LAMBDA = dict(llm=0.15, clf=0.30, h=1.5, risk=0.40)

# config.yaml 的 mcts 段默认值。此前这些参数在 run_kernel 里被硬编码成 90/1.3/2/1.0，
# 恰好与配置相同，所以"改配置不生效"这个 bug 一直没暴露出来。
MCTS_DEFAULTS = dict(n_simulations=90, c_ucb=1.3, rollout_depth=2,
                     reward_cap=1.0, aggregation="mean", hybrid_weight=0.0)


def stable_seed(sid: str, base: int = 0, mod: int = 9973) -> int:
    """进程无关的稳定种子。

    不能用内置 hash(str)：Python 默认开启 hash 随机化（PYTHONHASHSEED 随机），
    会让 MCTS 的采样序列在每次运行时都不同——实测同一份代码两次跑 test，
    M0 的 macro-F1 会从 0.6061 漂到 0.6051。用 crc32 才能保证"一键复现"为真。
    """
    return base + zlib.crc32(sid.encode("utf-8")) % mod


@dataclass
class Context:
    """一次实验运行共享的模型与参数。"""

    corpus: object
    head: object
    role_prior: dict
    hand_params: HParams = field(default_factory=HParams)
    ga_params: HParams | None = None
    clf: object | None = None
    cfg: ProblemConfig = field(default_factory=ProblemConfig)
    w_h: float = 1.5
    use_llm: str = "mock"
    hallucination_rate: float = 0.0
    lam: dict = field(default_factory=lambda: dict(LAMBDA))
    seed: int = 0
    mcts: dict = field(default_factory=lambda: dict(MCTS_DEFAULTS))


# ---------------------------------------------------------------- 融合优先级
class FusedPriority:
    """priority = −λ1·LLM_reward − λ2·P_clf + λ4·risk（越小越好，等价于提升优先级）。

    A* 里把它作为 f 的调整项；MCTS 里用 (1+bonus) 做 rollout / 分层采样的偏置。
    `explain()` 把每一步的三项分数写进 trace，报告里的可视化标注依赖它。
    """

    def __init__(self, problem: SearchProblem, evaluator, clf, lam: dict, rules):
        self.p = problem
        self.ev = evaluator
        self.clf = clf
        self.lam = lam
        self.rules = rules
        self._r: dict[tuple, float] = {}
        self._s: dict[tuple, dict] = {}
        self._c: dict[tuple, float] = {}
        self._sd = ScenarioDecision(getattr(problem, "decision_head", None),
                                    problem.graph, problem.sim, problem.text,
                                    problem.cfg.max_depth,
                                    facet_sim=getattr(problem, "facet_sim", None))

    def llm_reward(self, state: State) -> float:
        k = state.key()
        if k not in self._r:
            if self.ev is None or not state.used_edges:
                self._r[k] = 0.0
            else:
                self._r[k] = float(self.ev.path_reward(state.path(), self.p.text))
        return self._r[k]

    def p_clf(self, state: State) -> float:
        """路径价值分类器的 P(正确路径)。按 (状态键, facet 序列) 缓存。

        key() 只含条款集合与边类型多重集，不含具体 facet 序列；两个 key 相同但
        facet 不同的状态进入特征后并不相同，所以缓存键要带上 `state.nodes`。

        分类器有两种装配：B 线 LightGBM（`hybrid/path_value_lgb.py`，走
        `score_state`，需要 llm_reward / rule_risk 两个实时信号）与 A 线的
        逻辑回归兜底（走 15 维决策特征 + `proba`）。接口按鸭子类型分派。
        """
        if self.clf is None:
            return 0.5
        k = (state.key(), state.nodes)
        if k not in self._c:
            if hasattr(self.clf, "score_state"):
                risk = (float(self.rules.risk(state, state.path()[-1], self.p))
                        if state.used_edges else 0.0)
                self._c[k] = float(self.clf.score_state(
                    self.p, state, llm_reward=self.llm_reward(state), risk=risk))
            else:
                x = self._sd.features(state)
                self._c[k] = float(self.clf.proba(x))
        return self._c[k]

    def risk(self, state: State) -> float:
        if not state.used_edges:
            return 0.0
        a = state.path()[-1]
        return float(self.rules.risk(state, a, self.p))

    def components(self, state: State) -> dict:
        return {"llm_reward": round(self.llm_reward(state), 4),
                "clf_score": round(self.p_clf(state), 4),
                "risk": round(self.risk(state), 4)}

    def __call__(self, state: State, g: float, h: float) -> float:
        c = self.components(state)
        self._s[state.key()] = c
        return (-self.lam["llm"] * c["llm_reward"]
                - self.lam["clf"] * c["clf_score"]
                + self.lam["risk"] * c["risk"])

    def explain(self, state: State) -> dict:
        return self._s.get(state.key(), {})

    # MCTS 的偏置：奖励高、动作合法、风险低的动作更值得试
    def bias(self, state: State, action: Action) -> float:
        """M1 的动作偏置，把 LLM reward 与**分类器**都算进来。

        早期实现只用了 `action_valid` 的置信度与规则 risk，分类器 `p_clf` 从没进过
        MCTS，于是"MCTS + LLM + 分类器"里的分类器实际缺位，M1 与 M0 的差别只剩 LLM
        ——报告里"分类器贡献"那一段会站不住。这里按与 A* 相同的融合口径合成：

            0.30·(2·conf−1) + λ1·llm_reward − λ2·(0.5−p_clf)·2 − λ4·risk

        llm_reward 只取**已经算过并缓存**的子状态分数（不在此处新发 LLM 调用），
        p_clf 按子状态缓存计算；因此 MCTS 的采样不会放大 LLM 调用成本。
        """
        if self.ev is None:
            return 0.0
        child = self.p.apply(state, action)
        _, conf = self.ev.action_valid(state, action, self.p.text)
        # 只用**已经算过**的 LLM path_reward（命中 `_r` 缓存），不在这里新发调用：
        # MCTS 每次采样都会问很多候选动作，若在 bias 里现算 reward，真实 LLM 模式的
        # 调用量会按 rollout 状态数膨胀，直接违背文档 §3.6 的成本约束。
        llm = self._r.get(child.key(), 0.0)
        return (0.30 * (2.0 * conf - 1.0)
                + self.lam["llm"] * llm
                + self.lam["clf"] * (2.0 * self.p_clf(child) - 1.0)
                - self.lam["risk"] * self.rules.risk(state, action, self.p))


class HybridRules:
    """规则掩码 + LLM 合法性校验的组合（A2 / H / M1 用）。

    文档 §2.3 Day2 要求"低置信度进入可疑缓冲区"：LLM 判合法但置信度低
    （< `SUSPECT_CONF`）的动作不当场丢弃，而是排到候选列表末尾——既不浪费一次
    可能正确的展开，也不让它抢在高置信度动作前面。计数写入 `SearchResult.llm_suspicious`。
    """

    SUSPECT_CONF = 0.40

    def __init__(self, rules: RuleEngine, evaluator, problem, cache_size: int = 4096):
        self.rules = rules
        self.ev = evaluator
        self.p = problem
        self._cache: dict[tuple, tuple[bool, float]] = {}
        self.rejected = 0
        self.suspicious = 0
        self.veto_log: list[dict] = []      # 供 LLM 误杀率统计（见 tools/llm_reject_report.py）

    @property
    def pruned(self) -> int:
        """规则剪枝数（含 LLM 驳回前的那一层），由被包裹的规则引擎统计。"""
        return getattr(self.rules, "pruned", 0)

    @property
    def hits(self) -> dict:
        return getattr(self.rules, "hits", {})

    def reasons(self, state: State, action: Action, problem=None) -> tuple[str, ...]:
        """只透传"规则"这一层的命中原因；LLM 驳回单列在 llm_rejected，不混进 rule_mask。"""
        fn = getattr(self.rules, "reasons", None)
        return tuple(fn(state, action, problem or self.p)) if callable(fn) else ()

    def mask(self, state: State, actions: list[Action], problem=None):
        problem = problem or self.p
        allowed, pruned = self.rules.mask(state, actions, problem)
        if self.ev is None:
            return allowed, pruned
        keep, suspect = [], []
        for a in allowed:
            k = (state.key(), a.src, a.edge_type, a.dst)
            if k not in self._cache:
                self._cache[k] = self.ev.action_valid(state, a, problem.text)
            ok, conf = self._cache[k]
            if not ok:
                self.rejected += 1
                pruned.append(a)
                self.veto_log.append({"src": a.src, "edge_type": a.edge_type,
                                      "dst": a.dst, "conf": float(conf)})
            elif float(conf) < self.SUSPECT_CONF:
                suspect.append(a)          # 可疑缓冲区：合法但低置信
            else:
                keep.append(a)
        self.suspicious += len(suspect)
        # 低置信动作排到后面：搜索若在预算内就能避开它们，避不开也能试到
        ordered = keep + suspect
        if not ordered and allowed:
            return allowed[:1], pruned
        return ordered, pruned

    def risk(self, state: State, action: Action, problem=None) -> float:
        return self.rules.risk(state, action, problem or self.p)


# ---------------------------------------------------------------- 装配
def build_problem(corpus, sid: str, condition: str, ctx: Context):
    spec = CONDITIONS[condition]
    sc = corpus.scen(sid)
    graph = corpus.graph(sid)
    sim = corpus.clause_sim(sid)
    fsim = corpus.facet_sim(sid)

    head = ctx.head
    decision = ScenarioDecision(head, graph, sim, sc.text, ctx.cfg.max_depth, facet_sim=fsim)

    # ---- h
    if spec["h"] == "zero":
        h_fn, h_obj = (lambda s, p: 0.0), None
    else:
        params = ctx.hand_params if spec["h"] == "hand" else (ctx.ga_params or ctx.hand_params)
        h_obj = HandcraftedH(params, ctx.role_prior)
        h_fn = lambda s, p, _h=h_obj: _h(s, p)   # noqa: E731

    rules = RuleEngine() if spec["rules"] else NoRules()
    problem = SearchProblem(graph, sc.text, sim, decision=decision, h_fn=h_fn,
                            rules=rules, cfg=ctx.cfg, sid=sid, facet_sim=fsim)
    problem.decision_head = head

    # ---- LLM / 融合
    evaluator, mode = (None, "off")
    if spec["llm"] != "off":
        evaluator, mode = build_evaluator(problem, ctx.use_llm, ctx.hallucination_rate,
                                          seed=stable_seed(sid, ctx.seed, 1000))
    problem.llm_mode = mode

    need_fusion = evaluator is not None or spec.get("clf")
    priority = None
    if need_fusion:
        # A2 只有 LLM，没有硬规则：用 NoRules 包一层，避免规则悄悄进入 A2 消融
        base_rules = rules if spec["rules"] else NoRules()
        merged = HybridRules(base_rules, evaluator, problem)
        problem.rules = merged
        priority = FusedPriority(problem, evaluator, ctx.clf if spec.get("clf") else None,
                                 ctx.lam, merged)
    return problem, priority, spec


def run_kernel(problem: SearchProblem, priority, spec: dict, ctx: Context):
    from search.astar import astar
    from search.bfs import bfs
    from search.mcts import mcts
    from search.tree import TreeRecorder

    rec = TreeRecorder()
    seed = stable_seed(problem.sid, ctx.seed, 9973)
    if spec["kernel"] == "bfs":
        res = bfs(problem, recorder=rec)
    elif spec["kernel"] == "astar":
        res = astar(problem, w_h=ctx.w_h, recorder=rec, priority_fn=priority)
    else:
        m = {**MCTS_DEFAULTS, **(ctx.mcts or {})}
        res = mcts(problem, recorder=rec, seed=seed,
                   bias=None if priority is None else priority.bias,
                   priority=priority,
                   n_simulations=int(m["n_simulations"]),
                   c_ucb=float(m["c_ucb"]),
                   rollout_depth=int(m["rollout_depth"]),
                   reward_cap=float(m["reward_cap"]),
                   agg=str(m["aggregation"]),
                   hybrid_weight=float(m["hybrid_weight"]))
    # 计数量回填（规则剪枝 / LLM 驳回 / LLM 调用）
    res.pruned_by_rule = getattr(problem.rules, "pruned", 0)
    res.llm_rejected = getattr(problem.rules, "rejected", 0)
    res.llm_suspicious = getattr(problem.rules, "suspicious", 0)
    if priority is not None and priority.ev is not None:
        res.llm_calls = getattr(priority.ev, "calls", 0)
    return res
