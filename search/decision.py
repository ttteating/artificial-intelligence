"""终局评估器：把"一条推理路径"映射为合规标签 + 置信度。

⚠️ 角色说明
文档里这一步由 B 同学的**路径价值分类器 + LLM reward** 承担。A 同学要跑通
A0/A1/A2/H 的对照实验，需要一个与搜索解耦的终局判据；否则"搜索更优"无法体现在
准确率上。因此这里实现 `DecisionHead`：
  * 输入：路径已审查的 facet 集合（主句/例外/条件/定义/引用/救济）与场景线索
  * 输出：三分类概率，argmax 作为 answer，max proba 作为 conf
  * 训练：**只用 train_rest 上随机游走采样的路径**（dev/heldout/test 完全不参与）

关键设计：特征必须是 **facet 级**的
--------------------------------
如果相似度只到"条款级"，那么一旦进入条款，sim_max 就饱和，再多走几步也不会改变
判断——搜索必然退化成"看第一条条款就下结论"，A* 与 BFS 会给出同一个答案。
所以这里把相似度拆到 facet：`sim_obl`（对主句）、`sim_exc`（对例外句）、
`exc_rel = sim_exc − sim_obl`。只有走到例外节点，`sim_exc` 才会从 0 抬起来，
"义务是否被豁免"才可能翻转判断——这正是多跳搜索的语义价值所在。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data.clause_graph import (
    E_CONFLICTS, ROLE_COND, ROLE_DEF, ROLE_EXC, ROLE_OBL, ROLE_REMEDY, ScenarioGraph,
)
from data.features import (
    CLAUSE_CONSENT, SCENARIO_NO_CONSENT, polarity_signals, scenario_signals,
)
from state.core import LABELS, State

FEATURE_NAMES = [
    "sim_obl", "sim_exc", "sim_max_clause", "exc_rel",
    "cov_facet", "cov_clause", "depth_norm",
    "breach_sig", "breach_x_obl", "breach_x_exc",
    "consent_gap", "na_gap", "exc_flag", "def_flag", "conflict_frac",
]
N_FEATURES = len(FEATURE_NAMES)
SIM_FLOOR = 0.08
RULE_ROLES = (ROLE_OBL, ROLE_REMEDY)


@dataclass
class ScenarioDecision:
    """单场景的绑定视图：缓存节点级线索与 facet 相似度，避免重复计算。"""

    head: "DecisionHead | None"
    graph: ScenarioGraph
    sim: dict[str, float]                       # 条款级
    text: str
    max_depth: int = 4
    facet_sim: dict[str, float] | None = None   # facet 级
    _node_sig: dict[str, dict] = field(default_factory=dict)
    _sc_sig: dict | None = None

    def __post_init__(self):
        self._sc_sig = scenario_signals(self.text)
        for n in self.graph.nodes.values():
            self._node_sig[n.nid] = polarity_signals(n.text)
        if self.facet_sim is None:
            self.facet_sim = {}

    # ---------------------------------------------------------- 特征
    def features(self, state: State) -> np.ndarray:
        nodes = [self.graph.node(nid) for nid in state.nodes]
        rules = [n for n in nodes if n.role in RULE_ROLES]
        excs = [n for n in nodes if n.role in (ROLE_EXC, ROLE_COND)]
        defs = [n for n in nodes if n.role == ROLE_DEF]

        sim_obl = max((self._fs(n.nid) for n in rules), default=0.0)
        sim_exc = max((self._fs(n.nid) for n in excs), default=0.0)
        sim_max_clause = max((self.sim.get(n.clause, 0.0) for n in nodes), default=0.0)

        cov_f = len(nodes) / max(1, len(self.graph.nodes))
        cov_c = len(state.clauses) / max(1, self.graph.n_clauses)

        breach = self._sc_sig["breach"]

        # 程序性违反：条款要求同意/通知，场景却"未取得同意"
        clause_consent = any(CLAUSE_CONSENT.search(n.text) for n in rules)
        sc_no_consent = bool(SCENARIO_NO_CONSENT.search(self.text))
        consent_gap = 1.0 if (clause_consent and sc_no_consent) else 0.0

        etc = state.edge_type_counts()
        n_edges = max(1, sum(etc.values()))

        x = [
            sim_obl,
            sim_exc,
            sim_max_clause,
            sim_exc - sim_obl,
            cov_f,
            cov_c,
            state.depth / max(1, self.max_depth),
            breach,
            breach * sim_obl,
            breach * sim_exc,
            consent_gap,
            max(0.0, SIM_FLOOR - sim_max_clause),
            1.0 if excs else 0.0,
            1.0 if defs else 0.0,
            etc.get(E_CONFLICTS, 0) / n_edges,
        ]
        return np.asarray(x, dtype=np.float32)

    def _fs(self, nid: str) -> float:
        return float(self.facet_sim.get(nid, 0.0)) if self.facet_sim else 0.0

    # ---------------------------------------------------------- 推理
    def __call__(self, state: State) -> tuple[str, float]:
        if self.head is None:
            raise RuntimeError("DecisionHead 未绑定")
        p = self.head.proba(self.features(state).reshape(1, -1))[0]
        k = int(np.argmax(p))
        return LABELS[k], float(p[k])


@dataclass
class DecisionHead:
    """多项逻辑回归（scikit-learn），可序列化为 JSON。"""

    C: float = 1.0
    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None
    classes: tuple[str, ...] = LABELS

    # ---------------------------------------------------------- 训练
    def fit(self, X: np.ndarray, y: list[str], C: float | None = None) -> "DecisionHead":
        from sklearn.linear_model import LogisticRegression

        self.C = float(C or self.C)
        try:      # sklearn <1.7 需要显式 multinomial；>=1.7 已默认移除该参数
            lr = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced",
                                    multi_class="multinomial")
        except TypeError:
            lr = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        lr.fit(X, y)
        self.coef = lr.coef_.astype(np.float64)
        self.intercept = lr.intercept_.astype(np.float64)
        self.classes = tuple(lr.classes_)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("DecisionHead 未训练")
        z = X @ self.coef.T + self.intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[str]:
        p = self.proba(X)
        return [self.classes[i] for i in np.argmax(p, axis=1)]

    # ---------------------------------------------------------- 持久化
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "C": self.C, "classes": list(self.classes),
            "coef": self.coef.tolist(), "intercept": self.intercept.tolist(),
            "features": FEATURE_NAMES,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DecisionHead":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("features") and d["features"] != FEATURE_NAMES:
            raise RuntimeError("decision_head.json 的特征集与当前代码不一致，请 --retrain")
        h = cls(C=d["C"], classes=tuple(d["classes"]))
        h.coef = np.asarray(d["coef"], dtype=np.float64)
        h.intercept = np.asarray(d["intercept"], dtype=np.float64)
        return h

    def weights_report(self) -> list[dict]:
        rows = []
        for i, c in enumerate(self.classes):
            for j, f in enumerate(FEATURE_NAMES):
                rows.append({"class": c, "feature": f, "w": round(float(self.coef[i, j]), 4)})
        return rows


# ---------------------------------------------------------------- 训练集构造
def sample_paths(graph: ScenarioGraph, sim: dict[str, float], text: str,
                 rng, k: int = 8, max_depth: int = 4,
                 facet_sim: dict[str, float] | None = None) -> list[State]:
    """随机游走采样路径状态（长度 0..max_depth），覆盖不同"看了多少证据"的情形。"""
    out: list[State] = []
    for _ in range(k):
        st = State(frozenset(), (), 0)
        target_depth = rng.randint(1, max_depth)
        out.append(st)
        while st.depth < target_depth:
            acts = _geo_actions(graph, st, max_depth)
            if not acts:
                break
            a = rng.choice(acts)
            n = graph.node(a.dst)
            if a.src == "ROOT":
                st = State(frozenset({n.clause}), (("ROOT", "ENTRY", a.dst),), 1,
                           nodes=(a.dst,), g=1.0)
            else:
                st = st.with_action(a, n.clause)
            out.append(st)
    return out


def _geo_actions(graph: ScenarioGraph, st: State, max_depth: int):
    from state.core import Action

    if st.depth >= max_depth:
        return []
    if not st.nodes:
        return [Action("ROOT", "ENTRY", nid) for nid in sorted(graph.entry)]
    last = st.nodes[-1]
    acts, seen = [], set()
    for nid in st.nodes:
        for e in graph.actions(nid):
            t = (e.src, e.etype, e.dst)
            if t in seen or (e.dst == last and e.src == nid):
                continue
            seen.add(t)
            acts.append(Action(*t))
    return acts


def build_training_set(corpus, sids: list[str], k_per_scen: int = 8,
                       max_depth: int = 4, seed: int = 7) -> tuple[np.ndarray, list[str]]:
    import random

    rng = random.Random(seed)
    X: list[np.ndarray] = []
    y: list[str] = []
    for sid in sids:
        sc = corpus.scen(sid)
        g = corpus.graph(sid)
        sim = corpus.clause_sim(sid)
        fsim = corpus.facet_sim(sid)
        sd = ScenarioDecision(None, g, sim, sc.text, max_depth, facet_sim=fsim)
        for st in sample_paths(g, sim, sc.text, rng, k=k_per_scen, max_depth=max_depth):
            X.append(sd.features(st))
            y.append(sc.label)
    return np.vstack(X), y


def train_decision_head(corpus, out_path: str | Path, k_per_scen: int = 8,
                        max_depth: int = 4, seed: int = 7, limit: int | None = None) -> DecisionHead:
    sids = corpus.ids("train_rest")
    if limit:
        sids = sids[:limit]
    X, y = build_training_set(corpus, sids, k_per_scen, max_depth, seed)
    head = DecisionHead().fit(X, y)
    head.save(out_path)
    print(f"[decision] trained on {X.shape[0]} paths / {len(sids)} scenarios "
          f"({N_FEATURES} feats) -> {out_path}")
    return head


if __name__ == "__main__":
    from data.corpus import Corpus

    c = Corpus().load()
    train_decision_head(c, "results/models/decision_head.json")
