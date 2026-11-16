"""LightGBM 路径价值分类器的装配层（B 线 `hybrid/path_classifier.py` 的模型接入）。

背景
----
B 线的 `hybrid/path_classifier.py` 只提供"训练 + 单点推理"能力
（`PathClassifier.success_prob(state, question, path_history, llm_reward_avg, rule_risk, graph)`），
它要求一个 `nx.DiGraph` 与**已拟合的 TF-IDF 语料**，而实验链路的图是 A 侧的
`ScenarioGraph`。这个适配层负责三件事：

1. 把 `ScenarioGraph` 转成 `nx.DiGraph`（节点带 text/role/clause，边带 type/weight）；
2. 在训练语料上拟合 TF-IDF（`fit_tfidf`，进程内一次）；
3. 暴露 `score_state(problem, state, llm_reward, risk)`，签名与 `FusedPriority.p_clf` 对齐。

模型由 `python -m experiments.train_lgb_classifier` 用**真实 ACE 轨迹**训练产出。
"""

from __future__ import annotations

import random
from pathlib import Path

import networkx as nx


def to_nx(problem) -> nx.DiGraph:
    """`ScenarioGraph` → `nx.DiGraph`（B 线特征提取所需）。"""
    g = nx.DiGraph()
    for nid, node in problem.graph.nodes.items():
        g.add_node(nid, text=node.text, role=node.role, clause=node.clause)
    for u, es in problem.graph.out.items():
        for e in es:
            g.add_edge(u, e.dst, type=e.etype, weight=float(e.w))
            g.add_edge(e.dst, u, type=e.etype + "_REV", weight=float(e.w))
    return g


class LGBPathValue:
    """把 B 的 LightGBM 分类器包成 `p_clf(state)` 可用的对象。"""

    def __init__(self, model_path: str | Path, corpus=None,
                 n_fit: int = 400, seed: int = 7919):
        from hybrid.path_classifier import PathClassifier, fit_tfidf, reset_tfidf

        reset_tfidf()
        if corpus is not None:
            texts = self._corpus_texts(corpus, n_fit, seed)
            fit_tfidf(texts)
        # ⚠️ 本机 Windows + 含中文的项目路径下，LightGBM 的 fopen 对**绝对路径**会报
        #    "Could not open ..."，对相对路径正常。因此绝对路径失败时改用 relpath 重试。
        import os

        try:
            self.clf = PathClassifier(Path(model_path))
        except Exception:
            rel = os.path.relpath(str(Path(model_path).resolve()), os.getcwd())
            self.clf = PathClassifier(Path(rel))
        self.meta = dict(getattr(self.clf, "train_meta", {}) or {})
        self._nx: dict[str, nx.DiGraph] = {}

    @staticmethod
    def _corpus_texts(corpus, n_fit: int, seed: int) -> list[str]:
        ids = corpus.ids("train_rest")
        rng = random.Random(seed)
        picked = rng.sample(ids, min(n_fit, len(ids)))
        out: list[str] = []
        for sid in picked:
            out.append(str(corpus.scen(sid).text))
            for node in corpus.graph(sid).nodes.values():
                out.append(str(node.text))
        return out

    def _graph(self, problem) -> nx.DiGraph:
        g = self._nx.get(problem.sid)
        if g is None:
            g = to_nx(problem)
            self._nx[problem.sid] = g
        return g

    def score_state(self, problem, state, llm_reward: float = 0.0,
                    risk: float = 0.0) -> float:
        """P(该路径判对)；depth 0 无动作返回中性 0.5。"""
        if not state.used_edges:
            return 0.5
        return float(self.clf.success_prob(
            state=state, question=problem.text,
            path_history=list(state.path()),
            llm_reward_avg=float(llm_reward), rule_risk=float(risk),
            graph=self._graph(problem)))

    # 兼容旧接口（历史代码会调用 clf.proba(x)）；本适配层不按 15 维向量打分
    def proba(self, _x) -> float:
        return 0.5
