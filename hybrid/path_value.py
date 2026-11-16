"""路径价值分类器（A 侧补齐，供 M1 / H 的融合优先级使用）。

背景
----
文档 §2.4 把"路径价值分类器"划给 B 线（`hybrid/path_classifier.py`）。B 的那个
模块现在是 `extract_features / build_training_data` 的 pandas 版，**不再导出**
`PathValueClassifier` 与 `train_path_classifier`，而 `experiments/setup.py` 要靠
这两个符号装配 M1/H。为了让 A 的搜索结果能落地，这里补一个与契约一致的最小实现：

    PathValueClassifier.proba(x: 1-D np.ndarray) -> float      # P(该路径判对)
    PathValueClassifier.load(path) / .save(path)
    train_path_classifier(corpus, head, out_path, max_depth=...) -> PathValueClassifier

输入特征与终局决策头**完全同一套**（`search.decision.ScenarioDecision.features`，
15 维 facet 级），因此 `experiments/factory.py::FusedPriority.p_clf` 可以直接
`float(clf.proba(features))` 取值。

训练集口径
----------
只用 `train_rest` 的场景（dev / heldout / test 一律不参与），在每条场景上做
限深随机游走采样路径；标签 = "该路径经决策头给出的标签是否等于金标"。
这样分类器学的是"当前离线策略判对了没有"，天然与搜索解耦。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from search.decision import ScenarioDecision, sample_paths


@dataclass
class PathValueClassifier:
    """逻辑回归包装：`proba` 必须返回**标量**（factory 里是 float(...) 取值）。"""

    model: object | None = None
    C: float = 1.0
    n_rows: int = 0
    pos_rate: float = 0.0

    @property
    def ready(self) -> bool:
        return self.model is not None

    def proba(self, x) -> float:
        if self.model is None:
            return 0.5
        v = np.asarray(x, dtype=np.float64).reshape(1, -1)
        try:
            return float(self.model.predict_proba(v)[0, 1])
        except (IndexError, AttributeError):
            return float(self.model.predict(v)[0])

    # ---------------------------------------------------------- 持久化
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(
            {"model": self.model, "C": self.C, "n_rows": self.n_rows,
             "pos_rate": self.pos_rate}, protocol=4))

    @classmethod
    def load(cls, path: str | Path) -> "PathValueClassifier":
        d = pickle.loads(Path(path).read_bytes())
        return cls(model=d.get("model"), C=float(d.get("C", 1.0)),
                   n_rows=int(d.get("n_rows", 0)), pos_rate=float(d.get("pos_rate", 0.0)))


def train_path_classifier(corpus, head, out_path: str | Path, max_depth: int = 4,
                          k_per_scen: int = 6, limit: int = 320,
                          seed: int = 11) -> PathValueClassifier:
    """在 train_rest 上采样路径并训一个逻辑回归路径价值分类器。"""
    import random

    rng = random.Random(seed)
    all_sids = corpus.ids("train_rest")
    # ⚠️ ACE 的 train.json 按标签排序，`[:limit]` 取到的 320 个场景**全是 Compliant**
    #    （实测 train_rest 前 500 条均为 Compliant），训练集单标签会让分类器只学到
    #    "Compliant 场景里判对了吗"，H/M1 的 clf 项因此失真。这里改成固定种子随机抽样。
    if limit and limit < len(all_sids):
        sids = rng.sample(all_sids, limit)
    else:
        sids = list(all_sids)
    X: list[np.ndarray] = []
    y: list[int] = []
    for sid in sids:
        sc = corpus.scen(sid)
        g = corpus.graph(sid)
        csim = corpus.clause_sim(sid)
        sd = ScenarioDecision(head, g, csim, sc.text, max_depth,
                              facet_sim=corpus.facet_sim(sid))
        for st in sample_paths(g, csim, sc.text, rng, k=k_per_scen, max_depth=max_depth):
            X.append(sd.features(st))
            y.append(1 if sd(st)[0] == sc.label else 0)

    Xa = np.vstack(X)
    ya = np.asarray(y, dtype=np.int32)
    pos = float(ya.mean()) if len(ya) else 0.0

    from sklearn.linear_model import LogisticRegression

    if len(set(ya.tolist())) < 2:      # 标签退化成单一类别时给一个常数分类器
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="prior").fit(Xa, ya)
    else:
        model = LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced").fit(Xa, ya)

    clf = PathValueClassifier(model=model, C=1.0, n_rows=int(Xa.shape[0]), pos_rate=pos)
    clf.save(out_path)
    print(f"[path_value] trained on {Xa.shape[0]} paths / {len(sids)} scenarios "
          f"(pos_rate={pos:.3f}) -> {out_path}")
    return clf
