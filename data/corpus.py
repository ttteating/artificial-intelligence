"""Corpus：把 ACE 样本 + 条款图 + 相似度封装成搜索内核可用的对象。

一切按需构建并落盘缓存（`results/cache/`），保证 GA 与实验的多次进程启动不重复计算。
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from data.ace_loader import load_raw, normalize_split
from data.clause_graph import ScenarioGraph
from data.facet_graph import (GRAPH_VERSION, build_graph, precompute_clause_sim,
                              precompute_facet_sim)
from data.features import build_scorer

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Scenario:
    sid: str
    clauses: dict[str, str]
    text: str
    label: str
    split: str

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)


@dataclass
class Corpus:
    raw_dir: Path = ROOT / "data" / "raw"
    split_path: Path = ROOT / "data" / "split.json"
    cache_dir: Path = ROOT / "results" / "cache"
    backend: str = "tfidf"

    scenarios: dict[str, Scenario] = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    scorer: object | None = None
    _graphs: dict[str, ScenarioGraph] = field(default_factory=dict)
    _sim: dict[str, dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------ 构建
    def load(self) -> "Corpus":
        import json

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # split.json 由 B 侧 ace_loader.make_split 产出，键面是
        # metadata / splits / distributions；这里归一成 Corpus 用的
        # sids / train_rest / dev / heldout（见 ace_loader.normalize_split）。
        self.split = normalize_split(json.loads(self.split_path.read_text(encoding="utf-8")))

        # Use load_raw with split argument to get data consistently with split.json
        train_indices = load_raw(split="train", raw_dir=self.raw_dir)
        sids = self.split["sids"]

        # Load only the train-split samples by index
        train_path = self.raw_dir / "train.json"
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # train_indices gives us the indices into the full train.json scenarios
        for pos_key in ("train_rest", "dev", "heldout"):
            for i in self.split[pos_key]:
                idx = train_indices[i]  # get the original index from train.json
                r = data["scenarios"][idx]  # get the sample by index
                self.scenarios[sids[i]] = Scenario(
                    sid=sids[i], clauses=dict(r["clauses"]),
                    text=r["scenario_text"], label=r["gd_tr"], split=pos_key,
                )

        test_path = self.raw_dir / "test.json"
        if test_path.exists():
            for j, r in enumerate(load_raw(test_path, split="test")):
                sid = f"te{j:05d}"
                self.scenarios[sid] = Scenario(
                    sid=sid, clauses=dict(r["clauses"]),
                    text=r["scenario_text"], label=r["gd_tr"], split="test",
                )

        self._fit_encoder()
        return self

    def _fit_encoder(self) -> None:
        """IDF 只在 train_rest 上估计，避免 dev/heldout/test 的词汇信息泄漏。"""
        texts: list[str] = []
        for s in self.ids("train_rest"):
            sc = self.scenarios[s]
            texts.append(sc.text)
            texts.extend(sc.clauses.values())
        self.scorer = build_scorer(self.backend, texts)

    # ------------------------------------------------------------ 访问
    def ids(self, split: str) -> list[str]:
        if split == "test":
            return [s for s, sc in self.scenarios.items() if sc.split == "test"]
        return [self.split["sids"][i] for i in self.split[split]]

    def scen(self, sid: str) -> Scenario:
        return self.scenarios[sid]

    def n_scen(self, split: str) -> int:
        return len(self.ids(split))

    # ------------------------------------------------------------ 图
    def _graph_cache_file(self) -> Path:
        h = hashlib.md5(f"{GRAPH_VERSION}:{self.backend}".encode()).hexdigest()[:10]
        return self.cache_dir / f"graphs_{h}.pkl"

    def graph(self, sid: str) -> ScenarioGraph:
        if sid in self._graphs:
            return self._graphs[sid]
        f = self._graph_cache_file()
        if not hasattr(self, "_graph_store"):
            if f.exists():
                try:
                    self._graph_store = pickle.loads(f.read_bytes())
                except Exception:
                    self._graph_store = {}
            else:
                self._graph_store = {}
        g = self._graph_store.get(sid)
        if g is None:
            sc = self.scen(sid)
            g = build_graph(sc.sid, sc.clauses, self.clause_sim(sid))
            self._graph_store[sid] = g
        self._graphs[sid] = g
        return g

    def save_graph_cache(self) -> None:
        f = self._graph_cache_file()
        store = getattr(self, "_graph_store", {})
        f.write_bytes(pickle.dumps(store, protocol=4))

    # ------------------------------------------------------------ 相似度
    def facet_sim(self, sid: str) -> dict[str, float]:
        """facet 级相似度（缓存）。"""
        if not hasattr(self, "_fsim"):
            self._fsim = {}
        if sid not in self._fsim:
            sc = self.scen(sid)
            self._fsim[sid] = precompute_facet_sim(self.scorer, sc.text, self.graph(sid))
        return self._fsim[sid]

    def clause_sim(self, sid: str) -> dict[str, float]:
        if sid in self._sim:
            return self._sim[sid]
        sc = self.scen(sid)
        sims = precompute_clause_sim(self.scorer, sc.text, sc.clauses)
        self._sim[sid] = sims
        return sims
