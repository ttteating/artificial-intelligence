"""HTTP 服务层：把项目已有的搜索/实验能力包装成只读 API。

设计原则（对应"改造框架不要修改既定功能"）
-----------------------------------------
* **只读**：本层不重写任何搜索、启发、规则、融合逻辑，只调用
  `experiments.factory.build_problem / run_kernel` 与 `experiments.setup.prepare`，
  参数与主表实验完全一致。
* **惰性加载**：`prepare()` 要读 76 MB 的 train.json + test.json 并拟合 IDF，
  首次调用需要几十秒。因此服务启动秒回，语料在后台线程加载，`/api/health` 的
  `ready` 字段告诉前端什么时候可以发重请求。
* **串行执行搜索**：搜索是纯 CPU 密集，多请求并发只会互相拖慢并让 MCTS 的
  wall-clock 抖动。用一个 `Semaphore(1)` 把搜索排队。
* **结果优先读盘**：主表、扫描、GA 曲线、树、失败案例全部读 `results/` 里的
  既有产物，保证前端看到的数字与 `tools.check_report_numbers` 校验的是同一份。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TREES = RESULTS / "trees"
CUSTOM_DIR = ROOT / "data" / "custom"

_CLAUSE_KEY = re.compile(r"^evidence_\d+$")


class NotReady(RuntimeError):
    """语料/模型尚未加载完成：应回 503，而不是让请求挂住。"""

# name -> (kind, relative path)
TABLES: dict[str, tuple[str, str]] = {
    "summary_dev": ("json", "results/summary_dev.json"),
    "summary_test": ("json", "results/summary_test.json"),
    "budget_sweep": ("csv", "results/budget_sweep.csv"),
    "stratified_dev": ("csv", "results/stratified_dev.csv"),
    "stratified_test": ("csv", "results/stratified_test.csv"),
    "hallucination_sweep": ("csv", "results/hallucination_sweep.csv"),
    "hparam_sweep": ("csv", "results/hparam_sweep.csv"),
    "lambda_grid": ("json", "logs/lambda_grid.json"),
    "failure_cases": ("json", "results/failure_cases.json"),
    "graph_stats": ("json", "results/graph_stats_dev.json"),
    "ga_history": ("json", "results/ga_history.json"),
    "decision_path_eval": ("json", "results/decision_path_eval.json"),
    "llm_reject_rate": ("json", "results/llm_reject_rate.json"),
    "main_table_md": ("markdown", "results/main_table.md"),
    "main_table_test_md": ("markdown", "results/main_table_test.md"),
    "llm_cost_md": ("markdown", "results/llm_cost_table.md"),
    "llm_real_vs_mock_md": ("markdown", "results/llm_real_vs_mock.md"),
    "llm_real_vs_mock_a2_md": ("markdown", "results/llm_real_vs_mock_A2.md"),
    "llm_smoke_md": ("markdown", "results/llm_smoke.md"),
    "failure_cases_md": ("markdown", "results/failure_cases.md"),
    "readme": ("markdown", "README.md"),
}

CONDITION_DESC = {
    "B0": "BFS + 无规则 + h=0（盲目基准）",
    "B1": "BFS + 规则硬约束（规则 + 盲目）",
    "A0": "A* + 固定手工 h（无规则、无 LLM）",
    "A1": "A* + GA 优化 h（遗传算法优化）",
    "A2": "A* + GA h + LLM（动作校验 + 路径 reward）",
    "M0": "MCTS + h=0（蒙特卡洛基准）",
    "M1": "MCTS + 规则 + LLM + 路径分类器",
    "H": "A* + 规则 + 分类器 + GA h + LLM（混合架构）",
}


class Service:
    """全局单例：一次加载语料，之后所有请求复用。"""

    def __init__(self) -> None:
        self._ctx = None
        self._init_lock = threading.Lock()
        self._search_lock = threading.Semaphore(1)
        self._csv_cache: dict[str, Any] = {}
        self._json_cache: dict[str, Any] = {}
        self._idx_cache: dict[str, list[dict]] = {}
        self._custom_lock = threading.Lock()
        self.started_at = time.time()
        self.error: str | None = None

    # ------------------------------------------------------------ 初始化
    @property
    def ready(self) -> bool:
        return self._ctx is not None

    def ctx(self):
        if self._ctx is not None:
            return self._ctx
        with self._init_lock:
            if self._ctx is not None:
                return self._ctx
            from experiments.setup import prepare

            try:
                self._ctx = prepare(quiet=True)
                self._load_custom_scenarios()
                self.error = None
            except Exception as exc:            # noqa: BLE001 - 启动失败要把原因回给前端
                self.error = f"{type(exc).__name__}: {exc}"
                raise
        return self._ctx

    def require_ready(self):
        """公共端点入口：未就绪时快速失败，交给 main 转 503。"""
        if self._ctx is None:
            raise NotReady("语料与模型仍在加载中，请稍后重试")
        return self._ctx

    def warmup(self) -> None:
        try:
            self.ctx()
        except Exception:                      # noqa: BLE001 - 错误已记录在 self.error
            pass

    # ------------------------------------------------------------ 自定义合同
    def _load_custom_scenarios(self) -> None:
        """把 data/custom/ 里持久化过的合同重新登记进语料（仅在显式 persist 时存在）。"""
        if not CUSTOM_DIR.exists():
            return
        from data.corpus import Scenario

        corpus = self._ctx.corpus
        for p in sorted(CUSTOM_DIR.glob("cus_*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                sid = str(d["sid"])
                if sid in corpus.scenarios or not d.get("clauses"):
                    continue
                corpus.scenarios[sid] = Scenario(
                    sid=sid, clauses=dict(d["clauses"]), text=str(d["text"]),
                    label=str(d.get("label") or "Unlabeled"), split="custom")
            except Exception:                  # noqa: BLE001 - 单个坏文件不应阻断启动
                continue

    @staticmethod
    def _normalize_clauses(clauses: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i, (k, v) in enumerate(clauses.items(), start=1):
            key = str(k).strip()
            if not _CLAUSE_KEY.match(key) or key in out:
                key = f"evidence_{i}"
                while key in out:
                    key = f"{key}_{i}"
            text = str(v).strip()
            if not text:
                raise ValueError(f"条款 {k} 为空")
            if len(text) > 4000:
                raise ValueError(f"条款 {k} 超过 4,000 字符，请拆分后再提交")
            out[key] = text
        return out

    def create_scenario(self, scenario_text: str, clauses: dict[str, str],
                        title: str | None = None, label: str | None = None,
                        persist: bool = False) -> dict:
        """登记一份客户合同，返回图统计。

        只做"登记 + 建图"，不在这里跑搜索：前端拿到 sid 后仍走 `/api/search`，
        保证现场审查与主表实验共用同一条装配路径。
        """
        from data.facet_graph import GRAPH_VERSION

        ctx = self.require_ready()
        text = (scenario_text or "").strip()
        if not text:
            raise ValueError("场景描述不能为空")
        if len(text) > 20000:
            raise ValueError("场景描述超过 20,000 字符")
        cleaned = self._normalize_clauses(clauses or {})
        if not 1 <= len(cleaned) <= 10:
            raise ValueError("条款数量需在 1~10 条之间")

        digest = hashlib.sha1(
            (text + "|" + json.dumps(cleaned, ensure_ascii=False, sort_keys=True))
            .encode("utf-8")).hexdigest()[:10]
        sid = f"cus_{digest}"
        cached = sid in ctx.corpus.scenarios

        if not cached:
            from data.corpus import Scenario

            with self._custom_lock:
                if sid not in ctx.corpus.scenarios:
                    ctx.corpus.scenarios[sid] = Scenario(
                        sid=sid, clauses=cleaned, text=text,
                        label=(label or "Unlabeled"), split="custom")

        sc = ctx.corpus.scen(sid)
        g = ctx.corpus.graph(sid)
        # 自定义合同常出现"整条都是例外句"的条款：它不该作为搜索起点，
        # 否则第一步就从例外开始、再回头找义务，语义是反的。仅在自定义场景上收口。
        obl_entries = [nid for nid in g.entry
                       if nid in g.nodes and g.nodes[nid].role == "OBL"]
        if obl_entries:
            g.entry = obl_entries
        sim = ctx.corpus.clause_sim(sid)
        max_sim = max(sim.values()) if sim else 0.0
        floor = max(ctx.cfg.sim_floor, ctx.cfg.min_clause_cover * max_sim)
        relevant = sorted(c for c, v in sim.items() if v >= floor) or sorted(g.clause_ids)

        if persist:
            CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
            p = CUSTOM_DIR / f"{sid}.json"
            if not p.exists():
                p.write_text(json.dumps({
                    "sid": sid, "title": title or "", "text": sc.text,
                    "clauses": dict(sc.clauses), "label": sc.label,
                }, ensure_ascii=False, indent=1), encoding="utf-8")

        edges = [e for es in g.out.values() for e in es]
        return {
            "sid": sid, "title": (title or f"自定义合同 {sid}")[:120],
            "n_clauses": len(sc.clauses), "n_nodes": len(g.nodes),
            "n_edges": len(edges), "entry": list(g.entry),
            "clause_sim": {k: round(float(v), 4) for k, v in sim.items()},
            "relevant": relevant, "graph_version": GRAPH_VERSION,
            "cached": cached,
        }

    # ------------------------------------------------------------ 数据读取
    def _json(self, rel: str) -> Any:
        if rel not in self._json_cache:
            self._json_cache[rel] = json.loads(
                (ROOT / rel).read_text(encoding="utf-8"))
        return self._json_cache[rel]

    def _csv(self, rel: str):
        if rel not in self._csv_cache:
            import pandas as pd

            df = pd.read_csv(ROOT / rel).astype(object)
            self._csv_cache[rel] = df.where(pd.notna(df), None)
        return self._csv_cache[rel]

    # ------------------------------------------------------------ 元信息
    def health(self) -> dict:
        ctx = self._ctx
        return {
            "status": "error" if self.error else ("ok" if ctx else "loading"),
            "ready": bool(ctx),
            "uptime_sec": round(time.time() - self.started_at, 1),
            "n_scenarios": len(ctx.corpus.scenarios) if ctx else 0,
            "llm_mode": getattr(ctx, "use_llm", "") if ctx else "",
            "graph_version": self._graph_version(),
            "error": self.error,
        }

    @staticmethod
    def _graph_version() -> str:
        try:
            from data.facet_graph import GRAPH_VERSION

            return GRAPH_VERSION
        except Exception:                       # noqa: BLE001
            return ""

    def conditions(self) -> list[dict]:
        from experiments.factory import CONDITIONS

        out = []
        for cid, spec in CONDITIONS.items():
            out.append({
                "id": cid, "kernel": spec["kernel"], "rules": bool(spec["rules"]),
                "h": spec["h"], "llm": spec["llm"], "clf": bool(spec.get("clf")),
                "description": CONDITION_DESC.get(cid, ""),
            })
        return out

    def overview(self) -> dict:
        dev = self._json("results/summary_dev.json")
        test = self._json("results/summary_test.json")
        return {
            "health": self.health(),
            "conditions": self.conditions(),
            "summary_dev": dev,
            "summary_test": test,
            "graph_stats": self._json("results/graph_stats_dev.json"),
            "trees": self._json("results/trees/manifest.json"),
            "llm_reject": self._json("results/llm_reject_rate.json"),
            "ga_summary": self._json("results/ga_history.json").get("summary", {}),
            "lambda_best": self._json("logs/lambda_grid.json").get("best_lambda"),
            "counts": {
                "dev": (dev.get("B0") or {}).get("n"),
                "test": (test.get("B0") or {}).get("n"),
            },
        }

    # ------------------------------------------------------------ 表格
    def table(self, name: str) -> dict:
        if name not in TABLES:
            raise KeyError(f"未知表名 {name}，可选：{sorted(TABLES)}")
        kind, rel = TABLES[name]
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(rel)
        if kind == "json":
            return {"name": name, "kind": "json", "data": self._json(rel)}
        if kind == "csv":
            df = self._csv(rel)
            return {"name": name, "kind": "csv", "n_rows": int(len(df)),
                    "columns": [str(c) for c in df.columns],
                    "data": df.to_dict("records")}
        return {"name": name, "kind": "markdown", "text": path.read_text(encoding="utf-8")}

    # ------------------------------------------------------------ 场景
    def scenario_index(self, split: str) -> list[dict]:
        """把 rows_all.csv 聚合成"每场景一条、含 8 条件判定"的索引。"""
        split = "test" if split == "test" else "dev"
        if split in self._idx_cache:
            return self._idx_cache[split]
        df = self._csv(f"results/rows_{split}_all.csv")
        out: dict[str, dict] = {}
        for row in df.to_dict("records"):
            sid = row["sid"]
            d = out.setdefault(sid, {
                "sid": sid, "split": split, "gold": row["gold"],
                "n_clauses": int(row["n_clauses"] or 0),
                "oracle_hops": None if row["oracle_depth"] is None else int(row["oracle_depth"]),
                "runs": {},
            })
            d["runs"][row["condition"]] = {
                "answer": row["answer"],
                "correct": bool(row["answer"] == row["gold"]),
                "expanded": int(row["expanded"] or 0),
                "steps": int(row["steps"] or 0),
                "redundant": int(row["redundant"] or 0),
                "llm_calls": int(row["llm_calls"] or 0),
                "llm_rejected": int(row.get("llm_rejected") or 0),
                "conf": float(row["conf"] or 0.0),
            }
        self._idx_cache[split] = [out[k] for k in sorted(out)]
        return self._idx_cache[split]

    def list_scenarios(self, split: str = "dev", condition: str | None = None,
                       correct: bool | None = None, min_hops: int | None = None,
                       max_hops: int | None = None, n_clauses: int | None = None,
                       q: str | None = None, limit: int = 50, offset: int = 0) -> dict:
        items = self.scenario_index(split)
        if q:
            items = [it for it in items if q.lower() in it["sid"].lower()]
        if n_clauses is not None:
            items = [it for it in items if it["n_clauses"] == n_clauses]
        if min_hops is not None:
            items = [it for it in items if (it["oracle_hops"] or 0) >= min_hops]
        if max_hops is not None:
            items = [it for it in items if (it["oracle_hops"] or 0) <= max_hops]
        if condition:
            if correct is None:
                items = [it for it in items if condition in it["runs"]]
            else:
                items = [it for it in items
                         if it["runs"].get(condition, {}).get("correct") is correct]
        total = len(items)
        page = items[offset: offset + limit]
        return {"total": total, "offset": offset, "limit": limit, "items": page}

    def scenario(self, sid: str) -> dict:
        ctx = self.require_ready()
        if sid not in ctx.corpus.scenarios:
            raise KeyError(f"场景 {sid} 不存在")
        sc = ctx.corpus.scen(sid)
        runs: dict = {}
        oracle_hops = None
        if sc.split != "custom":
            idx = {it["sid"]: it for it in self.scenario_index(
                "test" if sc.split == "test" else "dev")}
            runs = idx.get(sid, {}).get("runs", {})
            oracle_hops = idx.get(sid, {}).get("oracle_hops")
        return {
            "sid": sid, "split": sc.split, "gold": sc.label,
            "n_clauses": len(sc.clauses), "oracle_hops": oracle_hops,
            "text": sc.text, "clauses": dict(sc.clauses),
            "runs": [{"condition": c, "answer": v["answer"], "gold": sc.label,
                      "correct": v["correct"], "expanded": v["expanded"],
                      "steps": v["steps"], "redundant": v["redundant"],
                      "llm_calls": v["llm_calls"],
                      "llm_rejected": v.get("llm_rejected", 0),
                      "conf": v["conf"]}
                     for c, v in sorted(runs.items())],
        }

    def graph(self, sid: str) -> dict:
        ctx = self.require_ready()
        if sid not in ctx.corpus.scenarios:
            raise KeyError(f"场景 {sid} 不存在")
        g = ctx.corpus.graph(sid)
        edges = [{"src": e.src, "etype": e.etype, "dst": e.dst, "w": round(float(e.w), 4)}
                 for nid in g.out for e in g.out[nid]]
        return {
            "sid": sid,
            "graph_version": self._graph_version(),
            "n_nodes": len(g.nodes), "n_edges": len(edges),
            "entry": list(g.entry),
            "nodes": [{"nid": n.nid, "clause": n.clause, "role": n.role,
                       "text": n.text, "sent_idx": n.sent_idx}
                      for n in g.nodes.values()],
            "edges": edges,
        }

    # ------------------------------------------------------------ 树
    def trees(self) -> dict:
        return self._json("results/trees/manifest.json")

    def tree(self, name: str) -> dict:
        p = TREES / f"{name}.json"
        if not p.exists():
            raise KeyError(f"树 {name} 不存在，可选：{list(self.trees())}")
        return json.loads(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ 搜索
    def search(self, sid: str, condition: str = "H", explain: bool = False,
               explain_mode: str = "template", max_expansions: int | None = None,
               include_trace: bool = True, include_tree: bool = True) -> dict:
        from experiments.factory import CONDITIONS, build_problem, run_kernel
        from llm.explainer import explain_result

        if condition not in CONDITIONS:
            raise ValueError(f"未知条件 {condition}，可选：{sorted(CONDITIONS)}")
        t0 = time.time()
        ctx = self.require_ready()
        if sid not in ctx.corpus.scenarios:
            raise KeyError(f"场景 {sid} 不存在")
        if max_expansions:
            ctx = replace(ctx, cfg=replace(ctx.cfg, max_expansions=int(max_expansions)))

        with self._search_lock:
            problem, priority, spec = build_problem(ctx.corpus, sid, condition, ctx)
            res = run_kernel(problem, priority, spec, ctx)

        sc = ctx.corpus.scen(sid)
        nodes = [n.to_dict() for n in res.tree_root.nodes] if res.tree_root else []
        n_raw = len(nodes)
        if include_tree and nodes:
            from eval.visualize_tree import prune_frontier

            nodes = prune_frontier(nodes)

        exp = None
        if explain:
            exp = explain_result(res, problem, mode=explain_mode)

        meta = {
            "sid": sid, "condition": condition, "gold": sc.label, "pred": res.answer,
            "steps": res.steps, "expanded": res.expanded_nodes,
            "confidence": res.confidence, "llm_calls": res.llm_calls,
            "n_clauses": len(sc.clauses),
            "llm_mode": getattr(problem, "llm_mode", "off"),
            "needs_exc": bool(getattr(problem, "needs_exc", False)),
            "n_relevant": len(getattr(problem, "relevant", []) or []),
            "clauses": dict(sc.clauses),
            "scenario": sc.text,
            "path": [a.to_dict() for a in res.path],
        }
        return {
            "sid": sid, "condition": condition, "answer": res.answer,
            "confidence": res.confidence, "steps": res.steps,
            "expanded_nodes": res.expanded_nodes, "reached_goal": res.reached_goal,
            "llm_calls": res.llm_calls, "pruned_by_rule": res.pruned_by_rule,
            "llm_rejected": res.llm_rejected,
            "llm_mode": meta["llm_mode"],
            "tree": nodes if include_tree else None,
            "n_nodes_raw": n_raw,
            "trace": res.trace if include_trace else None,
            "path": meta["path"],
            "meta": meta,
            "explanation": exp,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    def explain(self, sid: str, condition: str = "H", mode: str = "template") -> dict:
        out = self.search(sid, condition, explain=True, explain_mode=mode,
                          include_trace=True, include_tree=False)
        return {"sid": sid, "condition": condition, "answer": out["answer"],
                "explanation": out["explanation"], "meta": out["meta"]}

    # ------------------------------------------------------------ LLM 报告
    def llm_report(self) -> dict:
        return {
            "reject": self._json("results/llm_reject_rate.json"),
            "cost_md": (RESULTS / "llm_cost_table.md").read_text(encoding="utf-8"),
            "real_vs_mock_md": (RESULTS / "llm_real_vs_mock.md").read_text(encoding="utf-8"),
            "real_vs_mock_a2_md": (RESULTS / "llm_real_vs_mock_A2.md").read_text(
                encoding="utf-8"),
            "smoke_md": (RESULTS / "llm_smoke.md").read_text(encoding="utf-8"),
        }


service = Service()
