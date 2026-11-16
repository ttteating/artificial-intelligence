"""实验执行器：跑一个条件、聚合指标、落盘。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from eval.metrics import Row, result_to_row, summarize
from experiments.factory import Context, build_problem, run_kernel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------- oracle 深度
def load_oracle(path: Path) -> dict[str, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_oracle(d: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d), encoding="utf-8")


def compute_oracle_depth(corpus, sid: str, ctx: Context, budget: int = 120) -> int:
    """用宽松配置的 BFS 求"最少几跳能凑齐证据"，作为按跳数分层的锚。"""
    from state.core import ProblemConfig
    from search.bfs import bfs
    from data.clause_graph import ScenarioGraph  # noqa: F401

    cfg = ProblemConfig(max_depth=ctx.cfg.max_depth, max_expansions=budget,
                        tau_conf=ctx.cfg.tau_conf)
    sc = corpus.scen(sid)
    from search.decision import ScenarioDecision

    g = corpus.graph(sid)
    sim = corpus.clause_sim(sid)
    fsim = corpus.facet_sim(sid)
    dec = ScenarioDecision(ctx.head, g, sim, sc.text, cfg.max_depth, facet_sim=fsim)
    from state.core import SearchProblem

    p = SearchProblem(g, sc.text, sim, decision=dec, h_fn=lambda s, x: 0.0, cfg=cfg,
                      sid=sid, facet_sim=fsim)
    res = bfs(p, budget=budget)
    return res.steps


# ---------------------------------------------------------------- 单条件
def run_condition(corpus, condition: str, split: str, ctx: Context,
                  limit: int | None = None, verbose: bool = True,
                  oracle_path: Path | None = None) -> list[Row]:
    sids = corpus.ids(split)
    if limit and limit < len(sids):
        # ⚠️ ACE 的 train.json 是按标签排好序的，`sids[:limit]` 会取到"清一色 Compliant"
        # 的前缀（dev 前 300 条里 200 条 Compliant），F1 会系统性偏低到 0.34。
        # 因此限量子集一律用固定种子随机抽样，保证类别比例接近全集。
        import random

        rng = random.Random(ctx.seed * 7919 + limit)
        picked = set(rng.sample(sids, limit))
        sids = [s for s in sids if s in picked]
    oracle_path = oracle_path or (CACHE / f"oracle_{split}.json")
    oracle = load_oracle(oracle_path)

    rows: list[Row] = []
    t0 = time.time()
    for i, sid in enumerate(sids):
        if sid not in oracle:
            oracle[sid] = compute_oracle_depth(corpus, sid, ctx)
        problem, priority, spec = build_problem(corpus, sid, condition, ctx)
        res = run_kernel(problem, priority, spec, ctx)
        sc = corpus.scen(sid)
        rows.append(result_to_row(res, sc.label, condition, oracle[sid]))
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{condition}/{split}] {i+1}/{len(sids)} "
                  f"({(time.time()-t0)/(i+1)*1000:.0f} ms/scen)", flush=True)

    save_oracle(oracle, oracle_path)
    s = summarize(rows)
    if verbose:
        print(f"[{condition}/{split}] n={s['n']} macroF1={s['macro_f1']:.4f} "
              f"acc={s['accuracy']:.4f} expanded={s['avg_expanded']:.2f} "
              f"steps={s['avg_steps']:.2f} redundant={s['redundant_ratio']:.3f} "
              f"llm={s['avg_llm_calls']:.1f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return rows


def write_summary(rows_by_cond: dict[str, list[Row]], path: str | Path) -> dict:
    out = {c: summarize(rs) for c, rs in rows_by_cond.items()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def rows_to_csv(rows_by_cond: dict[str, list[Row]], path: str | Path) -> None:
    import pandas as pd

    recs = []
    for cond, rows in rows_by_cond.items():
        for r in rows:
            recs.append(r.to_dict())
    df = pd.DataFrame(recs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
