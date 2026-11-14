"""搜索树可视化导出：从 dev 上自动挑 3 棵有代表性的树。

    python -m experiments.run_visuals

挑选规则
--------
1. success       —— H 判对、步数 ≥ 2、扩展节点最少的场景（结构完整且干净）
2. astar_misled  —— A1 判错但 B0 判对的场景（启发误导）
3. llm_misjudge  —— H 判错但 A1 判对、且出现 LLM 驳回/规则剪枝的场景（混合架构失效）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.visualize_tree import export
from experiments.factory import build_problem, run_kernel
from experiments.runner import run_condition
from experiments.setup import ROOT, prepare

OUT = ROOT / "results" / "trees"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=699)
    args = ap.parse_args()

    ctx = prepare()
    rows = {c: {r.sid: r for r in run_condition(ctx.corpus, c, "dev", ctx,
                                                limit=args.n, verbose=False)}
            for c in ("B0", "A0", "A1", "H")}

    def run_one(sid: str, cond: str):
        problem, priority, spec = build_problem(ctx.corpus, sid, cond, ctx)
        return run_kernel(problem, priority, spec, ctx)

    def pick(cands_sids, cond, max_nodes=34):
        """在候选里挑"树规模可读"的一棵（先跑一遍看节点数）。"""
        best = None
        for sid in cands_sids[:10]:
            res = run_one(sid, cond)
            n = len(res.tree_root.nodes)
            score = (0 if n <= max_nodes else 1, n)
            if best is None or score < best[0]:
                best = (score, sid, res)
            if n <= max_nodes:
                break
        return best

    cands: dict[str, tuple | None] = {"success": None, "astar_misled": None, "llm_misjudge": None}
    ok = [r for r in rows["H"].values() if r.correct and r.reached_goal and r.steps >= 3]
    if not ok:
        ok = [r for r in rows["H"].values() if r.correct and r.reached_goal and r.steps >= 2]
    if ok:
        sids = [r.sid for r in sorted(ok, key=lambda r: (r.steps, -r.expanded), reverse=True)]
        cands["success"] = pick(sids, "H")

    mis = [r for r in rows["A1"].values()
           if (not r.correct)
           and ((rows["B0"].get(r.sid) and rows["B0"][r.sid].correct)
                or (rows["A0"].get(r.sid) and rows["A0"][r.sid].correct))]
    if mis:
        sids = [r.sid for r in sorted(mis, key=lambda r: (r.steps, -r.expanded), reverse=True)]
        cands["astar_misled"] = pick(sids, "A1")

    llm = [r for r in rows["H"].values()
           if not r.correct and rows["A1"].get(r.sid) and rows["A1"][r.sid].correct]
    if llm:
        sids = [r.sid for r in sorted(llm, key=lambda r: (r.llm_rejected, -r.expanded), reverse=True)]
        cands["llm_misjudge"] = pick(sids, "H")

    manifest = {}
    for name, item in cands.items():
        if not item:
            print(f"[viz] 跳过 {name}：未找到合适场景")
            continue
        _score, sid, res = item
        cond = "H" if name != "astar_misled" else "A1"
        sc = ctx.corpus.scen(sid)
        meta = {
            "condition": cond, "sid": sid, "gold": sc.label, "pred": res.answer,
            "steps": res.steps, "expanded": res.expanded_nodes,
            "confidence": res.confidence, "llm_calls": res.llm_calls,
            "n_clauses": len(sc.clauses),
            "clauses": {k: v[:220] + ("…" if len(v) > 220 else "") for k, v in sc.clauses.items()},
            "scenario": sc.text[:900] + ("…" if len(sc.text) > 900 else ""),
            "path": [a.to_dict() for a in res.path],
        }
        info = export(res, OUT, name, meta)
        manifest[name] = {"sid": sid, "condition": cond, **info}
        print(f"[viz] {name}: {sid} ({cond}) gold={sc.label} pred={res.answer} "
              f"nodes={info['nodes']} -> {info['html']}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
