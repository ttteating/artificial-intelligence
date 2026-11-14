"""规则致错归因（一次性诊断脚本，产物写 results/rule_diagnosis.json）。

对"B1 错而 B0 对"的样本，跑四种配置看差异究竟由哪条规则造成：
  B0                : 无规则
  B1                : 全部规则 + strict_exception=True
  B1_loose          : 全部规则 + strict_exception=False（R4 只降权不剪枝）
  B1_noR4           : 只关掉 R4

如果 B1_loose / B1_noR4 与 B0 一致，则该例确由 R4 过约束造成；
若仍不一致，则说明来自 R1/R2/R3，或与规则无关（判据在预算边界上的抖动）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.setup import prepare
from experiments.factory import build_problem, run_kernel, Context
from hybrid.rule_engine import RuleEngine

CASES = [
    ("dev", "tr01078"), ("test", "te00231"), ("test", "te00438"),
    ("test", "te00593"), ("test", "te00793"), ("test", "te00859"),
    ("test", "te01063"),
]


def go(ctx, sid, variant):
    problem, priority, spec = build_problem(ctx.corpus, sid, "B1", ctx)
    if variant == "B0":
        problem.rules = None
    else:
        eng = RuleEngine(strict_exception=(variant == "B1"))
        if variant == "B1_noR4":
            eng.enabled = ("R1", "R2", "R3")
        problem.rules = eng
    res = run_kernel(problem, priority, spec, ctx)
    return {
        "answer": res.answer, "steps": res.steps, "expanded": res.expanded_nodes,
        "pruned": res.pruned_by_rule, "reached_goal": res.reached_goal,
        "hits": dict(getattr(problem.rules, "hits", {}) or {}),
        "path": [(e.src, e.edge_type, e.dst) for e in res.path],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rule_diagnosis.json")
    args = ap.parse_args()

    ctx = prepare()
    rows = []
    for split, sid in CASES:
        gold = ctx.corpus.scen(sid).label
        rec = {"split": split, "sid": sid, "gold": gold}
        for v in ("B0", "B1", "B1_loose", "B1_noR4"):
            rec[v] = go(ctx, sid, v)
        rows.append(rec)
        print(f"{split} {sid} gold={gold}")
        for v in ("B0", "B1", "B1_loose", "B1_noR4"):
            r = rec[v]
            print(f"   {v:9s} -> {r['answer']:14s} steps={r['steps']} exp={r['expanded']:2d} "
                  f"pruned={r['pruned']:2d} hits={r['hits']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", args.out)


if __name__ == "__main__":
    main()
