"""单条件实验入口。

    python -m experiments.run_baselines --condition B0 --split dev --limit 10
    python -m experiments.run_baselines --condition A1 --split dev

产物：results/rows_{condition}_{split}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import write_rows
from experiments.runner import run_condition
from experiments.setup import ROOT, prepare

TRAIN_ONLY_WARN = "⚠️  你正在读取 test 划分。按项目纪律，test 只应在 Day4 下午之后跑一次。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="B0")
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "test", "train_rest"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default="tfidf")
    ap.add_argument("--llm", default="mock", choices=["off", "mock", "real"])
    ap.add_argument("--hallucination", type=float, default=0.0)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--w-h", type=float, default=1.5)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test":
        print(TRAIN_ONLY_WARN)

    ctx = prepare(use_llm=args.llm, backend=args.backend, max_depth=args.max_depth,
                  w_h=args.w_h, retrain=args.retrain,
                  hallucination_rate=args.hallucination)
    rows = run_condition(ctx.corpus, args.condition, args.split, ctx, limit=args.limit)
    out = args.out or str(ROOT / "results" / f"rows_{args.condition}_{args.split}.json")
    write_rows(rows, out)
    print(f"[saved] {out}")

    if rows:
        ex = rows[0]
        print(f"  示例: {ex.sid} gold={ex.gold} pred={ex.answer} "
              f"expanded={ex.expanded} steps={ex.steps} redundant={ex.redundant}")


if __name__ == "__main__":
    main()
