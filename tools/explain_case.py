"""单场景"推理自然语言解释"生成器（课程第 3 项的交付入口）。

用法：
    # 离线模板解释（不需要 key，答辩现场断网也能跑）
    python -m tools.explain_case --sid tr00632 --condition H

    # 真实 LLM 解释（需要 LLM_API_KEY；越权给结论会自动回退模板并标 violation）
    python -m tools.explain_case --sid tr00632 --condition H --mode real

    # 同时导出带解释的搜索树 HTML/JSON
    python -m tools.explain_case --sid tr00632 --condition H --export-tree

    # 批量：对三棵样例树对应的场景都生成解释
    python -m tools.explain_case --preset samples --mode template

输出默认落 `results/explanations/{sid}_{condition}.json`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.factory import build_problem, run_kernel
from experiments.setup import ROOT, prepare
from llm.explainer import attach_and_dump

OUT_DIR = ROOT / "results" / "explanations"

# 与 results/trees/manifest.json 对齐的三棵样例树
SAMPLES = [
    ("success", "tr00632", "H"),
    ("astar_misled", "tr00715", "A1"),
    ("llm_misjudge", "tr01689", "H"),
]


def explain_one(ctx, sid: str, condition: str, mode: str = "template",
                export_tree: bool = False, out_dir: Path = OUT_DIR) -> dict:
    problem, priority, spec = build_problem(ctx.corpus, sid, condition, ctx)
    res = run_kernel(problem, priority, spec, ctx)
    out = out_dir / f"{sid}_{condition}.json"
    exp = attach_and_dump(res, problem, out, mode=mode)

    if export_tree:
        from eval.visualize_tree import export

        meta = {
            "condition": condition, "sid": sid,
            "gold": ctx.corpus.scen(sid).label, "pred": res.answer,
            "steps": res.steps, "expanded": res.expanded_nodes,
            "confidence": res.confidence, "llm_calls": res.llm_calls,
        }
        export(res, ROOT / "results" / "trees", f"{sid}_{condition}_explained", meta=meta)
    return exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="")
    ap.add_argument("--condition", default="H")
    ap.add_argument("--mode", default="template", choices=("template", "real", "auto"))
    ap.add_argument("--export-tree", action="store_true")
    ap.add_argument("--preset", default="", choices=("", "samples"))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    ctx = prepare(quiet=False)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    targets = SAMPLES if args.preset == "samples" else [(args.sid, args.sid, args.condition)]
    if not args.preset and not args.sid:
        ap.error("需要 --sid 或 --preset samples")

    for _name, sid, cond in targets:
        exp = explain_one(ctx, sid, cond, mode=args.mode,
                          export_tree=args.export_tree, out_dir=out_dir)
        print(f"\n===== {sid} / {cond} / mode={exp['mode']} "
              f"violation={exp['violation']} =====")
        print(exp["text"])
        print(f"[explain] -> {out_dir / f'{sid}_{cond}.json'}")


if __name__ == "__main__":
    main()
