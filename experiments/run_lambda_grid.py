"""λ 融合权重的小网格搜索（文档 §2.3 Day3：dev 上 ≤20 组）。

搜索对象是 H（规则 + LightGBM 分类器 + GA 启发式 + LLM）的融合权重：

    priority = −λ1·LLM_reward − λ2·P_clf + λ4·rule_risk

λ3（h 的权重）属于 A* 的 `w_h`，不在这里搜。评分口径沿用 B 线 `hybrid/fusion.py` 的
`macro_f1 − 0.1 · normalized_avg_expanded_nodes`（以 dev 上 A1 的扩展数为归一化锚）。

产物：
    logs/lambda_grid.json     —— 逐组记录 + best_lambda（B 线 `fusion.py` 的日志格式）
    results/lambda_grid.md    —— 人读版表格

用法：
    python -m experiments.run_lambda_grid                 # 18 组，dev 300
    python -m experiments.run_lambda_grid --limit 600
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from experiments.runner import run_condition, CACHE
from experiments.setup import prepare
from eval.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

# ≤20 组：λ_llm × λ_clf × λ_risk = 2 × 3 × 3
GRID = {
    "llm": [0.15, 0.35],
    "clf": [0.15, 0.30, 0.50],
    "risk": [0.10, 0.25, 0.40],
}
ANCHOR_EXPANDED = 5.5      # dev 上 A1 的平均扩展节点（归一化锚，见报告 §5.2）


def score_of(macro_f1: float, avg_expanded: float) -> float:
    return macro_f1 - 0.1 * (avg_expanded / ANCHOR_EXPANDED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--condition", default="H")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    ctx = prepare(quiet=True)
    combos = list(itertools.product(GRID["llm"], GRID["clf"], GRID["risk"]))
    print(f"[lambda-grid] {len(combos)} combos on {args.condition}/{args.split} "
          f"(limit={args.limit})", flush=True)

    records = []
    t0 = time.time()
    for i, (l1, l2, l4) in enumerate(combos, 1):
        ctx.lam = {"llm": l1, "clf": l2, "h": ctx.lam.get("h", 1.5), "risk": l4}
        rows = run_condition(ctx.corpus, args.condition, args.split, ctx,
                             limit=args.limit, verbose=False,
                             oracle_path=CACHE / f"oracle_{args.split}.json")
        s = summarize(rows)
        rec = {"λ1": l1, "λ2": l2, "λ3": ctx.lam["h"], "λ4": l4,
               "macro_f1": round(s["macro_f1"], 4),
               "avg_expanded_nodes": round(s["avg_expanded"], 3),
               "avg_llm_calls": round(s["avg_llm_calls"], 2),
               "score": round(score_of(s["macro_f1"], s["avg_expanded"]), 4)}
        records.append(rec)
        print(f"  [{i}/{len(combos)}] λ=({l1},{l2},{l4}) F1={rec['macro_f1']:.4f} "
              f"exp={rec['avg_expanded_nodes']:.2f} score={rec['score']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    best = max(records, key=lambda r: (r["score"], r["macro_f1"]))
    payload = {
        "results": records,
        "best_lambda": [best["λ1"], best["λ2"], best["λ3"], best["λ4"]],
        "best_score": best["score"],
        "score_formula": "macro_f1 - 0.1 * normalized_average_expanded_nodes",
        "anchor_expanded": ANCHOR_EXPANDED,
        "split": args.split,
        "limit": args.limit,
        "condition": args.condition,
        "n_combos": len(combos),
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "lambda_grid.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# λ 融合权重网格搜索（H，dev）\n",
          f"- 条件：`{args.condition}` / `{args.split}` 前 {args.limit} 条（固定种子随机子集）",
          f"- 组合数：{len(combos)}（≤20，符合文档 §2.3 Day3）",
          f"- 评分：`{payload['score_formula']}`，归一化锚 = {ANCHOR_EXPANDED} 个扩展节点",
          f"- 最优：λ = ({best['λ1']}, {best['λ2']}, {best['λ3']}, {best['λ4']})，"
          f"score = {best['score']:.4f}\n",
          "| λ1 (LLM) | λ2 (clf) | λ3 (h) | λ4 (risk) | macro-F1 | 扩展节点 | LLM 调用 | score |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        md.append(f"| {r['λ1']} | {r['λ2']} | {r['λ3']} | {r['λ4']} | {r['macro_f1']:.4f} | "
                  f"{r['avg_expanded_nodes']:.2f} | {r['avg_llm_calls']:.1f} | {r['score']:.4f} |")
    (RESULTS / "lambda_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[lambda-grid] best λ=({best['λ1']}, {best['λ2']}, {best['λ3']}, {best['λ4']}) "
          f"score={best['score']:.4f} F1={best['macro_f1']:.4f} -> logs/lambda_grid.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
