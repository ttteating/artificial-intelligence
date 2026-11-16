"""LLM 幻觉敏感性曲线：p(hallucination) 从 0 拉到 0.3，看 H / A2 怎么退化。

为什么需要这个脚本
------------------
主表的 A2 / M1 / H 跑在 `MockLLMEvaluator` 上（未配置 `LLM_API_KEY` 时的兜底）。
mock 本身是**规则模拟器**，它用 `hallucination_rate` 参数模拟"看似合理但错误的奖励"。
主表把该参数固定为 0，于是"LLM 幻觉让搜索失效"只能靠失败案例定性描述。
本脚本补一条定量曲线：同一批 dev 场景，只改幻觉强度，其余全部冻结。

用法：
    python -m experiments.run_hallucination_sweep
    python -m experiments.run_hallucination_sweep --conditions H A2 --rates 0 0.1 0.2 0.3 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import summarize
from experiments.runner import run_condition
from experiments.setup import prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=["H", "A2"])
    ap.add_argument("--rates", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    recs = []
    for p in args.rates:
        ctx = prepare(use_llm="mock", hallucination_rate=p, quiet=True)
        for cond in args.conditions:
            rows = run_condition(ctx.corpus, cond, args.split, ctx,
                                 limit=args.limit, verbose=False)
            s = summarize(rows)
            recs.append({
                "hallucination_rate": p,
                "condition": cond,
                "n": s["n"],
                "macro_f1": s["macro_f1"],
                "accuracy": s["accuracy"],
                "avg_expanded": s["avg_expanded"],
                "avg_steps": s["avg_steps"],
                "redundant_ratio": s["redundant_ratio"],
                "avg_llm_calls": s["avg_llm_calls"],
                "avg_llm_rejected": s["avg_llm_rejected"],
            })
            print(f"[halluc] p={p:.2f} {cond}/{args.split} n={s['n']} "
                  f"F1={s['macro_f1']:.4f} acc={s['accuracy']:.4f} "
                  f"exp={s['avg_expanded']:.2f} red={s['redundant_ratio']:.3f}", flush=True)

    import pandas as pd

    df = pd.DataFrame(recs)
    csv = RESULTS / "hallucination_sweep.csv"
    df.to_csv(csv, index=False)

    md = [f"# LLM 幻觉敏感性（{args.split}）", "",
          "口径：`MockLLMEvaluator.hallucination_rate` 为唯一自变量，其余配置冻结；"
          "mock 是规则模拟器（非真实 LLM），用于量化\"评估值被反转为自信错误\"时的搜索退化。", "",
          "| p(幻觉) | 条件 | n | macro-F1 | acc | 平均扩展 | 平均步数 | 冗余占比 | LLM 调用 | 被拒动作 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in recs:
        md.append(f"| {r['hallucination_rate']:.2f} | {r['condition']} | {r['n']} | "
                  f"{r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['avg_expanded']:.2f} | "
                  f"{r['avg_steps']:.2f} | {r['redundant_ratio']:.3f} | "
                  f"{r['avg_llm_calls']:.1f} | {r['avg_llm_rejected']:.1f} |")
    (RESULTS / "hallucination_sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for cond in args.conditions:
            sub = df[df["condition"] == cond].sort_values("hallucination_rate")
            ax.plot(sub["hallucination_rate"], sub["macro_f1"], marker="o", label=cond)
        ax.set_xlabel("hallucination rate p")
        ax.set_ylabel("macro-F1")
        ax.set_title(f"LLM hallucination sensitivity ({args.split})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = RESULTS / "figs" / "hallucination_curve.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        print("[fig]", out, flush=True)
    except Exception as exc:      # noqa: BLE001 - 无 matplotlib 不影响数字
        print("[fig] skipped:", exc, flush=True)

    print("[saved]", csv, "|", RESULTS / "hallucination_sweep.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
