"""按跳数分层表：把主表数字拆成「1-2 跳 / 3 跳 / 4+ 跳」三档。

Day 5 要求"准备'启发式为什么高效/不高效'的按跳数分层数据"。本脚本只读
`results/rows_{split}_all.csv`（主表跑完就有了），不重跑搜索，因此可随时重算。

用法：
    python -m experiments.run_stratified              # dev + test 都出
    python -m experiments.run_stratified --split dev  # 只出 dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval.metrics import macro_f1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDS = ("B0", "B1", "A0", "A1", "A2", "M0", "M1", "H")
ORDER = ("1-2 跳", "3 跳", "4+ 跳")


def hop_bucket(depth: int) -> str:
    return "1-2 跳" if depth <= 2 else ("3 跳" if depth == 3 else "4+ 跳")


def build(split: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"rows_{split}_all.csv")
    df["hops"] = df["oracle_depth"].apply(hop_bucket)
    recs = []
    for bucket in ORDER:
        sub_all = df[df["hops"] == bucket]
        for cond in CONDS:
            sub = sub_all[sub_all["condition"] == cond]
            if sub.empty:
                continue
            recs.append({
                "split": split,
                "stratum": bucket,
                "condition": cond,
                "n": len(sub),
                "macro_f1": round(macro_f1(list(sub["gold"]), list(sub["answer"])), 4),
                "accuracy": round(float((sub["gold"] == sub["answer"]).mean()), 4),
                "avg_expanded": round(float(sub["expanded"].mean()), 2),
                "avg_steps": round(float(sub["steps"].mean()), 2),
                "redundant_ratio": round(float(sub["redundant"].sum() / max(1, sub["expanded"].sum())), 3),
            })
    return pd.DataFrame(recs)


def to_md(df: pd.DataFrame, split: str) -> str:
    lines = [f"# 按跳数分层表（{split}）", "",
             "分层口径：`oracle_depth`（BFS 手工启发式下的最短路径跳数）≤2 记为 1-2 跳，"
             "=3 记为 3 跳，≥4 记为 4+ 跳。", "",
             "| 跳数层 | 条件 | n | macro-F1 | acc | 平均扩展节点 | 平均步数 | 冗余占比 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for bucket in ORDER:
        sub = df[df["stratum"] == bucket]
        if sub.empty:
            lines.append(f"| {bucket} | — | 0 | — | — | — | — | — |")
            continue
        for _, r in sub.iterrows():
            lines.append(f"| {bucket} | {r['condition']} | {r['n']} | {r['macro_f1']:.4f} | "
                         f"{r['accuracy']:.4f} | {r['avg_expanded']:.2f} | {r['avg_steps']:.2f} | "
                         f"{r['redundant_ratio']:.3f} |")
        lines.append("")
    # 关键对比：A* 相对 BFS 的扩展节点节省率（同层同数据集）
    lines += ["## 同层扩展节点节省率（相对 BFS）", "",
              "| 跳数层 | B0 | A0 | A1 | H | M0 |",
              "|---|---:|---:|---:|---:|---:|"]
    for bucket in ORDER:
        sub = df[df["stratum"] == bucket].set_index("condition")
        if sub.empty or "B0" not in sub.index:
            continue
        b0 = float(sub.loc["B0", "avg_expanded"])
        cells = []
        for c in ("A0", "A1", "H", "M0"):
            if c in sub.index:
                v = float(sub.loc[c, "avg_expanded"])
                cells.append(f"{v:.2f} ({(v - b0) / b0 * 100:+.1f}%)")
            else:
                cells.append("—")
        lines.append(f"| {bucket} | {b0:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["dev", "test", "both"])
    args = ap.parse_args()
    splits = ("dev", "test") if args.split == "both" else (args.split,)
    all_df = []
    for split in splits:
        f = RESULTS / f"rows_{split}_all.csv"
        if not f.exists():
            print(f"[skip] 缺少 {f}，先跑 run_main_table")
            continue
        df = build(split)
        all_df.append(df)
        (RESULTS / f"stratified_{split}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        (RESULTS / f"stratified_{split}.md").write_text(to_md(df, split), encoding="utf-8")
        print(f"[stratified] {split}: {len(df)} 行 -> results/stratified_{split}.md", flush=True)
    if all_df:
        both = pd.concat(all_df, ignore_index=True)
        (RESULTS / "stratified_all.csv").write_text(both.to_csv(index=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
