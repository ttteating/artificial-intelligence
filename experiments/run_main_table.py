"""主表实验：8 个条件 × (dev | test) + 预算扫描 + 分层表 + 失败案例。

    python -m experiments.run_main_table                 # 只跑 dev
    python -m experiments.run_main_table --allow-test    # dev 之后再跑一次 test（只跑一次！）

产物：
    results/main_table.csv / .md      主表
    results/by_hops.csv               按跳数分层
    results/by_clauses.csv            按条款数分层
    results/budget_sweep.csv          准确率-预算曲线数据
    results/failure_cases.json        失败案例候选
    results/figs/main_f1.png 等
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from eval.metrics import Row, bucket_by, macro_f1, summarize
from experiments.factory import CONDITIONS
from experiments.runner import rows_to_csv, run_condition, write_summary
from experiments.setup import ROOT, prepare
from state.core import ProblemConfig

RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
ORDER = ["B0", "B1", "A0", "A1", "A2", "M0", "M1", "H"]
DESC = {
    "B0": "BFS（盲目基准）",
    "B1": "BFS + 规则",
    "A0": "A* + 手工 h",
    "A1": "A* + GA 优化 h",
    "A2": "A* + GA + LLM",
    "M0": "MCTS（基准）",
    "M1": "MCTS + LLM + 分类器",
    "H": "混合：规则+分类器+GA h+LLM",
}


def setup_cjk_font() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _table(rows_by_cond: dict[str, list[Row]], title: str) -> str:
    head = ("| 条件 | macro-F1 | accuracy | 平均扩展节点 | 推理步数 | 冗余路径占比 | LLM 调用 | 规则剪枝 |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n")
    lines = [f"### {title}\n", head]
    for c in ORDER:
        if c not in rows_by_cond:
            continue
        s = summarize(rows_by_cond[c])
        lines.append(f"| {c} {DESC[c]} | {s['macro_f1']:.4f} | {s['accuracy']:.4f} | "
                     f"{s['avg_expanded']:.2f} | {s['avg_steps']:.2f} | "
                     f"{s['redundant_ratio']:.3f} | {s['avg_llm_calls']:.1f} | "
                     f"{s['avg_pruned']:.1f} |\n")
    return "".join(lines)


def stratified(rows_by_cond: dict[str, list[Row]], key: str) -> str:
    buckets = {}
    for c, rows in rows_by_cond.items():
        for b, rs in bucket_by(rows, key).items():
            buckets.setdefault(b, {})[c] = rs
    order = sorted(buckets, key=lambda x: (len(x), x))
    head = "| 分层 | " + " | ".join(f"{c} F1 / exp" for c in ORDER if c in rows_by_cond) + " |\n"
    head += "|---|" + "---|" * len([c for c in ORDER if c in rows_by_cond]) + "\n"
    lines = [head]
    for b in order:
        cells = []
        for c in ORDER:
            if c not in buckets.get(b, {}):
                continue
            s = summarize(buckets[b][c])
            cells.append(f"{s['macro_f1']:.3f} / {s['avg_expanded']:.1f} (n={s['n']})")
        lines.append(f"| {b} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


def failure_cases(rows_by_cond: dict[str, list[Row]]) -> dict:
    def idx(cond):
        return {r.sid: r for r in rows_by_cond.get(cond, [])}

    B0, B1, A0, A1 = idx("B0"), idx("B1"), idx("A0"), idx("A1")
    H, M0 = idx("H"), idx("M0")
    out = {"heuristic_misled": [], "llm_hallucination": [], "rule_overconstraint": [],
           "mcts_wander": []}

    for sid, r in A1.items():
        if not r.correct and sid in B0 and B0[sid].correct:
            out["heuristic_misled"].append({
                "sid": sid, "gold": r.gold, "A1_pred": r.answer, "B0_pred": B0[sid].answer,
                "A1_expanded": r.expanded, "B0_expanded": B0[sid].expanded,
                "A1_steps": r.steps, "note": "A* 被 h 引向错误分支，BFS 反而命中",
            })
    for sid, r in H.items():
        if not r.correct and sid in A1 and A1[sid].correct:
            out["llm_hallucination"].append({
                "sid": sid, "gold": r.gold, "H_pred": r.answer, "A1_pred": A1[sid].answer,
                "llm_calls": r.llm_calls, "llm_rejected": r.llm_rejected,
                "expanded": r.expanded,
                "note": "加入 LLM 打分后偏离正确路径（mock LLM 的打分噪声/幻觉）",
            })
    for sid, r in B1.items():
        if not r.correct and sid in B0 and B0[sid].correct and not r.reached_goal:
            out["rule_overconstraint"].append({
                "sid": sid, "gold": r.gold, "B1_pred": r.answer, "B0_pred": B0[sid].answer,
                "pruned": r.pruned, "steps": r.steps, "expanded": r.expanded,
                "note": "规则把通往例外的分支剪掉，导致证据不足",
            })
    for sid, r in M0.items():
        if not r.correct and sid in A0 and A0[sid].correct:
            out["mcts_wander"].append({
                "sid": sid, "gold": r.gold, "M0_pred": r.answer, "A0_pred": A0[sid].answer,
                "expanded": r.expanded, "steps": r.steps,
                "note": "MCTS 在高 reward 的相似条款上反复采样，忽略例外",
            })
    for k in out:
        out[k] = sorted(out[k], key=lambda x: -x.get("expanded", 0))[:3]
    return out


def budget_sweep(ctx, budget_values=(4, 6, 8, 12, 16, 24, 48), n: int = 200,
                 conds=("B0", "A0", "A1", "M0")) -> list[dict]:
    recs = []
    for b in budget_values:
        c2 = replace(ctx, cfg=ProblemConfig(max_depth=ctx.cfg.max_depth,
                                           max_expansions=b, tau_conf=ctx.cfg.tau_conf))
        for cond in conds:
            rows = run_condition(c2.corpus, cond, "dev", c2, limit=n, verbose=False)
            s = summarize(rows)
            recs.append({"budget": b, "condition": cond, "macro_f1": s["macro_f1"],
                         "accuracy": s["accuracy"], "avg_expanded": s["avg_expanded"],
                         "redundant_ratio": s["redundant_ratio"]})
            print(f"  [sweep] budget={b:3d} {cond} F1={s['macro_f1']:.4f} "
                  f"exp={s['avg_expanded']:.2f}", flush=True)
    return recs


def plots(rows_by_cond, sweep: list[dict]) -> None:
    setup_cjk_font()
    import matplotlib.pyplot as plt
    import pandas as pd

    FIGS.mkdir(parents=True, exist_ok=True)
    conds = [c for c in ORDER if c in rows_by_cond]
    f1 = [summarize(rows_by_cond[c])["macro_f1"] for c in conds]
    exp = [summarize(rows_by_cond[c])["avg_expanded"] for c in conds]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    ax.bar(conds, f1, color="#1c7ed6")
    ax.set_ylim(0, max(f1) * 1.25 + 1e-6)
    ax.set_ylabel("macro-F1")
    ax.set_title("各条件 macro-F1（" + ("test" if any(r.sid.startswith("te") for r in
                 rows_by_cond[conds[0]]) else "dev") + "）", fontsize=10)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "main_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    ax.bar(conds, exp, color="#e8590c")
    ax.set_ylabel("平均扩展节点数")
    ax.set_title("各条件搜索代价", fontsize=10)
    for i, v in enumerate(exp):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "main_expanded.png")
    plt.close(fig)

    df = pd.DataFrame(sweep)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
        for c in df["condition"].unique():
            d = df[df.condition == c].sort_values("budget")
            ax.plot(d["budget"], d["macro_f1"], "o-", label=c, markersize=4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("扩展节点预算")
        ax.set_ylabel("macro-F1")
        ax.set_title("预算-准确率曲线：启发式的收益随预算收紧而放大", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "budget_sweep.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", default="mock", choices=["off", "mock", "real"])
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    ctx = prepare(use_llm=args.llm)
    print(f"[main] dev n={ctx.corpus.n_scen('dev')}  conditions={ORDER}")

    rows_by_cond: dict[str, list[Row]] = {}
    for c in ORDER:
        rows_by_cond[c] = run_condition(ctx.corpus, c, "dev", ctx, limit=args.limit)

    rows_to_csv(rows_by_cond, RESULTS / "rows_dev_all.csv")
    write_summary(rows_by_cond, RESULTS / "summary_dev.json")

    md = ["# 主表：ACE 合规多跳推理（dev）\n", _table(rows_by_cond, "主表（dev）"), "\n"]
    md.append("## 按跳数分层（F1 / 平均扩展节点）\n" + stratified(rows_by_cond, "hops"))
    md.append("\n## 按条款数分层\n" + stratified(rows_by_cond, "clauses"))
    md.append("\n## 混淆矩阵（H）\n```\n" +
              json.dumps(summarize(rows_by_cond["H"])["confusion"], ensure_ascii=False, indent=1) +
              "\n```\n")

    fcs = failure_cases(rows_by_cond)
    (RESULTS / "failure_cases.json").write_text(
        json.dumps(fcs, ensure_ascii=False, indent=2), encoding="utf-8")

    sweep = []
    if not args.no_sweep:
        print("[sweep] 预算扫描")
        sweep = budget_sweep(ctx, n=min(200, args.limit or 200))
        pd_rows = sweep
        import pandas as pd

        pd.DataFrame(pd_rows).to_csv(RESULTS / "budget_sweep.csv", index=False,
                                     encoding="utf-8-sig")
    plots(rows_by_cond, sweep)
    (RESULTS / "main_table.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))

    if args.allow_test:
        print("\n[test] ⚠️ 读取 test（只应执行一次）")
        test_rows: dict[str, list[Row]] = {}
        for c in ORDER:
            test_rows[c] = run_condition(ctx.corpus, c, "test", ctx, limit=args.limit)
        rows_to_csv(test_rows, RESULTS / "rows_test_all.csv")
        write_summary(test_rows, RESULTS / "summary_test.json")
        tm = ["\n# 主表（test，一次性）\n", _table(test_rows, "主表（test）"),
              "\n## 按跳数分层\n" + stratified(test_rows, "hops")]
        (RESULTS / "main_table_test.md").write_text("".join(tm), encoding="utf-8")
        with open(RESULTS / "main_table.md", "a", encoding="utf-8") as f:
            f.write("".join(tm))
        print("".join(tm))


if __name__ == "__main__":
    main()
