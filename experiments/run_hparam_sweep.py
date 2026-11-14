"""h_θ 的适应度地形扫描：为什么 GA 在小图上"学不到东西"？

对 (w_sim, w_exc) 做 5×5 网格（w_prior=0.2, w_depth=0.1 固定），在两个工作点上各扫一遍：
  * budget=16（主表默认）—— 搜索空间几乎被穷举，h 只影响访问顺序
  * budget=8（受限预算）—— 顺序真正决定"先看到哪条证据"，地形才出现结构

    python -m experiments.run_hparam_sweep
产物：results/hparam_sweep.csv、results/figs/hparam_landscape.png
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import macro_f1
from experiments.runner import run_condition
from experiments.setup import ROOT, prepare
from heuristic.handcrafted import HParams
from state.core import ProblemConfig

RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
W_SIM = (0.2, 0.6, 1.0, 1.4, 1.8)
W_EXC = (0.0, 0.5, 1.0, 1.5, 2.0)


def main() -> None:
    base = prepare()
    recs = []
    for budget in (16, 8):
        ctx = replace(base, cfg=ProblemConfig(max_depth=base.cfg.max_depth,
                                             max_expansions=budget,
                                             tau_conf=base.cfg.tau_conf))
        for ws in W_SIM:
            for we in W_EXC:
                p = HParams(ws, 0.2, we, 0.1)
                c2 = replace(ctx, ga_params=p)
                rows = run_condition(c2.corpus, "A1", "dev", c2, limit=300, verbose=False)
                f1 = macro_f1([r.gold for r in rows], [r.answer for r in rows])
                exp = sum(r.expanded for r in rows) / len(rows)
                recs.append({"budget": budget, "w_sim": ws, "w_exc": we,
                             "macro_f1": round(f1, 4), "avg_expanded": round(exp, 3)})
        sub = [r for r in recs if r["budget"] == budget]
        vals = [r["macro_f1"] for r in sub]
        print(f"[sweep] budget={budget}: F1 mean={np.mean(vals):.4f} "
              f"std={np.std(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "hparam_sweep.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=150)
    for ax, budget in zip(axes, (16, 8)):
        d = df[df.budget == budget].pivot(index="w_exc", columns="w_sim", values="macro_f1")
        im = ax.imshow(d.values, origin="lower", cmap="RdYlGn", vmin=0.56, vmax=0.66,
                       aspect="auto")
        ax.set_xticks(range(len(d.columns)), [f"{c:g}" for c in d.columns])
        ax.set_yticks(range(len(d.index)), [f"{c:g}" for c in d.index])
        ax.set_xlabel("w_sim")
        ax.set_ylabel("w_exc")
        ax.set_title(f"budget={budget}（{d.values.mean():.3f}±{d.values.std():.3f}）",
                     fontsize=9)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ax.text(j, i, f"{d.values[i, j]:.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.85, label="dev macro-F1")
    fig.suptitle("h_θ 适应度地形：宽松预算下几乎全平，受限预算下才出现可优化结构", fontsize=9)
    fig.savefig(FIGS / "hparam_landscape.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[sweep] -> {RESULTS/'hparam_sweep.csv'}, {FIGS/'hparam_landscape.png'}")


if __name__ == "__main__":
    main()
