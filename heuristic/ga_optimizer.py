"""GA 优化器：自动搜索 h_θ 的 4 个权重。

实现严格对齐文档 Day3：
  * 染色体 4 维实数（w_sim / w_prior / w_exc / w_depth），范围 [0,2]
  * 种群 12、代数 8~10、锦标赛选择 k=3、SBX 交叉（eta=15, pc=0.9）、高斯变异（σ=0.12, pm=0.35）
  * 适应度 = dev 子集 macro-F1 − λ·扩展节点惩罚
  * **每代把当前最优个体放到 held-out（100 条，GA 完全不可见）上评估**，
    用于画 dev vs held-out 曲线、量化过拟合风险
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from eval.metrics import macro_f1
from experiments.factory import Context
from experiments.runner import run_condition
from heuristic.handcrafted import HPARAM_BOUNDS, DEFAULT_PARAMS, HParams

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GAConfig:
    pop: int = 12
    gens: int = 9
    k_tour: int = 3
    pc: float = 0.9
    eta: float = 15.0
    pm: float = 0.35
    sigma: float = 0.12
    elitism: int = 1
    dev_n: int = 300
    lam_exp: float = 0.5
    seed: int = 42
    cond: str = "A1"


def _clip(v):
    return [min(hi, max(lo, x)) for x, (lo, hi) in zip(v, HPARAM_BOUNDS)]


def tournament(pop, fits, k, rng):
    idx = rng.sample(range(len(pop)), k)
    return max(idx, key=lambda i: fits[i])


def sbx(p1, p2, eta, rng):
    c1, c2 = list(p1), list(p2)
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return _clip(c1), _clip(c2)


def mutate(v, sigma, pm, rng):
    return _clip([x + rng.gauss(0, sigma) if rng.random() < pm else x for x in v])


class GAOptimizer:
    def __init__(self, corpus, ctx: Context, cfg: GAConfig | None = None):
        self.c = corpus
        self.ctx = ctx
        self.cfg = cfg or GAConfig()
        self.rng = random.Random(self.cfg.seed)
        self.history: list[dict] = []
        self.evals = 0
        self.baseline_exp = 1.0
        self._cache: dict[tuple, tuple[float, float]] = {}

    # ------------------------------------------------------------ 适应度
    def _eval(self, params: HParams) -> tuple[float, float, float, float]:
        """返回 (fitness, dev_f1, dev_avg_expanded, dev_acc)。带缓存。"""
        key = tuple(round(x, 4) for x in params.vec())
        if key in self._cache:
            return self._cache[key]
        ctx = replace(self.ctx, ga_params=params)
        rows = run_condition(self.c, self.cfg.cond, "dev", ctx,
                            limit=self.cfg.dev_n, verbose=False)
        f1 = macro_f1([r.gold for r in rows], [r.answer for r in rows])
        exp = sum(r.expanded for r in rows) / max(1, len(rows))
        acc = sum(r.correct for r in rows) / max(1, len(rows))
        pen = self.cfg.lam_exp * max(0.0, exp / max(1e-6, self.baseline_exp) - 1.0)
        fit = f1 - pen
        self.evals += 1
        self._cache[key] = (fit, f1, exp, acc)
        return fit, f1, exp, acc

    def _heldout(self, params: HParams) -> tuple[float, float]:
        ctx = replace(self.ctx, ga_params=params)
        sids = self.c.ids("heldout")
        rows = run_condition(self.c, self.cfg.cond, "heldout", ctx, limit=None, verbose=False)
        f1 = macro_f1([r.gold for r in rows], [r.answer for r in rows])
        exp = sum(r.expanded for r in rows) / max(1, len(rows))
        return f1, exp

    # ------------------------------------------------------------ 主循环
    def run(self) -> HParams:
        cfg = self.cfg
        # 基线扩展节点数（B0 在同一 dev 子集上）
        b0 = run_condition(self.c, "B0", "dev", self.ctx, limit=cfg.dev_n, verbose=False)
        self.baseline_exp = max(1e-6, sum(r.expanded for r in b0) / max(1, len(b0)))
        print(f"[GA] baseline(B0) avg_expanded={self.baseline_exp:.2f}, "
              f"pop={cfg.pop}, gens={cfg.gens}, dev_n={cfg.dev_n}", flush=True)

        pop = [_clip(list(np.asarray(DEFAULT_PARAMS) + np.random.RandomState(i).normal(0, 0.35, 4)))
               for i in range(cfg.pop)]
        fits = [self._eval(HParams.from_vec(v))[0] for v in pop]

        for g in range(cfg.gens):
            order = np.argsort(fits)[::-1]
            best_i = int(order[0])
            best_p = HParams.from_vec(pop[best_i])
            fit, f1, exp, acc = self._eval(best_p)
            hf1, hexp = self._heldout(best_p)
            self.history.append({
                "gen": g, "best_fitness": round(fit, 4),
                "mean_fitness": round(float(np.mean(fits)), 4),
                "dev_macro_f1": round(f1, 4), "dev_acc": round(acc, 4),
                "dev_expanded": round(exp, 3),
                "heldout_macro_f1": round(hf1, 4), "heldout_expanded": round(hexp, 3),
                "params": {k: round(x, 4) for k, x in zip(
                    ("w_sim", "w_prior", "w_exc", "w_depth"), best_p.vec())},
            })
            print(f"[GA] gen {g}: fit={fit:.4f} devF1={f1:.4f} devExp={exp:.2f} "
                  f"heldF1={hf1:.4f} {best_p}", flush=True)

            new_pop = [pop[i] for i in order[: cfg.elitism]]
            while len(new_pop) < cfg.pop:
                a = tournament(pop, fits, cfg.k_tour, self.rng)
                b = tournament(pop, fits, cfg.k_tour, self.rng)
                c1, c2 = (pop[a], pop[b]) if self.rng.random() > cfg.pc else sbx(pop[a], pop[b], cfg.eta, self.rng)
                new_pop.append(mutate(c1, cfg.sigma, cfg.pm, self.rng))
                if len(new_pop) < cfg.pop:
                    new_pop.append(mutate(c2, cfg.sigma, cfg.pm, self.rng))
            pop = new_pop[: cfg.pop]
            fits = [self._eval(HParams.from_vec(v))[0] for v in pop]

        best_v = pop[int(np.argmax(fits))]
        best = HParams.from_vec(best_v)
        # 收尾：最优个体在全量 dev + heldout 上复核
        ctx = replace(self.ctx, ga_params=best)
        dev_rows = run_condition(self.c, self.cfg.cond, "dev", ctx, verbose=False)
        held_rows = run_condition(self.c, self.cfg.cond, "heldout", ctx, verbose=False)
        summary = {
            "best_params": {k: round(x, 4) for k, x in zip(
                ("w_sim", "w_prior", "w_exc", "w_depth"), best.vec())},
            "dev_macro_f1": round(macro_f1([r.gold for r in dev_rows], [r.answer for r in dev_rows]), 4),
            "dev_expanded": round(sum(r.expanded for r in dev_rows) / len(dev_rows), 3),
            "heldout_macro_f1": round(macro_f1([r.gold for r in held_rows], [r.answer for r in held_rows]), 4),
            "heldout_expanded": round(sum(r.expanded for r in held_rows) / len(held_rows), 3),
            "baseline_B0_expanded": round(self.baseline_exp, 3),
            "n_evaluations": self.evals,
            "config": cfg.__dict__,
        }
        print(f"[GA] best {best} devF1={summary['dev_macro_f1']} "
              f"heldF1={summary['heldout_macro_f1']}", flush=True)
        return best, summary


def save(ga: "GAOptimizer", best: HParams, summary: dict) -> None:
    (ROOT / "results" / "ga_history.json").write_text(
        json.dumps({"history": ga.history, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (ROOT / "results" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "models" / "ga_params.json").write_text(
        json.dumps(summary["best_params"], indent=2), encoding="utf-8")


def plot_curve(path: Path, tag: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = ROOT / "results" / (f"ga_history_{tag}.json" if tag else "ga_history.json")
    h = json.loads(f.read_text(encoding="utf-8"))["history"]
    gens = [r["gen"] for r in h]
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax1.plot(gens, [r["dev_macro_f1"] for r in h], "o-", color="#1c7ed6", label="dev macro-F1（适应度可见）")
    ax1.plot(gens, [r["heldout_macro_f1"] for r in h], "s--", color="#e8590c", label="held-out macro-F1（GA 不可见）")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("macro-F1")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(gens, [r["dev_expanded"] for r in h], "^:", color="#868e96", label="dev 平均扩展节点")
    ax2.set_ylabel("avg expanded nodes")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, fontsize=7, loc="lower right")
    ax1.set_title("GA 优化 h_θ：dev 与 held-out 的分叉 = 过拟合信号", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
