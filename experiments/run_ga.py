"""GA 优化 h_θ 的入口。

    python -m experiments.run_ga                 # 用 config.yaml 的 ga 段
    python -m experiments.run_ga --gens 12 --pop 16

产物：results/ga_history.json、results/models/ga_params.json、results/figs/ga_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.setup import ROOT, load_config, prepare
from heuristic.ga_optimizer import GAConfig, GAOptimizer, plot_curve, save


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--dev-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-expansions", type=int, default=None,
                    help="工作点预算；与主表不同时请配合 --tag 以免覆盖主实验")
    ap.add_argument("--tag", default=None, help="产物后缀，如 b8")
    args = ap.parse_args()

    cfg = load_config().get("ga", {}) or {}
    for k, v in (("gens", args.gens), ("pop", args.pop), ("dev_n", args.dev_n),
                 ("seed", args.seed)):
        if v is not None:
            cfg[k] = v
    ga_cfg = GAConfig(**{k: v for k, v in cfg.items() if k in GAConfig.__dataclass_fields__})

    ctx = prepare(max_expansions=args.max_expansions)
    ga = GAOptimizer(ctx.corpus, ctx, ga_cfg)
    best, summary = ga.run()
    if args.tag:
        summary["tag"] = args.tag
        (ROOT / "results" / f"ga_history_{args.tag}.json").write_text(
            __import__("json").dumps({"history": ga.history, "summary": summary},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        plot_curve(ROOT / "results" / "figs" / f"ga_curve_{args.tag}.png", args.tag)
    else:
        save(ga, best, summary)
        plot_curve(ROOT / "results" / "figs" / "ga_curve.png")
    print(f"[GA] 参数 -> results/models/ga_params.json  ({ga.evals} 次适应度评估)")


if __name__ == "__main__":
    main()
