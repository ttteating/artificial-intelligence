"""评估"终局判据单独用时"的路径级性能（报告 6.1 的归因依据）。

搜索把答案抬到 0.62~0.65，究竟有多少来自"选对证据"、多少来自"判据本身"？
这份脚本回答下半句：把判据直接作用在 held-out 场景的**随机游走路径**上，
不跑任何搜索、不做任何剪枝，得到的就是判据自身的路径级 macro-F1 上限。

产物：results/decision_path_eval.json
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from experiments.setup import MODELS, prepare
from search.decision import DecisionHead, build_training_set


def main():
    ctx = prepare(quiet=True)
    corpus = ctx.corpus
    head = DecisionHead.load(MODELS / "decision_head.json")

    out = {}
    for split in ("heldout", "dev"):
        sids = corpus.ids(split)
        # 换一个采样种子，避免与训练时的采样路径重合
        X, y = build_training_set(corpus, sids, k_per_scen=8, max_depth=ctx.cfg.max_depth, seed=99)
        pred = head.predict(X)
        out[split] = {
            "n_paths": len(y),
            "n_scen": len(sids),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        }
        print(f"[{split}] n_scen={len(sids)} n_paths={len(y)} "
              f"acc={out[split]['accuracy']:.4f} macroF1={out[split]['macro_f1']:.4f}")

    p = Path("results/decision_path_eval.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", p)


if __name__ == "__main__":
    main()
