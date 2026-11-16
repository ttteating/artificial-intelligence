"""重训模型产物（不重跑实验）：路径价值分类器 / 终局决策头 / 角色先验。

为什么单独一个脚本：`experiments/setup.py::prepare(retrain=True)` 会把决策头
（68k 路径）也一起重训，换个分类器却要付一次全量训练成本。这里只重训指定的。

用法：
    python -m tools.retrain_models --what clf
    python -m tools.retrain_models --what head prior clf
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="*", default=["clf"],
                    choices=["clf", "head", "prior"], )
    args = ap.parse_args()
    wants = set(args.what)
    MODELS.mkdir(parents=True, exist_ok=True)

    from data.corpus import Corpus

    corpus = Corpus().load()

    head = None
    if "head" in wants or "clf" in wants:
        from search.decision import DecisionHead, train_decision_head

        hp = MODELS / "decision_head.json"
        if "head" in wants or not hp.exists():
            head = train_decision_head(corpus, hp)
        else:
            head = DecisionHead.load(hp)
            print(f"[head] reusing {hp}")

    if "prior" in wants or not (MODELS / "role_prior.json").exists():
        from heuristic.handcrafted import compute_role_prior

        compute_role_prior(corpus, MODELS / "role_prior.json")

    if "clf" in wants:
        from hybrid.path_value import PathValueClassifier, train_path_classifier

        clf = train_path_classifier(corpus, head, MODELS / "path_classifier.pkl")
        print(f"[clf] n_rows={clf.n_rows} pos_rate={clf.pos_rate:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
