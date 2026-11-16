"""实验公共装配：语料、决策头、角色先验、分类器、GA 参数。缺什么训什么（并落盘）。

配置优先级：命令行参数 > config.yaml > 代码默认值。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from data.corpus import Corpus
from experiments.factory import LAMBDA, MCTS_DEFAULTS, Context
from heuristic.handcrafted import DEFAULT_PARAMS, HParams, compute_role_prior, load_role_prior
from state.core import ProblemConfig

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    if not p.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def prepare(use_llm: str | None = None, backend: str | None = None,
            max_depth: int | None = None, tau_conf: float | None = None,
            w_h: float | None = None, seed: int | None = None,
            hallucination_rate: float | None = None,
            max_expansions: int | None = None,
            retrain: bool = False, quiet: bool = False,
            config_path: str | Path | None = None) -> Context:
    cfg_file = load_config(config_path)
    d = cfg_file.get("data", {}) or {}
    sc = cfg_file.get("search", {}) or {}
    lc = cfg_file.get("llm", {}) or {}
    hc = cfg_file.get("hybrid", {}) or {}
    mc = cfg_file.get("mcts", {}) or {}

    backend = backend or d.get("backend", "tfidf")
    use_llm = use_llm or lc.get("mode", "mock")
    max_depth = int(max_depth or sc.get("max_depth", 4))
    tau_conf = float(tau_conf or sc.get("tau_conf", 0.60))
    max_expansions = int(max_expansions or sc.get("max_expansions", 16))
    w_h = float(w_h or sc.get("w_h", 1.5))
    seed = int(seed if seed is not None else cfg_file.get("seed", 0))
    hallucination_rate = float(hallucination_rate if hallucination_rate is not None
                               else lc.get("hallucination_rate", 0.0))

    corpus = Corpus(backend=backend)
    if not corpus.split_path.exists():
        from data.ace_loader import build_split

        build_split(corpus.split_path, corpus.raw_dir / "train.json")
    corpus.load()
    MODELS.mkdir(parents=True, exist_ok=True)

    # ---- 终局决策头
    head_path = MODELS / "decision_head.json"
    from search.decision import DecisionHead

    if retrain or not head_path.exists():
        from search.decision import train_decision_head

        head = train_decision_head(corpus, head_path, max_depth=max_depth)
    else:
        head = DecisionHead.load(head_path)

    # ---- 角色先验
    rp_path = MODELS / "role_prior.json"
    if retrain or not rp_path.exists():
        role_prior = compute_role_prior(corpus, rp_path)
    else:
        role_prior = load_role_prior(rp_path)

    # ---- 路径价值分类器（只在 H / M1 需要）
    # 优先级：B 线 LightGBM（`models/path_classifier.lgb`，须由
    # `experiments.train_lgb_classifier` 用真实轨迹训练、meta 标 synthetic=false）
    # > A 线逻辑回归兜底（`results/models/path_classifier.pkl`）。
    clf = None
    lgb_path = ROOT / "models" / "path_classifier.lgb"
    lgb_meta = ROOT / "models" / "path_classifier.lgb.meta.json"
    use_lgb = os.environ.get("CLF_MODEL", "auto")
    if lgb_path.exists() and use_lgb != "logreg":
        meta = {}
        if lgb_meta.exists():
            try:
                meta = json.loads(lgb_meta.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if meta.get("synthetic") is False:
            try:
                from hybrid.path_value_lgb import LGBPathValue

                clf = LGBPathValue(lgb_path, corpus)
                if not quiet:
                    print(f"[setup] 路径价值分类器 = LightGBM "
                          f"(真实轨迹, dev AUC={meta.get('dev_auc', float('nan')):.4f})")
            except Exception as exc:      # noqa: BLE001 - 缺 lightgbm / 模型损坏都回退
                print(f"[setup] LightGBM 分类器不可用（{exc}），回退逻辑回归")
                clf = None

    if clf is None:
        clf_path = MODELS / "path_classifier.pkl"
        from hybrid.path_value import PathValueClassifier, train_path_classifier

        if retrain or not clf_path.exists():
            clf = train_path_classifier(corpus, head, clf_path, max_depth=max_depth)
        else:
            clf = PathValueClassifier.load(clf_path)

    # ---- GA 参数
    ga_path = MODELS / "ga_params.json"
    ga_params = HParams(*DEFAULT_PARAMS)
    if ga_path.exists():
        dg = json.loads(ga_path.read_text(encoding="utf-8"))
        ga_params = HParams(dg.get("w_sim", 0.55), dg.get("w_prior", 0.20),
                            dg.get("w_exc", 0.15), dg.get("w_depth", 0.10))
    elif not quiet:
        print("[setup] 未找到 results/models/ga_params.json，A1/A2/H 暂用默认手工参数")

    lam = dict(LAMBDA)
    lam.update({"llm": hc.get("lambda_llm", lam["llm"]),
                "clf": hc.get("lambda_clf", lam["clf"]),
                "risk": hc.get("lambda_risk", lam["risk"])})

    return Context(
        corpus=corpus, head=head, role_prior=role_prior,
        hand_params=HParams(*DEFAULT_PARAMS), ga_params=ga_params, clf=clf,
        cfg=ProblemConfig(max_depth=max_depth, tau_conf=tau_conf,
                          max_expansions=max_expansions,
                          min_clause_cover=float(sc.get("min_clause_cover", 0.6)),
                          sim_floor=float(sc.get("sim_floor", 0.08)),
                          require_exception_resolution=bool(
                              sc.get("require_exception_resolution", True))),
        w_h=w_h, use_llm=use_llm, hallucination_rate=hallucination_rate,
        lam=lam, seed=seed,
        mcts={**MCTS_DEFAULTS, **{k: mc[k] for k in MCTS_DEFAULTS if k in mc}},
    )
