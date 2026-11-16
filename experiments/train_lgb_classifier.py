"""用**真实 ACE 搜索轨迹**训练 B 线的 LightGBM 路径价值分类器。

此前 `models/path_classifier.lgb` 是用 `_synthetic_corpus` 合成轨迹训出来的
（`best_dev_auc=0.943` 但 `n_train=null`、`train_data_hash` 为空），文档 §2.3 Day2/3
要求"用 BFS 在 dev 上生成第一批轨迹，开始构造分类器特征和标签"。本脚本补上这一步：

1. 在 `train_rest`（随机抽样，避免 ACE 按标签排序的前缀陷阱）上跑 **B1（BFS + 规则）**，
   逐条记录 (state, action, path_history, llm_reward_avg, rule_risk) —— 即文档 2.4 的
   trace 口径；LLM reward 由 mock 评估器给出（真实模式可换 `--llm real`）。
2. 标签回溯：轨迹答案与金标一致 → 路径上的迁移为正、其余为负；失败轨迹全负。
3. 在 `dev`（随机抽样）上评估 AUC / precision@10，再落盘 `models/path_classifier.lgb`
   + 完整 meta（n_train / pos_ratio / 数据哈希 / 特征名）。

用法：
    python -m experiments.train_lgb_classifier                # 默认 600 训练 / 200 验证场景
    python -m experiments.train_lgb_classifier --n-train 300 --n-dev 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

from data.corpus import Corpus
from experiments.factory import Context, build_problem
from heuristic.handcrafted import compute_role_prior, load_role_prior
from llm.mock_evaluator import MockLLMEvaluator
from search.common import better
from search.tree import TreeRecorder
from state.core import SearchProblem

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

import networkx as nx


def to_nx(problem: SearchProblem) -> nx.DiGraph:
    """A 的 ScenarioGraph（facet 级）→ B 的建图口径 nx.DiGraph。"""
    g = nx.DiGraph()
    for nid, node in problem.graph.nodes.items():
        g.add_node(nid, text=node.text, role=node.role, clause=node.clause)
    for u, es in problem.graph.out.items():
        for e in es:
            g.add_edge(u, e.dst, type=e.etype, weight=float(e.w))
            g.add_edge(e.dst, u, type=e.etype + "_REV", weight=float(e.w))
    return g


def bfs_collect(problem: SearchProblem, mock: MockLLMEvaluator):
    """复刻 search/bfs.py 的 BFS，但额外记下每个 frontier 迁移的完整状态。"""
    root = problem.initial_state()
    frontier = deque([(root, None)])
    closed = {root.key()}
    rows = []
    expanded = 0
    budget = problem.cfg.max_expansions

    while frontier:
        state, parent_act = frontier.popleft()
        if expanded >= budget:
            break
        expanded += 1
        if problem.is_goal(state):
            answer, _ = problem.evaluate(state)
            return answer, state.path(), rows, True
        succ, _pruned = problem.expand(state)
        for act, nxt in succ:
            k = nxt.key()
            if k in closed:
                continue
            closed.add(k)
            # 文档 2.4 trace 口径：迁移后的状态 + 该动作 + 路径历史
            rows.append({
                "state": nxt,
                "action": act,
                "path_history": list(nxt.path()),
                "llm_reward_avg": float(mock.path_reward(nxt.path(), problem.text)),
                "rule_risk": float(problem.rules.risk(state, act, problem)),
            })
            frontier.append((nxt, act))

    best = None
    for state, _a in list(frontier):
        if state.depth > 0:
            best = better(best, (problem.terminal_score(state), state, 0))
    if best is None:
        answer, _ = problem.evaluate(root)
        return answer, [], rows, False
    answer, _ = problem.evaluate(best[1])
    return answer, best[1].path(), rows, False


def collect_split(corpus: Corpus, ctx: Context, sids: list[str], condition: str,
                  seed: int):
    graphs, trajs, labels = {}, [], {}
    for sid in sids:
        problem, _priority, _spec = build_problem(corpus, sid, condition, ctx)
        mock = MockLLMEvaluator(problem=problem, hallucination_rate=0.0, seed=seed)
        answer, path, rows, _goal = bfs_collect(problem, mock)
        graphs[sid] = to_nx(problem)
        labels[sid] = corpus.scen(sid).label
        trajs.append({"sample_id": sid, "question": problem.text,
                      "answer": answer, "path": path, "trace": rows})
    return trajs, graphs, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--condition", default="B1", choices=["B0", "B1"],
                    help="B1 = BFS + 规则（rule_risk 特征才有信息量）")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--out", default=str(MODELS / "path_classifier.lgb"),
                    help="模型输出路径（Windows 上目标文件被占用时可换名再改名）")
    args = ap.parse_args()

    from search.decision import DecisionHead
    from hybrid.path_classifier import (PathClassifier, build_training_data,
                                        evaluate_classifier, train_classifier)

    corpus = Corpus().load()
    head_path = ROOT / "results" / "models" / "decision_head.json"
    head = DecisionHead.load(head_path)
    rp_path = ROOT / "results" / "models" / "role_prior.json"
    role_prior = load_role_prior(rp_path) if rp_path.exists() else compute_role_prior(
        corpus, ROOT / "results" / "models" / "role_prior.json")
    ctx = Context(corpus=corpus, head=head, role_prior=role_prior, use_llm="mock")

    rng = random.Random(args.seed)
    train_ids = rng.sample(corpus.ids("train_rest"), args.n_train)
    dev_ids = rng.sample(corpus.ids("dev"), args.n_dev)

    print(f"[collect] {args.condition} on {len(train_ids)} train_rest + {len(dev_ids)} dev ...")
    t_tr, g_tr, l_tr = collect_split(corpus, ctx, train_ids, args.condition, args.seed)
    t_dv, g_dv, l_dv = collect_split(corpus, ctx, dev_ids, args.condition, args.seed + 1)

    X_train, y_train = build_training_data(t_tr, g_tr, l_tr, fit_corpus=True)
    X_dev, y_dev = build_training_data(t_dv, g_dv, l_dv, fit_corpus=False)
    print(f"[data] train nodes={len(X_train)} pos={int(y_train.sum())} "
          f"({y_train.mean():.3f}) | dev nodes={len(X_dev)} pos={int(y_dev.sum())} "
          f"({y_dev.mean():.3f})")

    meta: dict = {}
    model = train_classifier(X_train, y_train, X_dev, y_dev, seed=args.seed, meta=meta)
    metrics = evaluate_classifier(model, X_dev, y_dev)
    print(f"[eval] dev AUC={metrics['auc']:.4f} p@10={metrics['precision_at_10']:.4f} "
          f"p@20={metrics['precision_at_20']:.4f}")

    clf = PathClassifier()
    clf.model = model
    clf.feature_columns = list(model.feature_name())
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.out)
    if model_path.exists():
        model_path.unlink()
    clf.save(model_path)          # 会写一份 sidecar，下面用真实口径覆盖它

    meta.update({
        "model_file": "path_classifier.lgb",
        "feature_names": list(model.feature_name()),
        "condition": args.condition,
        "n_train_scenarios": len(train_ids),
        "n_dev_scenarios": len(dev_ids),
        "n_train_nodes": len(X_train),
        "n_dev_nodes": len(X_dev),
        "dev_auc": metrics["auc"],
        "precision_at_10": metrics["precision_at_10"],
        "precision_at_20": metrics["precision_at_20"],
        "synthetic": False,
        "top_features": dict(list(metrics["feature_importance"].items())[:8]),
    })
    meta_path = MODELS / "path_classifier.lgb.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {model_path} + {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
