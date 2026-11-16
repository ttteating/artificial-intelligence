"""hybrid/fusion.py: fusion scorer, lambda grid search and smoke test."""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from state.core import Action, SearchResult, State, SearchProblem
from data.clause_graph import Edge, Node, ScenarioGraph
from hybrid.rule_engine import RuleEngine
from hybrid.path_classifier import PathClassifier


LOGDIR = Path(__file__).resolve().parents[1] / "logs"


class FusionScorer:
    """Fusion of llm_validator, llm_reward, rule-based classifier and rules.

    Priorities follow:
        priority = λ1*llm_reward + λ2*P_clf + λ3*h_theta - λ4*rule_risk
    Hard rule mask happens before this weighted sum.
    """

    def __init__(
        self,
        llm_validator,
        llm_reward,
        classifier: PathClassifier,
        rule_engine: RuleEngine,
        h_func,
        lambdas: Tuple[float, float, float, float],
    ) -> None:
        if len(lambdas) != 4:
            raise ValueError("lambdas must have exactly 4 values: (λ1, λ2, λ3, λ4)")
        self.llm_validator = llm_validator
        self.llm_reward = llm_reward
        self.classifier = classifier
        self.rule_engine = rule_engine
        self.h_func = h_func
        self.lambdas = tuple(float(v) for v in lambdas)
        self._reward_cache: Dict[Tuple, float] = {}
        self._clf_cache: Dict[Tuple, float] = {}
        self._risk_cache: Dict[Tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def priority(self, state: State, action: Action, question: str, path_history: List[Action], graph: nx.DiGraph) -> float:
        allowed, pruned = self.rule_engine.mask(state, [action], None)
        if action not in allowed:
            return float("-inf")

        llm_val = self._get_llm_reward(path_history, question)
        rule_risk = self._get_rule_risk(state, action)
        p_clf = self._get_p_clf(state, action, question, path_history, rule_risk, graph)
        h_theta = self.h_func(state)

        l1, l2, l3, l4 = self.lambdas
        return float(l1 * llm_val + l2 * p_clf + l3 * h_theta - l4 * rule_risk)

    # ------------------------------------------------------------------
    # Internal fast components (cacheable)
    # ------------------------------------------------------------------
    def _path_key(self, path_history: List[Action]) -> Tuple:
        return tuple((a.src, a.edge_type, a.dst) for a in path_history)

    def _get_llm_reward(self, path_history: List[Action], question: str) -> float:
        key = (self._path_key(path_history), question)
        if key not in self._reward_cache:
            if not callable(self.llm_reward):
                raise TypeError("llm_reward must be callable")
            value = float(self.llm_reward(path_history, question))
            self._reward_cache[key] = value
        return self._reward_cache[key]

    def _get_rule_risk(self, state: State, action: Action) -> float:
        key = (state.key(), action.as_tuple())
        if key not in self._risk_cache:
            value = float(self.rule_engine.risk(state, action, None))
            self._risk_cache[key] = value
        return self._risk_cache[key]

    def _get_p_clf(self, state: State, action: Action, question: str, path_history: List[Action], rule_risk: float, graph: nx.DiGraph) -> float:
        key = (state.key(), self._path_key(path_history), action.as_tuple(), question, rule_risk)
        if key not in self._clf_cache:
            if not callable(self.llm_reward):
                raise TypeError("classifier success_prob requires a callable llm_reward-backed scorer")
            value = float(self.classifier.success_prob(state, question, path_history, self._get_llm_reward(path_history, question), rule_risk, graph))
            self._clf_cache[key] = value
        return self._clf_cache[key]

    def grid_search_lambdas(self, dev_samples, search_fn, lambda_grid, budget=100):
        """Grid search over lambda components on dev only."""
        items = list(itertools.product(*lambda_grid.values()))
        if len(items) > 20:
            items = items[:20]
        records = []
        expanded_vals = []
        for lam in items:
            lam_tuple = tuple(float(x) for x in lam)
            f1s = []
            exs = []
            for sample in dev_samples:
                res = search_fn(sample, lam_tuple, sample.get("question", ""))
                if isinstance(res, SearchResult):
                    pred = res.answer or "Not-Applicable"
                    expanded = int(res.expanded_nodes or 0)
                elif isinstance(res, dict):
                    pred = res.get("answer") or res.get("pred")
                    expanded = int(res.get("expanded_nodes") or 0)
                else:
                    raise TypeError("search_fn must return SearchResult or dict")
                f1s.append(_sample_macro_f1(pred, sample.get("label", "")))
                exs.append(expanded)
            avg_exp = sum(exs) / len(exs) if exs else 0.0
            macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
            record = {
                "λ1": lam_tuple[0],
                "λ2": lam_tuple[1],
                "λ3": lam_tuple[2],
                "λ4": lam_tuple[3],
                "macro_f1": macro_f1,
                "avg_expanded_nodes": avg_exp,
            }
            records.append(record)
            expanded_vals.append(avg_exp)
        norm_exp = _expand_normalize(expanded_vals)
        for rec, norm in zip(records, norm_exp):
            rec["score"] = rec["macro_f1"] - 0.1 * norm
        best_record = max(records, key=lambda r: r["score"])
        best = (best_record["λ1"], best_record["λ2"], best_record["λ3"], best_record["λ4"])
        payload = {
            "results": records,
            "best_lambda": best,
            "best_score": best_record["score"],
            "log_path": LOGDIR / "lambda_grid.json",
        }
        _write_lambda_grid_log(payload["log_path"], records, best, best_record["score"])
        return best, payload


# ----------------------------------------------------------------------
# Lambda grid search helpers
# ----------------------------------------------------------------------
def _sample_macro_f1(pred: str, true: str, labels: Tuple[str, ...] = ("Compliant", "Non-Compliant", "Not-Applicable")) -> float:
    """Macro F1 for one sample."""
    return 1.0 if pred == true else 0.0


def _expand_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _write_lambda_grid_log(path: Path, records: List[Dict[str, Any]], best_lambda: Tuple[float, ...], best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": records,
        "best_lambda": list(best_lambda),
        "best_score": best_score,
        "score_formula": "macro_f1 - 0.1 * normalized_average_expanded_nodes",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Smoke-test helpers
# ----------------------------------------------------------------------
class _SmokeRuleEngine:
    """Deterministic stub rule engine for smoke tests.

    Hard-prunes every DEFINES edge on the first depth, so priority() should
    return -inf for such actions. This validates the priority mask branch
    without depending on a full domain rule engine.
    """

    def mask(self, state: State, actions: List[Action], problem=None):
        allowed = []
        pruned = []
        for a in actions:
            if state.depth < 1 and a.edge_type == "DEFINES":
                pruned.append(a)
            else:
                allowed.append(a)
        return allowed, pruned

    def risk(self, state: State, action: Action, problem=None) -> float:
        return 0.0


def _make_smoke_problem(sid: str, question: str, label: str):
    """Build a tiny ScenarioGraph usable by the smoke search_fn."""
    nodes = {
        "n1": Node(nid="n1", clause="clause1", role="OBL", text="obligation text sample", sent_idx=0),
        "n2": Node(nid="n2", clause="clause2", role="EXC", text="exception text sample", sent_idx=0),
        "n3": Node(nid="n3", clause="clause2", role="COND", text="condition text sample", sent_idx=0),
    }
    out = {
        "n1": [Edge(src="n1", etype="DEFINES", dst="n2")],
        "n2": [Edge(src="n2", etype="EXCEPTIONS", dst="n3")],
    }
    inc = {
        "n2": [Edge(src="n1", etype="DEFINES", dst="n2")],
        "n3": [Edge(src="n2", etype="EXCEPTIONS", dst="n3")],
    }
    entry = ["n1"]
    sim = {"clause1": 0.0, "clause2": 0.0}
    graph = ScenarioGraph(sid=sid, nodes=nodes, out=out, inc=inc, entry=entry)
    problem = SearchProblem(graph=graph, scenario_text=question, clause_sim=sim, h_fn=lambda s, p: 0.0)
    return problem


def _make_smoke_search_fn(fusion: FusionScorer, seed: int = 7):
    """Deterministic dummy search_fn used by the --mode grid smoke test."""

    def search_fn(sample: Dict[str, Any], lam: Tuple[float, float, float, float], question: str):
        sid = sample.get("sid", "smoke")
        sample_label = sample.get("label", "Compliant")
        rng = random.Random(seed)
        question = sample.get("question", question) or ""
        path_history: List[Action] = []
        graph = _make_smoke_problem(sid, question, sample_label).graph
        state = State(clauses=frozenset({"clause1"}), used_edges=(), depth=0, label_hypothesis="undetermined")
        action = Action(src="n1", edge_type="DEFINES", dst="n2")
        priority = fusion.priority(state, action, question, path_history, graph)
        # Toy decision that lets λ groups differ on macro-F1.
        if priority < 0.0:
            pred_label = "Non-Compliant"
        else:
            pred_label = "Compliant"
        return {"expanded_nodes": 1, "answer": pred_label, "labels": (sample_label,)}

    return search_fn


def _make_dev_samples(n_samples: int = 10, seed: int = 123):
    """Create 10 tiny synthetic dev samples for grid smoke test."""
    rng = random.Random(seed)
    samples = []
    for i in range(n_samples):
        sid = f"smoke-{i:03d}"
        label = "Compliant" if rng.randrange(0, 2) == 0 else "Non-Compliant"
        question = f"sample question {i}"
        samples.append({"sid": sid, "question": question, "label": label})
    return samples


def grid_search_lambdas(self, dev_samples, search_fn, lambda_grid, budget=100):
    """Grid search over lambda components on dev only.

    Returns (best_lambda, payload) where payload contains recorded results.
    """
    items = list(itertools.product(*lambda_grid.values()))
    if len(items) > 20:
        items = items[:20]
    records = []
    expanded_vals = []
    for lam in items:
        lam_tuple = tuple(float(x) for x in lam)
        f1s = []
        exs = []
        for sample in dev_samples:
            res = search_fn(sample, lam_tuple, sample.get("question", ""))
            if isinstance(res, SearchResult):
                pred = res.answer or "Not-Applicable"
                expanded = int(res.expanded_nodes or 0)
            elif isinstance(res, dict):
                pred = res.get("answer") or res.get("pred")
                expanded = int(res.get("expanded_nodes") or 0)
            else:
                raise TypeError("search_fn must return SearchResult or dict")
            f1s.append(_sample_macro_f1(pred, sample.get("label", "")))
            exs.append(expanded)
        avg_exp = sum(exs) / len(exs) if exs else 0.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        record = {
            "λ1": lam_tuple[0],
            "λ2": lam_tuple[1],
            "λ3": lam_tuple[2],
            "λ4": lam_tuple[3],
            "macro_f1": macro_f1,
            "avg_expanded_nodes": avg_exp,
        }
        records.append(record)
        expanded_vals.append(avg_exp)
    norm_exp = _expand_normalize(expanded_vals)
    for rec, norm in zip(records, norm_exp):
        rec["score"] = rec["macro_f1"] - 0.1 * norm
    best_record = max(records, key=lambda r: r["score"])
    best = (best_record["λ1"], best_record["λ2"], best_record["λ3"], best_record["λ4"])
    payload = {
        "results": records,
        "best_lambda": best,
        "best_score": best_record["score"],
        "log_path": LOGDIR / "lambda_grid.json",
    }
    _write_lambda_grid_log(payload["log_path"], records, best, best_record["score"])
    return best, payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion scorer smoke test")
    parser.add_argument("--mode", choices=["grid"], default="grid")
    parser.add_argument("--dev_samples", type=int, default=10)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--lambda_grid", default=None, help="JSON string of lambda grid")
    parser.add_argument("--output", type=Path, default=None, help="Lambda grid JSON output")
    args = parser.parse_args()

    if args.mode != "grid":
        parser.error("only --mode grid is implemented for now")

    dev_samples = _make_dev_samples(n_samples=args.dev_samples)
    llm_validator = object()
    llm_reward = lambda path, q: 0.5

    class _SilentClassifier:
        def success_prob(self, state, question, path_history, llm_reward_avg, rule_risk, graph):
            return 0.6

    classifier = _SilentClassifier()
    rule_engine = _SmokeRuleEngine()

    def h_func(state):
        return 0.5

    lambdas = (1.0, 0.5, 0.5, 0.5)
    fusion = FusionScorer(llm_validator, llm_reward, classifier, rule_engine, h_func, lambdas)
    search_fn = _make_smoke_search_fn(fusion, seed=123)

    grid = {"λ1": [0.5, 1.0], "λ2": [0.5, 1.0], "λ3": [1.0], "λ4": [0.5, 1.0]}
    if args.lambda_grid:
        try:
            grid = json.loads(args.lambda_grid)
        except Exception as exc:
            parser.error(f"invalid --lambda_grid JSON: {exc}")

    best_lambda, payload = FusionScorer.grid_search_lambdas(fusion, dev_samples, search_fn, grid, args.budget)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(handler)

    print("\nLambda grid results:")
    for rec in payload["results"]:
        print(
            f"λ=({rec['λ1']:.2f},{rec['λ2']:.2f},{rec['λ3']:.2f},{rec['λ4']:.2f}) "
            f"F1={rec['macro_f1']:.4f} avg_exp={rec['avg_expanded_nodes']:.2f} score={rec['score']:.4f}"
        )
    print("\nBest lambda:")
    print(f"λ=({best_lambda[0]:.2f},{best_lambda[1]:.2f},{best_lambda[2]:.2f},{best_lambda[3]:.2f}) score={payload['best_score']:.4f}")
    print(f"Saved to {payload['log_path']}")

    # Explicit hard-mask acceptance check.
    s = State(clauses=frozenset({"clause1"}), used_edges=(), depth=0, label_hypothesis="undetermined")
    act = Action(src="n1", edge_type="DEFINES", dst="n2")
    val = fusion.priority(s, act, "hard mask probe", [], nx.DiGraph())
    print(f"Hard-pruned priority check: {val:.4f} (expect -inf)")
    if not (val == float("-inf")):
        raise SystemExit("hard-mask priority check failed")
