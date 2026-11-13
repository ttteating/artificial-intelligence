"""
Path classifier feature engineering and label construction (no model training yet).

Leakage policy:
- No gold label, DeepSeek_prediction, or test-set information is read anywhere
  in this module. Features are derived only from State/Action, the question
  text, and the per-sample clause graph.
- The TF-IDF vocabulary is fitted once on the TRAIN corpus via fit_tfidf()
  (called by build_training_data with train-only inputs); it is never fitted
  on dev/test.
"""

import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from state.interface import Action, State

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EDGE_TYPES: Tuple[str, ...] = ("cites", "exception_of", "conditions_on", "defines", "supersedes", "same_section")
LABEL_HYPOTHESIS_ENCODING: Dict[str, int] = {
    "": 0,
    "undetermined": 0,
    "Compliant": 1,
    "Non-Compliant": 2,
    "Not-Applicable": 3,
}

# Column-name leakage patterns (case-insensitive substrings).
_FORBIDDEN_COL_SUBSTRINGS: Tuple[str, ...] = ("answer", "gd_tr", "deepseek_prediction")
# The spec mandates label_hypothesis_encoded as a feature (part of the frozen
# State contract); it encodes the *hypothesis*, never the gold answer.
_ALLOWED_LABEL_COLUMN = "label_hypothesis_encoded"


class FeatureLeakageError(ValueError):
    """Raised when a feature column looks like it leaks answer information."""


# ---------------------------------------------------------------------------
# TF-IDF (vocabulary fitted on train only)
# ---------------------------------------------------------------------------
_tfidf_vectorizer: Optional[TfidfVectorizer] = None


def fit_tfidf(corpus: List[str]) -> None:
    """
    Fits the module-level TF-IDF vectorizer. Must be called with a TRAIN-ONLY
    corpus; extract_features() refuses to compute sem_sim before this is fitted.
    """
    global _tfidf_vectorizer
    cleaned = [t for t in corpus if t and t.strip()]
    if not cleaned:
        raise ValueError("fit_tfidf received an empty corpus.")
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    _tfidf_vectorizer.fit(cleaned)
    logger.info(f"TF-IDF fitted on {len(cleaned)} train documents, "
                f"vocab_size={len(_tfidf_vectorizer.vocabulary_)}.")


def reset_tfidf() -> None:
    """Drops the fitted vectorizer (used by smoke tests to reset state)."""
    global _tfidf_vectorizer
    _tfidf_vectorizer = None


def _semantic_similarity(question: str, clause_text: str) -> float:
    """Cosine similarity between question and clause text under the fitted TF-IDF."""
    if not question.strip() or not clause_text.strip():
        return 0.0
    if _tfidf_vectorizer is None:
        logger.warning("TF-IDF not fitted yet; sem_sim falls back to 0.0. "
                       "Call fit_tfidf() on the train corpus first.")
        return 0.0
    vecs = _tfidf_vectorizer.transform([question, clause_text])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def _node_text(graph: nx.DiGraph, node: str) -> str:
    """Reads clause text stored as a node attribute ('text' or 'text_snippet')."""
    if not node or not graph.has_node(node):
        return ""
    data = graph.nodes[node]
    return str(data.get("text") or data.get("text_snippet") or "")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(
    state: State,
    action: Action,
    question: str,
    path_history: List[Action],
    llm_reward_avg: float,
    rule_risk: float,
    graph: nx.DiGraph,
) -> Dict[str, float]:
    """
    Extracts one interpretable feature vector for a search expansion node.

    Only uses: state, action, the question text, the path history, the two
    scorer signals, and clause-graph structure/text. Never touches gold
    labels, DeepSeek_prediction, or anything from the test split.
    """
    denom = max(state.depth, 1)
    n_exception = sum(1 for a in path_history if a.edge_type == "exception_of")
    n_condition = sum(1 for a in path_history if a.edge_type == "conditions_on")

    features: Dict[str, float] = {
        "depth": float(state.depth),
        "num_clauses": float(len(state.clauses)),
        "num_used_edges": float(len(state.used_edges)),
        "exception_ratio": n_exception / denom,
        "condition_ratio": n_condition / denom,
        "sem_sim": _semantic_similarity(question, _node_text(graph, action.dst)),
        "dst_out_degree": float(graph.out_degree(action.dst)) if graph.has_node(action.dst) else 0.0,
        "dst_in_degree": float(graph.in_degree(action.dst)) if graph.has_node(action.dst) else 0.0,
        "llm_reward_avg": float(llm_reward_avg),
        "rule_risk": float(rule_risk),
        "has_supersedes": 1.0 if any(a.edge_type == "supersedes" for a in path_history) else 0.0,
        "label_hypothesis_encoded": float(
            LABEL_HYPOTHESIS_ENCODING.get(state.label_hypothesis, 0)
        ),
    }
    # One-hot of the current edge type over the 6 graph edge types.
    # Termination actions get an all-zero vector by design.
    for et in EDGE_TYPES:
        features[f"edge_type_{et}"] = 1.0 if action.edge_type == et else 0.0
    return features


FEATURE_COLUMNS: List[str] = [
    "depth", "num_clauses", "num_used_edges",
    *[f"edge_type_{et}" for et in EDGE_TYPES],
    "exception_ratio", "condition_ratio", "sem_sim",
    "dst_out_degree", "dst_in_degree",
    "llm_reward_avg", "rule_risk",
    "has_supersedes", "label_hypothesis_encoded",
]


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------
def assert_no_leakage(features: pd.DataFrame) -> None:
    """
    Raises FeatureLeakageError if the feature matrix shows signs of leakage:
    - forbidden substrings ("answer", "gd_tr", "deepseek_prediction") in any
      column name; "label" is only allowed as the exact mandated column
      'label_hypothesis_encoded';
    - non-numeric columns;
    - corrupt sem_sim values (NaN / outside cosine range) — sem_sim is computed
      solely from the question and clause text inside extract_features(), so
      out-of-range values indicate tampering.
    """
    if features.empty:
        raise FeatureLeakageError("Feature matrix is empty.")
    for col in features.columns:
        low = col.lower()
        for bad in _FORBIDDEN_COL_SUBSTRINGS:
            if bad in low:
                raise FeatureLeakageError(f"Forbidden substring '{bad}' in column '{col}'.")
        if "label" in low and low != _ALLOWED_LABEL_COLUMN:
            raise FeatureLeakageError(
                f"Column '{col}' contains 'label'; only '{_ALLOWED_LABEL_COLUMN}' is permitted."
            )
    non_numeric = list(features.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise FeatureLeakageError(f"Non-numeric columns found: {non_numeric}.")
    if "sem_sim" in features.columns:
        s = features["sem_sim"]
        if s.isna().any() or not s.between(-1.0001, 1.0001).all():
            raise FeatureLeakageError("sem_sim contains NaN or out-of-cosine-range values.")


# ---------------------------------------------------------------------------
# Training data construction (train trajectories only)
# ---------------------------------------------------------------------------
def build_training_data(
    trajectories: List[Dict[str, Any]],
    graph_dict: Dict[str, nx.DiGraph],
    labels: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Builds (X, y) from search trajectories.

    Expected trajectory schema (from the search-side logs):
        {
          "sample_id": str,
          "question": str,                    # scenario text of the sample
          "answer": str,                      # the search's final answer
          "path": list[Action],               # final path chosen by the search
          "trace": [                          # one entry per expanded node
            {
              "state": State,
              "action": Action,               # the expansion taken at this node
              "path_history": list[Action],   # actions up to and including this one
              "llm_reward_avg": float,        # sliding mean of LLM path rewards
              "rule_risk": float,             # rule-engine risk of this action
            },
            ...
          ],
        }

    Label backtracking: for a trajectory whose answer matches the gold label
    (labels[sample_id]), a trace node is positive iff its action lies on the
    final path; nodes off the final path are negative. Failed trajectories
    contribute only negatives. The subtree-containment rule from the spec is
    approximated by on-final-path membership because the flat trace format
    carries no parent/child structure — confirm against the real search logs.

    graph_dict must contain TRAIN samples only; the TF-IDF vocabulary is
    fitted here on the train clause texts + train questions.
    """
    if not trajectories:
        raise ValueError("No trajectories provided.")

    # Fit TF-IDF on train-only corpus (clause texts + questions).
    corpus: List[str] = []
    for g in graph_dict.values():
        for _n, data in g.nodes(data=True):
            t = str(data.get("text") or data.get("text_snippet") or "")
            if t.strip():
                corpus.append(t)
    for traj in trajectories:
        q = str(traj.get("question", ""))
        if q.strip():
            corpus.append(q)
    fit_tfidf(corpus)

    rows: List[Dict[str, float]] = []
    targets: List[int] = []
    n_pos = n_neg = 0

    for traj in trajectories:
        sample_id = str(traj["sample_id"])
        graph = graph_dict.get(sample_id, nx.DiGraph())
        question = str(traj.get("question", ""))
        gold = labels.get(sample_id)
        success = (traj.get("answer") == gold)
        path_keys = {(a.src, a.edge_type, a.dst) for a in traj.get("path", [])}

        for node in traj.get("trace", []):
            action: Action = node["action"]
            feats = extract_features(
                state=node["state"],
                action=action,
                question=question,
                path_history=list(node.get("path_history", [])),
                llm_reward_avg=float(node.get("llm_reward_avg", 0.0)),
                rule_risk=float(node.get("rule_risk", 0.0)),
                graph=graph,
            )
            on_path = (action.src, action.edge_type, action.dst) in path_keys
            y = 1 if (success and on_path) else 0
            rows.append(feats)
            targets.append(y)
            if y == 1:
                n_pos += 1
            else:
                n_neg += 1

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = pd.Series(targets, name="on_final_path", dtype=int)

    total = n_pos + n_neg
    ratio = n_pos / total if total else 0.0
    logger.info(f"Built training data: {total} nodes (pos={n_pos}, neg={n_neg}, "
                f"pos_ratio={ratio:.2%}). No resampling performed; use class_weight at train time.")
    return X, y


# ---------------------------------------------------------------------------
# Smoke test fixtures (mock trajectories; real dev trajectories from the
# search side are not available yet — see module docstring note in summary)
# ---------------------------------------------------------------------------
def _build_mock_graphs() -> Dict[str, nx.DiGraph]:
    clauses = {
        "evidence_1": "The Recipient shall keep the Information confidential and shall not disclose it.",
        "evidence_2": "Information means any technical or business data disclosed under this Agreement.",
        "evidence_3": "Notwithstanding the foregoing, disclosure to employees with a need to know is permitted.",
        "evidence_4": "This Agreement terminates two years after the Effective Date.",
        "evidence_5": "The Recipient shall indemnify the Discloser against all third party claims.",
    }
    g = nx.DiGraph()
    for cid, text in clauses.items():
        g.add_node(cid, text=text)
    g.add_edge("evidence_1", "evidence_2", type="cites")
    g.add_edge("evidence_2", "evidence_3", type="exception_of")
    g.add_edge("evidence_1", "evidence_4", type="same_section")
    g.add_edge("evidence_1", "evidence_5", type="same_section")
    return {"s1": g, "s2": g}


def _node(state_clauses, used_edges, depth, hypothesis, action, path_history,
          llm_reward_avg, rule_risk) -> Dict[str, Any]:
    return {
        "state": State(
            clauses=frozenset(state_clauses),
            used_edges=tuple(used_edges),
            depth=depth,
            label_hypothesis=hypothesis,
        ),
        "action": action,
        "path_history": path_history,
        "llm_reward_avg": llm_reward_avg,
        "rule_risk": rule_risk,
    }


def _build_mock_trajectories() -> List[Dict[str, Any]]:
    a_cites = Action(src="evidence_1", edge_type="cites", dst="evidence_2")
    a_exc = Action(src="evidence_2", edge_type="exception_of", dst="evidence_3")
    a_same = Action(src="evidence_1", edge_type="same_section", dst="evidence_4")
    a_def = Action(src="evidence_2", edge_type="defines", dst="evidence_3")

    traj_success = {
        "sample_id": "s1",
        "question": "Is the Recipient allowed to share the data with a junior assistant under the exception?",
        "answer": "Compliant",
        "path": [a_cites, a_exc],
        "trace": [
            _node({"evidence_1"}, [], 0, "", a_cites, [a_cites], 0.7, 0.0),
            # depth=2, path_history has exactly 1 exception_of -> exception_ratio 0.5 (hand check)
            _node({"evidence_1", "evidence_2"}, [("evidence_1", "cites", "evidence_2")],
                  2, "", a_exc, [a_cites, a_exc], 0.8, 0.0),
            _node({"evidence_1", "evidence_2"}, [("evidence_1", "cites", "evidence_2")],
                  2, "", a_same, [a_cites], 0.4, 0.3),
        ],
    }
    traj_failed = {
        "sample_id": "s2",
        "question": "Does the confidentiality clause define the scope of Information?",
        "answer": "Non-Compliant",  # != gold -> failed trajectory, all-negative
        "path": [a_def],
        "trace": [
            _node({"evidence_2"}, [], 1, "", a_def, [a_def], 0.5, 0.3),
            _node({"evidence_2"}, [], 1, "undetermined", a_same, [a_def, a_same], 0.3, 0.0),
        ],
    }
    return [traj_success, traj_failed]


def main_features_smoke() -> None:
    reset_tfidf()
    graphs = _build_mock_graphs()
    trajectories = _build_mock_trajectories()
    labels = {"s1": "Compliant", "s2": "Compliant"}

    print("\n=== PathClassifier Feature Smoke Test (mock trajectories) ===")
    X, y = build_training_data(trajectories, graphs, labels)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Columns ({len(X.columns)}): {list(X.columns)}")

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"Label distribution: pos={n_pos}, neg={n_neg}, pos_ratio={n_pos / len(y):.2%}")

    # Hand check: exception_ratio of the 2nd trace node of traj s1.
    # path_history = [cites, exception_of] -> 1 exception_of; depth = 2 -> 1/2 = 0.5
    hand_computed = 0.5
    computed = float(X.loc[1, "exception_ratio"])
    print(f"\nHand check: exception_ratio row 1 -> computed={computed}, expected={hand_computed}")
    assert computed == hand_computed, "Hand check failed for exception_ratio."

    # Hand check: edge_type_onehot for the exception_of row.
    onehot_sum = float(X.loc[1, [f"edge_type_{et}" for et in EDGE_TYPES]].sum())
    print(f"Hand check: edge_type one-hot sum row 1 -> {onehot_sum} (expected 1.0)")
    assert onehot_sum == 1.0

    # Hand check: sem_sim is within cosine range and non-negative.
    sem = X["sem_sim"]
    print(f"Hand check: sem_sim in [{sem.min():.3f}, {sem.max():.3f}] (cosine range [-1, 1])")

    assert_no_leakage(X)
    print("Leakage check: PASS (assert_no_leakage raised no exception)")

    if 0.2 <= n_pos / len(y) <= 0.4:
        print("Pos ratio within expected 20%-40% band.")
    else:
        print(f"Note: pos ratio {n_pos / len(y):.2%} outside 20%-40% (mock data only).")

    print("\nSample rows:")
    print(X.head(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path classifier feature engineering smoke test.")
    parser.add_argument("--mode", type=str, default="features", choices=["features"],
                        help="Smoke test mode. Only 'features' is implemented (no training yet).")
    args = parser.parse_args()

    if args.mode == "features":
        main_features_smoke()
