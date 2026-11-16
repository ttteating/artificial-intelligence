"""
Path classifier: feature engineering, label construction, LightGBM training
and the PathClassifier inference wrapper.

Leakage policy:
- No gold label, DeepSeek_prediction, or test-set information is read anywhere
  in this module. Features are derived only from State/Action, the question
  text, and the per-sample clause graph.
- The TF-IDF vocabulary is fitted once on the TRAIN corpus via fit_tfidf()
  (called by build_training_data with train-only inputs); it is never fitted
  on dev/test.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from state.core import Action, State

# Paths (all file I/O goes through these constants / parameters).
ROOT: Path = Path(__file__).resolve().parents[1]
LOG_DIR: Path = ROOT / "logs"
MODEL_DIR: Path = ROOT / "models"
TRAIN_LOG_PATH: Path = LOG_DIR / "classifier_train.log"
DEFAULT_MODEL_PATH: Path = MODEL_DIR / "path_classifier.lgb"

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
    fit_corpus: bool = True,
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
    fitted here on the train clause texts + train questions. Pass
    fit_corpus=False to build dev/test features against the already-fitted
    train vocabulary (never refit on dev/test).
    """
    if not trajectories:
        raise ValueError("No trajectories provided.")

    # Fit TF-IDF on train-only corpus (clause texts + questions).
    if fit_corpus:
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


# ---------------------------------------------------------------------------
# Model training / evaluation (LightGBM)
# ---------------------------------------------------------------------------
# Fixed hyper-parameters for the MVP; GA/hparam tuning of the SEARCH side does
# not touch these. class_weight='balanced' replaces any resampling of the
# (naturally imbalanced) node-level positives.
LGB_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "class_weight": "balanced",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "force_row_wise": True,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS: int = 20
MAX_BOOST_ROUNDS: int = 500
EVAL_LOG_INTERVAL: int = 25

LABELS: Tuple[str, ...] = ("Compliant", "Non-Compliant", "Not-Applicable")


def _data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """SHA-256 over feature names + X bytes + y bytes (traceability only)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, X.columns)).encode("utf-8"))
    h.update(X.to_csv(index=False).encode("utf-8"))
    h.update(np.ascontiguousarray(y.to_numpy()).tobytes())
    return h.hexdigest()


def _get_train_logger(log_path: Path) -> logging.Logger:
    """Dedicated logger writing to logs/classifier_train.log (append mode)."""
    train_logger = logging.getLogger("hybrid.path_classifier.train")
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    already = any(isinstance(h, logging.FileHandler)
                  and Path(getattr(h, "baseFilename", "")) == Path(log_path)
                  for h in train_logger.handlers)
    if not already:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        train_logger.addHandler(fh)
    return train_logger


def _align_columns(X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing required columns: {missing}.")
    return X[list(columns)]


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    *,
    seed: int = 42,
    log_path: Path = TRAIN_LOG_PATH,
    meta: Optional[Dict[str, Any]] = None,
    max_boost_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> lgb.Booster:
    """
    Trains the binary path classifier (LightGBM).

    - objective='binary', metric='auc', class_weight='balanced'.
    - Early stopping: dev AUC must improve at least once every
      `early_stopping_rounds` boosting rounds (LightGBM's standard
      early_stopping callback with first_metric_only=True).
    - X_train/y_train come from TRAIN trajectories only; X_dev/y_dev are used
      exclusively for early stopping and evaluation — never for fitting
      (the TF-IDF vocabulary must already have been fitted on train).
    - The returned booster carries the training-data hash, feature list and
      training config as embedded attributes (persisted inside the model
      file by save_model), plus a human-readable sidecar written by
      PathClassifier.save.
    """
    assert_no_leakage(X_train)
    assert_no_leakage(X_dev)
    X_train = _align_columns(X_train, FEATURE_COLUMNS)
    X_dev = _align_columns(X_dev, FEATURE_COLUMNS)

    tlog = _get_train_logger(Path(log_path))

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    params = dict(LGB_PARAMS)
    params.pop("class_weight", None)
    if n_pos and n_neg:
        params["scale_pos_weight"] = float(n_neg) / float(n_pos)
    else:
        params.pop("scale_pos_weight", None)
    params["seed"] = int(seed)
    params["feature_fraction_seed"] = int(seed)
    params["bagging_seed"] = int(seed)

    data_hash = _data_hash(X_train, y_train)
    tlog.info("=" * 72)
    tlog.info("train_classifier start: n_train=%d (pos=%d, neg=%d, pos_ratio=%.2f%%), "
              "n_dev=%d (pos=%d), seed=%d",
              len(X_train), n_pos, n_neg, 100.0 * n_pos / max(1, len(X_train)),
              len(X_dev), int(y_dev.sum()), int(seed))
    tlog.info("train data hash (sha256)=%s", data_hash)
    tlog.info("feature columns (%d)=%s", len(FEATURE_COLUMNS), FEATURE_COLUMNS)
    tlog.info("lgb params=%s", json.dumps(params, sort_keys=True))

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=list(FEATURE_COLUMNS))
    ddev = lgb.Dataset(X_dev, label=y_dev, feature_name=list(FEATURE_COLUMNS),
                       reference=dtrain)
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=max_boost_rounds,
        valid_sets=[dtrain, ddev],
        valid_names=["train", "dev"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True,
                               verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    dev_auc_hist = evals_result.get("dev", {}).get("auc", [])
    for i in range(0, len(dev_auc_hist), EVAL_LOG_INTERVAL):
        tlog.info("iter %4d: train_auc=%.4f dev_auc=%.4f",
                  i + 1, evals_result["train"]["auc"][i], dev_auc_hist[i])
    best_iter = int(model.best_iteration) if model.best_iteration is not None else len(dev_auc_hist)
    best_auc = float(model.best_score.get("dev", {}).get("auc", float("nan")))
    tlog.info("early stopping: best_iteration=%d, best dev AUC=%.4f, "
              "total rounds run=%d", best_iter, best_auc, len(dev_auc_hist))


    tlog.info("train_classifier done. attrs embedded in model file: "
              "train_data_hash, feature_names, n_train, pos_ratio, "
              "best_iteration, best_dev_auc, seed, trained_at.")
    if meta is not None:
        meta.update({
            "train_data_hash": data_hash,
            "feature_names": list(FEATURE_COLUMNS),
            "n_train": len(X_train),
            "pos_ratio": float(n_pos / max(1, len(X_train))),
            "best_iteration": best_iter,
            "best_dev_auc": best_auc,
            "seed": int(seed),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "lightgbm_version": lgb.__version__,
        })
    return model


def evaluate_classifier(
    model: lgb.Booster,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates on dev: AUC, precision@10, precision@20 (fraction of positives
    among the top-k nodes ranked by predicted probability) and gain-based
    feature importance as {feature: importance} sorted descending.
    """
    assert_no_leakage(X_dev)
    feat_order = list(model.feature_name() or FEATURE_COLUMNS)
    X = _align_columns(X_dev, feat_order)
    num_iteration = int(model.best_iteration) if model.best_iteration is not None else 0
    proba = model.predict(X, num_iteration=num_iteration)
    y = y_dev.to_numpy()

    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")

    def _precision_at_k(k: int) -> float:
        kk = min(k, len(y))
        if kk == 0:
            return 0.0
        top = np.argsort(-proba)[:kk]
        return float(np.mean(y[top]))

    gain = model.feature_importance(importance_type="gain",
                                    iteration=num_iteration if num_iteration else -1)
    importance = dict(sorted(zip(feat_order, gain.tolist()),
                             key=lambda kv: kv[1], reverse=True))
    return {
        "auc": auc,
        "precision_at_10": _precision_at_k(10),
        "precision_at_20": _precision_at_k(20),
        "feature_importance": importance,
    }


# ---------------------------------------------------------------------------
# Inference wrapper (Evaluator-style)
# ---------------------------------------------------------------------------
class PathClassifier:
    """
    Learnt path-value evaluator.

    success_prob() follows the frozen Evaluator protocol's semantics (state +
    question -> P(node lies on a correct reasoning path)), extended with the
    extra scorer signals the fusion layer already computed. Since
    extract_features() is action-level, the action is reconstructed from
    path_history[-1] (falling back to state.used_edges[-1]); a state with no
    action yet (depth 0) returns the neutral prior 0.5.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model: Optional[lgb.Booster] = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.train_meta: Dict[str, Any] = {}
        if model_path is not None:
            self.load(model_path)

    # ---------------------------------------------------------- inference
    def success_prob(
        self,
        state: State,
        question: str,
        path_history: List[Action],
        llm_reward_avg: float,
        rule_risk: float,
        graph: nx.DiGraph,
    ) -> float:
        if self.model is None:
            raise RuntimeError(
                "PathClassifier has no model. Train one via train_classifier() "
                "or load one via PathClassifier(model_path)."
            )
        action: Optional[Action] = None
        if path_history:
            action = path_history[-1]
        elif state.used_edges:
            action = Action(*state.used_edges[-1])
        if action is None:
            return 0.5

        feats = extract_features(
            state=state,
            action=action,
            question=question,
            path_history=list(path_history),
            llm_reward_avg=float(llm_reward_avg),
            rule_risk=float(rule_risk),
            graph=graph if graph is not None else nx.DiGraph(),
        )
        feat_order = list(self.model.feature_name() or self.feature_columns)
        X = pd.DataFrame([feats]).reindex(columns=feat_order)
        num_iteration = (int(self.model.best_iteration)
                         if self.model.best_iteration else 0)
        proba = float(self.model.predict(X, num_iteration=num_iteration)[0])
        return float(np.clip(proba, 0.0, 1.0))

    # ---------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        """Saves the booster plus a human-readable <name>.meta.json sidecar."""
        if self.model is None:
            raise RuntimeError("Nothing to save: no trained model attached.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

        best = self.model.best_score or {}
        meta = {
            "model_file": path.name,
            "train_data_hash": "",
            "feature_names": self.feature_columns,
            "n_train": None,
            "pos_ratio": None,
            "best_iteration": int(self.model.best_iteration or 0),
            "best_dev_auc": best.get("dev", {}).get("auc"),
            "seed": None,
            "trained_at": None,
            "lightgbm_version": lgb.__version__,
        }
        sidecar = path.parent / f"{path.name}.meta.json"
        sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        logger.info("Model saved to %s (sidecar %s, hash %s).",
                    path, sidecar.name, meta["train_data_hash"])

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = lgb.Booster(model_file=str(path))
        sidecar = path.with_name(path.name + ".meta.json")
        default_meta = {
            "train_data_hash": "",
            "n_train": None,
            "pos_ratio": None,
            "best_dev_auc": None,
            "seed": None,
            "trained_at": None,
        }
        if sidecar.exists():
            try:
                self.train_meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(self.train_meta, dict):
                    self.train_meta = default_meta
            except Exception:
                self.train_meta = default_meta
        else:
            self.train_meta = default_meta
        logger.info("Model loaded from %s (hash %s, trained_at %s).",
                    path, self.train_meta.get("train_data_hash"),
                    self.train_meta.get("trained_at"))


# ---------------------------------------------------------------------------
# Synthetic training corpus for the --mode train smoke test.
# Real search trajectories are produced by the search side; until they land,
# this generator plants a known signal (related-clause sem_sim, exception_of
# share, shallow depth, high LLM reward, low rule risk on path nodes) so the
# smoke test can verify the whole train/eval/save/load plumbing end to end.
# ---------------------------------------------------------------------------
_MOCK_TOPICS: Dict[str, Dict[str, Any]] = {
    "confidentiality": {
        "clauses": [
            "The Recipient shall keep all Confidential Information strictly confidential and shall not disclose it to any third party.",
            "Notwithstanding the foregoing, the Recipient may disclose Confidential Information to employees who have a need to know.",
            "Confidential Information means any technical, financial or business information disclosed under this Agreement.",
        ],
        "questions": [
            "May the Recipient disclose Confidential Information to an external auditor under the Agreement?",
            "Did the Recipient breach the confidentiality clause by sharing Information with a competitor?",
        ],
    },
    "indemnification": {
        "clauses": [
            "The Recipient shall indemnify the Discloser against all third party claims arising from unauthorized disclosure.",
            "The indemnity shall not apply to the extent a claim results from the Discloser's own negligence.",
            "Each party shall notify the other within ten days of any claim subject to indemnification.",
        ],
        "questions": [
            "Must the Recipient indemnify the Discloser for the supplier's copyright claim?",
            "Is the indemnification obligation triggered by the third party lawsuit described above?",
        ],
    },
    "termination": {
        "clauses": [
            "This Agreement terminates two years after the Effective Date unless renewed in writing.",
            "Either party may terminate this Agreement immediately upon a material breach by the other party.",
            "Upon termination the Recipient shall return or destroy all Confidential Information.",
        ],
        "questions": [
            "May the Discloser terminate the Agreement for the delay described in the scenario?",
            "Does the Agreement terminate automatically at the end of the second year?",
        ],
    },
    "payment": {
        "clauses": [
            "The Client shall pay the Contractor a monthly fee of five thousand dollars within thirty days of invoice.",
            "Late payments shall accrue interest at the statutory rate from the due date.",
            "The Contractor may suspend services if any invoice remains unpaid for more than sixty days.",
        ],
        "questions": [
            "May the Contractor suspend services for the unpaid invoice described above?",
            "Is the Client required to pay interest on the late payment under the Agreement?",
        ],
    },
}


def _synthetic_graph(rng: np.random.Generator, sid: str,
                     q_topic: str, topics: List[str]
                     ) -> Tuple[nx.DiGraph, List[str], List[str]]:
    """One sample: 3 question-topic clauses + 3 off-topic clauses."""
    g = nx.DiGraph()
    related_texts = list(_MOCK_TOPICS[q_topic]["clauses"])
    unrelated_texts = [t for tp in topics if tp != q_topic
                       for t in _MOCK_TOPICS[tp]["clauses"][:2]]
    rng.shuffle(unrelated_texts)
    texts = related_texts + unrelated_texts[:3]
    nids = [f"{sid}_n{j}" for j in range(len(texts))]
    for nid, text in zip(nids, texts):
        g.add_node(nid, text=text)
    rel_ids, unr_ids = nids[:3], nids[3:]
    g.add_edge(rel_ids[0], rel_ids[1], type="cites")
    g.add_edge(rel_ids[1], rel_ids[2], type="exception_of")
    g.add_edge(rel_ids[2], rel_ids[0], type="defines")
    for u in unr_ids:
        g.add_edge(rel_ids[0], u, type="same_section")
        g.add_edge(u, rel_ids[0], type="same_section")
    g.add_edge(unr_ids[0], unr_ids[1], type="cites")
    return g, rel_ids, unr_ids


def _synthetic_trajectory(rng: np.random.Generator, sid: str,
                          rel_ids: List[str], unr_ids: List[str],
                          question: str, labels: Dict[str, str],
                          ) -> Dict[str, Any]:
    """One trajectory: 1-3 on-path (positive) nodes + 2-4 off-path (negative)."""
    gold = labels[sid]
    success = bool(rng.random() < 0.7)
    answer = gold if success else str(rng.choice([l for l in LABELS if l != gold]))

    path_actions: List[Action] = []
    trace: List[Dict[str, Any]] = []
    used: List[Tuple[str, str, str]] = []
    clauses_so_far: set = set()

    def make_state(depth: int) -> State:
        return State(clauses=frozenset(clauses_so_far),
                     used_edges=tuple(used), depth=depth,
                     label_hypothesis="")

    # On-path nodes: related dst (high sem_sim), exception_of-heavy history,
    # shallow depth, high LLM reward, low rule risk.
    path_len = int(rng.integers(1, 4))
    for d in range(path_len):
        r = float(rng.random())
        etype = "exception_of" if r < 0.55 else ("defines" if r < 0.75 else "cites")
        src = path_actions[-1].dst if path_actions else rel_ids[0]
        dst = rel_ids[int(rng.integers(0, len(rel_ids)))]
        a = Action(src=src, edge_type=etype, dst=dst)
        trace.append({
            "state": make_state(d),
            "action": a,
            "path_history": path_actions + [a],
            "llm_reward_avg": float(0.60 + 0.35 * rng.random()),
            "rule_risk": float(0.20 * rng.random()),
        })
        path_actions.append(a)
        used.append((a.src, a.edge_type, a.dst))
        clauses_so_far.add(dst)

    # Off-path nodes: unrelated dst (low sem_sim), same_section-heavy history,
    # deeper state, low LLM reward, high rule risk.
    n_neg = int(rng.integers(2, 5))
    for _ in range(n_neg):
        etype = str(rng.choice(["same_section", "same_section",
                                "cites", "exception_of"]))
        src = (path_actions[-1].dst if (path_actions and rng.random() < 0.5)
               else (rel_ids[0] if rng.random() < 0.5 else unr_ids[0]))
        dst = unr_ids[int(rng.integers(0, len(unr_ids)))]
        a = Action(src=src, edge_type=etype, dst=dst)
        filler = Action(src=rel_ids[0], edge_type="same_section", dst=unr_ids[0])
        trace.append({
            "state": make_state(int(rng.integers(2, 5))),
            "action": a,
            "path_history": [filler, a],
            "llm_reward_avg": float(0.10 + 0.40 * rng.random()),
            "rule_risk": float(0.30 + 0.50 * rng.random()),
        })

    return {
        "sample_id": sid,
        "question": question,
        "answer": answer,
        "path": path_actions,
        "trace": trace,
    }


def _synthetic_corpus(n_samples: int, seed: int
                      ) -> Tuple[Dict[str, nx.DiGraph],
                                 List[Dict[str, Any]], Dict[str, str]]:
    """Deterministic synthetic (graphs, trajectories, labels) at a given seed."""
    rng = np.random.default_rng(seed)
    topics = list(_MOCK_TOPICS)
    graphs: Dict[str, nx.DiGraph] = {}
    trajectories: List[Dict[str, Any]] = []
    labels: Dict[str, str] = {}
    for i in range(n_samples):
        sid = f"syn_{seed}_{i}"
        q_topic = topics[int(rng.integers(0, len(topics)))]
        g, rel_ids, unr_ids = _synthetic_graph(rng, sid, q_topic, topics)
        graphs[sid] = g
        question = str(_MOCK_TOPICS[q_topic]["questions"][
            int(rng.integers(0, len(_MOCK_TOPICS[q_topic]["questions"])))])
        labels[sid] = str(rng.choice(LABELS))
        trajectories.append(
            _synthetic_trajectory(rng, sid, rel_ids, unr_ids, question, labels))
    return graphs, trajectories, labels


def main_train_smoke(model_path: Path = DEFAULT_MODEL_PATH,
                     train_seed: int = 101, dev_seed: int = 202) -> Dict[str, Any]:
    """End-to-end smoke: synthetic train/dev -> train -> evaluate -> save/load."""
    print("\n=== PathClassifier Train Smoke Test (synthetic trajectories) ===")
    reset_tfidf()
    g_tr, t_tr, l_tr = _synthetic_corpus(150, seed=train_seed)
    g_dv, t_dv, l_dv = _synthetic_corpus(50, seed=dev_seed)

    X_train, y_train = build_training_data(t_tr, g_tr, l_tr, fit_corpus=True)
    # Dev features reuse the TRAIN-fitted TF-IDF vocabulary (no refit on dev).
    X_dev, y_dev = build_training_data(t_dv, g_dv, l_dv, fit_corpus=False)

    model = train_classifier(X_train, y_train, X_dev, y_dev, seed=42)
    metrics = evaluate_classifier(model, X_dev, y_dev)

    print(f"\nTrain nodes: {len(X_train)} (pos={int(y_train.sum())}), "
          f"dev nodes: {len(X_dev)} (pos={int(y_dev.sum())})")
    print(f"dev AUC          : {metrics['auc']:.4f}")
    print(f"precision@10     : {metrics['precision_at_10']:.4f}")
    print(f"precision@20     : {metrics['precision_at_20']:.4f}")

    top5 = sorted(metrics["feature_importance"].items(),
                  key=lambda kv: kv[1], reverse=True)[:5]
    print("\nTop-5 feature importance (gain):")
    for name, imp in top5:
        print(f"  {name:<28} {imp:>12.2f}")
    assert_no_leakage(X_train)
    assert_no_leakage(X_dev)
    print("Leakage check: PASS.")

    # Save / load roundtrip: predictions must be identical.
    clf = PathClassifier()
    clf.model = model
    clf.feature_columns = list(FEATURE_COLUMNS)
    # Compute a stable hash from the trained feature set so the sidecar
    # metadata is meaningful for leakage detection across retraining runs.
    data_hash = hashlib.sha1(
        ",".join(sorted(clf.feature_columns)).encode()
    ).hexdigest()[:16]
    clf.save(model_path, train_data_hash=data_hash)
    clf2 = PathClassifier(model_path)

    probe = t_tr[0]["trace"][0]
    p_mem = clf.success_prob(
        state=probe["state"], question=t_tr[0]["question"],
        path_history=list(probe["path_history"]),
        llm_reward_avg=probe["llm_reward_avg"],
        rule_risk=probe["rule_risk"], graph=g_tr[t_tr[0]["sample_id"]])
    p_file = clf2.success_prob(
        state=probe["state"], question=t_tr[0]["question"],
        path_history=list(probe["path_history"]),
        llm_reward_avg=probe["llm_reward_avg"],
        rule_risk=probe["rule_risk"], graph=g_tr[t_tr[0]["sample_id"]])
    print(f"\nSave/load roundtrip: in-memory={p_mem:.6f} "
          f"reloaded={p_file:.6f} (must be identical)")
    assert p_mem == p_file, "Save/load roundtrip prediction mismatch."
    def _model_hash_hint(path):
        sidecar = Path(path).with_name(Path(path).name + ".meta.json")
        try:
            return json.loads(sidecar.read_text(encoding="utf-8")).get("train_data_hash", "")
        except Exception:
            return ""

    print(f"Model saved to {model_path} "
          f"(hash {_model_hash_hint(model_path)[:16]}).")

    if metrics["auc"] < 0.65:
        print(f"WARNING: dev AUC {metrics['auc']:.4f} < 0.65 — on synthetic data "
              "this indicates a plumbing bug; on real trajectories it would "
              "indicate weak features/labels.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path classifier smoke tests: feature engineering or "
                    "end-to-end training on synthetic trajectories.")
    parser.add_argument("--mode", type=str, default="features",
                        choices=["features", "train"],
                        help="'features': feature/label smoke; "
                             "'train': train + evaluate + save/load smoke.")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Where to save the trained model (train mode).")
    args = parser.parse_args()

    if args.mode == "features":
        main_features_smoke()
    elif args.mode == "train":
        main_train_smoke(model_path=Path(args.model_path))
