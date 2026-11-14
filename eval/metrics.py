"""指标：macro-F1、各类 recall、平均扩展节点、推理步数、冗余路径占比、LLM 调用。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABELS = ("Compliant", "Non-Compliant", "Not-Applicable")


@dataclass
class Row:
    sid: str
    condition: str
    gold: str
    answer: str
    expanded: int
    steps: int
    redundant: int
    llm_calls: int
    n_clauses: int
    oracle_depth: int
    reached_goal: bool
    pruned: int = 0
    llm_rejected: int = 0
    conf: float = 0.0
    score: float = 0.0

    @property
    def correct(self) -> bool:
        return self.answer == self.gold

    def to_dict(self) -> dict:
        return asdict(self)


def redundant_count(result) -> int:
    """冗余节点数：被扩展、但最终没有出现在返回路径上的节点。

    这是一个"解释成本"指标：这些人读过的节点在最终理由里没有被引用。
    """
    root = result.tree_root
    if root is None or not hasattr(root, "nodes"):
        return 0
    return sum(1 for n in root.nodes if "redundant" in n.flags)


def result_to_row(result, gold: str, condition: str, oracle_depth: int = 0) -> Row:
    return Row(
        sid=result.sid, condition=condition, gold=gold, answer=result.answer,
        expanded=result.expanded_nodes, steps=result.steps,
        redundant=redundant_count(result), llm_calls=result.llm_calls,
        n_clauses=result.margins.get("n_clauses", 0), oracle_depth=oracle_depth,
        reached_goal=result.reached_goal, pruned=result.pruned_by_rule,
        llm_rejected=result.llm_rejected, conf=result.confidence, score=result.score,
    )


# ---------------------------------------------------------------- 汇总
def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=list(LABELS),
                          average="macro", zero_division=0))


def per_class_recall(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    from sklearn.metrics import recall_score

    r = recall_score(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return {lab: float(v) for lab, v in zip(LABELS, r)}


def summarize(rows: list[Row]) -> dict:
    if not rows:
        return {}
    yt = [r.gold for r in rows]
    yp = [r.answer for r in rows]
    n = len(rows)
    exp = sum(r.expanded for r in rows)
    red = sum(r.redundant for r in rows)
    out = {
        "n": n,
        "macro_f1": round(macro_f1(yt, yp), 4),
        "accuracy": round(sum(r.correct for r in rows) / n, 4),
        "avg_expanded": round(exp / n, 3),
        "avg_steps": round(sum(r.steps for r in rows) / n, 3),
        "redundant_ratio": round(red / max(1, exp), 4),
        "avg_llm_calls": round(sum(r.llm_calls for r in rows) / n, 3),
        "goal_rate": round(sum(r.reached_goal for r in rows) / n, 4),
        "avg_pruned": round(sum(r.pruned for r in rows) / n, 3),
        "avg_llm_rejected": round(sum(r.llm_rejected for r in rows) / n, 3),
    }
    out.update({f"recall_{k}": round(v, 4) for k, v in per_class_recall(yt, yp).items()})
    out["confusion"] = confusion(yt, yp)
    return out


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    m = {a: {b: 0 for b in LABELS} for a in LABELS}
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def bucket_by(rows: list[Row], key: str) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if key == "hops":
            d = r.oracle_depth
            b = "1-2 跳" if d <= 2 else ("3 跳" if d == 3 else "4+ 跳")
        elif key == "clauses":
            b = f"{min(r.n_clauses, 3)} 条款" if r.n_clauses < 3 else "3+ 条款"
        else:
            b = str(getattr(r, key))
        out[b].append(r)
    return dict(out)


def write_rows(rows: list[Row], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=1),
                 encoding="utf-8")
