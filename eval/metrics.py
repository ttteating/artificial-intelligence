"""指标：macro-F1、各类 recall、平均扩展节点、推理步数、冗余路径占比、LLM 调用。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix as _sklearn_confusion_matrix
from sklearn.metrics import f1_score as _sklearn_f1_score
from sklearn.metrics import recall_score as _sklearn_recall_score


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


# ---------------------------------------------------------------- 单样本 / 整体指标
def compute_metrics(predictions: list[str], golds: list[str]) -> dict:
    """Overall metric abstraction for 3-way compliance classification.

    Definitions
    -----------
    * accuracy: correct predictions / total predictions
    * macro_f1: unweighted mean of per-class F1 (labels are fixed as
                Compliant / Non-Compliant / Not-Applicable)
    * per_class_recall: recall for each of the three fixed labels
    * confusion_matrix: gold(row) x prediction(column)
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must have the same length")

    n = len(predictions)
    correct = sum(1 for p, g in zip(predictions, golds) if p == g)
    accuracy = correct / n if n else 0.0

    macro_f1_val = float(_sklearn_f1_score(golds, predictions, labels=LABELS,
                                            average="macro", zero_division=0))

    if n:
        _rec = _sklearn_recall_score(golds, predictions, labels=LABELS,
                                     average=None, zero_division=0)
        per_class_recall = {lab: float(v) for lab, v in zip(LABELS, _rec)}
    else:
        per_class_recall = {lab: 0.0 for lab in LABELS}

    cm = _sklearn_confusion_matrix(golds, predictions, labels=LABELS)
    confusion_matrix = {
        lab: {other: int(cm[i][j]) for j, other in enumerate(LABELS)}
        for i, lab in enumerate(LABELS)
    }

    return {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1_val), 6),
        "per_class_recall": {k: round(float(v), 6) for k, v in per_class_recall.items()},
        "confusion_matrix": confusion_matrix,
    }


# ---------------------------------------------------------------- 搜索收敛指标
def _is_expanded(trace: dict) -> bool:
    """单条 trace 是否算作“扩展”节点。

    优先读取显式字段 `expanded`；若未提供，则把非 `pruned`、有 `node_id`
    的状态视为扩展节点。这与现有 `SearchResult.trace` 统一。
    """
    if trace.get("expanded") is True:
        return True
    if isinstance(trace.get("expanded"), (int, float)):
        return bool(trace.get("expanded"))
    node_id = trace.get("node_id")
    status = str(trace.get("status", ""))
    if node_id and status != "pruned":
        return True
    return False


def _final_nodes(trace: dict) -> set:
    """最终返回路径上的节点集合。

    优先读取显式字段：
    * `path_nodes` / `final_nodes`
    * 状态 `status in {goal, answer, final, success, terminal}`
    * 显式 `is_final=True`
    若均未提供，则返回空集。调用方单独确认 `final_nodes` 与扩展节点的差异。
    """
    final_nodes = trace.get("final_nodes")
    if isinstance(final_nodes, list):
        return {str(x) for x in final_nodes}

    if trace.get("is_final") is True:
        node_id = trace.get("node_id")
        if node_id is not None:
            return {str(node_id)}
        return set()

    status = str(trace.get("status", ""))
    if status in {"goal", "answer", "final", "success", "terminal"}:
        node_id = trace.get("node_id")
        if node_id is not None:
            return {str(node_id)}
        return set()

    path_nodes = trace.get("path_nodes")
    if isinstance(path_nodes, list):
        return {str(x) for x in path_nodes}
    if path_nodes is not None:
        return {str(path_nodes)}

    return set()


def compute_search_metrics(traces: list[dict]) -> dict:
    """从搜索日志 trace 列表计算搜索收敛指标。

    定义
    ----
    * avg_expanded_nodes: 所有 trace 中“扩展节点数”的平均值；每个 trace 内部
      先按 `node_id` 去重，避免重复统计同一个状态。
    * avg_reasoning_steps: 最终返回路径动作数的平均值。若 trace 提供
      `path`/`path_nodes`，直接取其长度；否则使用显式最终节点且长度为
      `len(final_nodes) - 1`。若完全没有最终路径节点，则该 sample 的步数
      记为 0。
    * redundant_path_ratio: 扩展节点数中，最终正确路径上没有出现的比例，计算
      公式为
          redundant_path_ratio = sum(扩展节点数 - 最终路径节点数) / sum(扩展节点数)
      若总扩展节点数为 0，则返回 0.0。
    * llm_call_count: 所有 trace 中 `llm_calls` 字段之和；若该字段缺失，则按
      `len(traces)` 近似。
    """
    if not traces:
        return {
            "avg_expanded_nodes": 0.0,
            "avg_reasoning_steps": 0.0,
            "redundant_path_ratio": 0.0,
            "llm_call_count": 0,
        }

    total_expanded = 0
    total_steps = 0
    total_redundant = 0
    total_llm_calls = 0

    for trace in traces:
        expanded_nodes = {str(t) for t in trace.get("nodes", []) if _is_expanded(t)}
        # 兼容只在单个字典里表达一条节点的场景
        if not expanded_nodes and isinstance(trace.get("expanded"), bool):
            if trace.get("expanded"):
                expanded_nodes = {str(trace.get("node_id"))} if trace.get("node_id") is not None else set()

        final_nodes = _final_nodes(trace)

        expanded_count = len(expanded_nodes)
        redundant = len(expanded_nodes - final_nodes)
        total_expanded += expanded_count
        total_redundant += redundant

        final_path_actions = trace.get("path") or trace.get("path_nodes")
        if isinstance(final_path_actions, list):
            steps = len(final_path_actions)
        else:
            steps = len(final_nodes) - 1 if final_nodes else 0
        total_steps += max(0, steps)

        trace_llm_calls = trace.get("llm_calls", 0)
        if isinstance(trace_llm_calls, (int, float)):
            total_llm_calls += int(trace_llm_calls)
        elif trace_llm_calls is None:
            total_llm_calls += 1

    avg_expanded = total_expanded / len(traces)
    avg_steps = total_steps / len(traces)
    redundant_ratio = total_redundant / total_expanded if total_expanded else 0.0

    return {
        "avg_expanded_nodes": round(avg_expanded, 4),
        "avg_reasoning_steps": round(avg_steps, 4),
        "redundant_path_ratio": round(redundant_ratio, 6),
        "llm_call_count": int(total_llm_calls),
    }

# ---------------------------------------------------------------- 推理成本
def _default_prices() -> dict[str, dict[str, float]]:
    """模型单价（USD / 1k tokens）。

    这是工程上的兜底估算表。真实网关配置若缺少当前模型，会回退到这里。
    如果希望精确报价，请在 config.yaml 里增加
    `llm.model_prices`，并在下次维护时同步进来。
    """
    return {
        "gpt-4o-mini": {
            "input": 0.150,
            "output": 0.600,
            "cache": 0.075,
        },
        "gpt-4o": {
            "input": 2.500,
            "output": 10.000,
            "cache": 1.250,
        },
        "gpt-4-turbo": {
            "input": 10.000,
            "output": 30.000,
            "cache": 5.000,
        },
        "gpt-3.5-turbo": {
            "input": 0.500,
            "output": 1.500,
            "cache": 0.250,
        },
    }


def _load_llm_prices() -> dict[str, float]:
    """按模型名读取单价（USD / 1k tokens）。

    优先级：
    1. config.yaml 中显式给出的价格；
    2. 内置兜底表 `_default_prices()`。
    """
    base = _default_prices()
    try:
        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
    except Exception:
        return base

    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    model = llm_cfg.get("model", "").strip()
    if not model:
        return base

    prices = {}
    for key in ("input", "output", "cache"):
        price = llm_cfg.get(f"model_{key}_price", llm_cfg.get(f"model_price_{key}"))
        if price is not None:
            prices[key] = float(price)

    if prices:
        return prices

    return base


def _tokens_from_trace(trace: dict) -> int:
    """从单条 trace 提取 token 数。

    优先级：
    `tokens` > `request_tokens` > `completion_tokens` > 默认 150。
    若连默认值都无法使用，则返回 0。
    """
    for key in ("tokens", "request_tokens", "completion_tokens"):
        val = trace.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return 0


def compute_llm_cost(traces: list[dict]) -> dict:
    """推理成本估算。

    定义
    ----
    * total_llm_calls: 所有 trace 中 `llm_calls` 之和；若字段缺失，则按
      `len(traces)` 近似。
    * cache_hit_rate: `cache_hit`/`llm_call` 按 trace 权重加权；若完全缺失，
      默认 0.0。
    * estimated_tokens: 每个 trace 的直接 token 数之和；若没有显式 token 字段，
      按 `total_llm_calls * DEFAULT_TOKENS_PER_CALL` 估算。
    * estimated_cost_usd: 估算总 token 数乘以读取到的模型单价（USD / 1k tokens）。
      若未找到模型价格，默认使用 `gpt-4o-mini` 单价。
    """
    if not traces:
        return {
            "total_llm_calls": 0,
            "cache_hit_rate": 0.0,
            "estimated_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    total_calls = 0
    total_cache_hit = 0
    total_tokens = 0
    default_tokens_per_call = 150

    for trace in traces:
        calls = trace.get("llm_calls", 0)
        if isinstance(calls, (int, float)):
            total_calls += int(calls)
        else:
            total_calls += 1

        cache_hits = trace.get("cache_hit", False)
        if isinstance(cache_hits, bool):
            total_cache_hit += 1 if cache_hits else 0
        elif isinstance(cache_hits, (int, float)) and cache_hits > 0:
            total_cache_hit += 1

        trace_tokens = _tokens_from_trace(trace)
        if trace_tokens > 0:
            total_tokens += trace_tokens

    cache_hit_rate = total_cache_hit / total_calls if total_calls else 0.0

    prices = _load_llm_prices()
    model = ""
    try:
        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
        llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
        model = str(llm_cfg.get("model", "")).strip()
    except Exception:
        pass

    if model and model in prices:
        price_per_1k = prices[model]
    else:
        price_per_1k = prices.get("gpt-4o-mini", {})
        if not isinstance(price_per_1k, dict):
            price_per_1k = prices.get("gpt-4o-mini", {})

    in_price = float(price_per_1k.get("input", 0.0))
    out_price = float(price_per_1k.get("output", 0.0))
    cache_price = float(price_per_1k.get("cache", 0.0))

    # 估算：输入/输出各按 70/30 分配，缓存按 10% 比例抵扣。
    total_tokens = total_tokens or (total_calls * default_tokens_per_call)
    base_in = total_tokens * 0.7
    base_out = total_tokens * 0.3
    cache_ratio = 0.10
    cached_tokens = int(base_in * cache_ratio)

    estimated_cost = (
        (base_in - cached_tokens) * (in_price + out_price) / 1000.0
        + cached_tokens * (cache_price if cache_price else 0.0)
    ) / 1000.0

    return {
        "total_llm_calls": int(total_calls),
        "cache_hit_rate": round(cache_hit_rate, 6),
        "estimated_tokens": int(total_tokens),
        "estimated_cost_usd": round(estimated_cost, 8),
    }



def write_rows(rows: list[Row], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=1),
                 encoding="utf-8")

# ---------------------------------------------------------------- 分层报告
def _normalize_record(predictions: list, golds: list, index: int, strata: list[str]) -> dict:
    """把 predictions/golds/strata 归一化为记录字典。

    支持两种输入形式：
    * 字符串：直接作为标签，对应 `strata[index]` 作为分层键。
    * 字典：期望包含 `prediction`（或 `answer`），`gold`，`strata`（或 `hops` / `trace_metadata.strata`）以及可选的 `expanded`（或 `expanded_nodes`）。
    """
    pred = predictions[index]
    gold = golds[index]
    stratum = strata[index] if index < len(strata) else ""

    if isinstance(pred, dict):
        record = {
            "prediction": pred.get("prediction", pred.get("answer", pred.get("gold"))),
            "gold": pred.get("gold", gold if isinstance(gold, str) else pred.get("gold")),
            "strata": pred.get("strata", pred.get("hops", pred.get("trace_metadata", {}).get("strata", stratum))),
            "expanded": int(pred.get("expanded", pred.get("expanded_nodes", 0))),
        }
    elif isinstance(gold, dict):
        record = {
            "prediction": pred if isinstance(pred, str) else pred.get("prediction", pred.get("answer")),
            "gold": gold.get("gold", gold.get("answer", gold)),
            "strata": gold.get("strata", gold.get("hops", gold.get("trace_metadata", {}).get("strata", stratum))),
            "expanded": int(gold.get("expanded", gold.get("expanded_nodes", 0))),
        }
    else:
        record = {
            "prediction": pred,
            "gold": gold,
            "strata": stratum,
            "expanded": 0,
        }
    return record


def stratified_report(predictions: list, golds: list, strata: list[str]) -> pd.DataFrame:
    """按跳数分层报告 macro-F1 与平均扩展节点。

    定义
    ----
    * 对每个 stratum，取该层所有样本的预测与真值，计算 3 类 macro-F1。
    * 该层的扩展节点取样本 `expanded` / `expanded_nodes` 的算术平均值；
      若无该字段，则记为 0。
    * 返回 DataFrame 列为：strata, n, macro_f1, avg_expanded。
    可直接粘贴到报告表格。
    """
    if len(predictions) != len(golds) or len(golds) != len(strata):
        raise ValueError("predictions, golds and strata must have the same length")

    records = [_normalize_record(predictions, golds, i, strata) for i in range(len(predictions))]

    unique_strata: list[str] = []
    for s in strata:
        s = str(s).strip()
        if s not in unique_strata:
            unique_strata.append(s)

    rows = []
    for stratum in unique_strata:
        sub = [r for r in records if str(r["strata"]).strip() == stratum]
        y_true = [r["gold"] for r in sub]
        y_pred = [r["prediction"] for r in sub]
        expandeds = [r["expanded"] for r in sub]
        macro_f1 = float(_sklearn_f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if sub else float("nan")
        avg_expanded = sum(expandeds) / len(expandeds) if expandeds else 0.0
        rows.append({
            "strata": stratum,
            "n": len(sub),
            "macro_f1": round(macro_f1, 6),
            "avg_expanded": round(avg_expanded, 4),
        })

    return pd.DataFrame(rows, columns=["strata", "n", "macro_f1", "avg_expanded"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    predictions = [
        "Compliant", "Non-Compliant", "Not-Applicable",
        "Compliant", "Non-Compliant", "Not-Applicable",
    ]
    golds = [
        "Compliant", "Non-Compliant", "Not-Applicable",
        "Non-Compliant", "Compliant", "Not-Applicable",
    ]

    metrics = compute_metrics(predictions, golds)
    print("compute_metrics:")
    print(f"  accuracy      : {metrics['accuracy']:.6f}")
    print(f"  macro_f1      : {metrics['macro_f1']:.6f}")
    print(f"  per_class_recall: {metrics['per_class_recall']}")
    print(f"  confusion_matrix:")
    for lab, row in metrics["confusion_matrix"].items():
        print(f"    {lab}: {row}")

    traces = [
        {
            "nodes": [
                {"node_id": "n1", "expanded": True},
                {"node_id": "n2", "expanded": True},
                {"node_id": "n3", "expanded": True},
            ],
            "final_nodes": ["n1", "n2"],
            "path": ["action1", "action2"],
            "llm_calls": 3,
        },
        {
            "nodes": [
                {"node_id": "a1", "expanded": True},
                {"node_id": "a2", "expanded": True},
            ],
            "final_nodes": ["a1"],
            "path": ["action1"],
            "llm_calls": 2,
        },
        {
            "nodes": [],
            "final_nodes": [],
            "path": [],
            "llm_calls": 1,
        },
    ]
    search_metrics = compute_search_metrics(traces)
    print("compute_search_metrics:")
    for k, v in search_metrics.items():
        print(f"  {k}: {v}")

    cost_metrics = compute_llm_cost(traces)
    print("compute_llm_cost:")
    for k, v in cost_metrics.items():
        print(f"  {k}: {v}")

    strata_map = ["2 跳", "3 跳", "4 跳", "2 跳", "3 跳", "4 跳"]
    report = stratified_report(predictions, golds, strata_map)
    print("stratified_report:")
    print(report.to_string(index=False))

