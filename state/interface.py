"""已废弃（deprecated）：Day1 冻结接口的历史快照，**不要 import**。

正式契约在 `state/core.py`。本文件的同名类并不等价：

* `State` 少了 `nodes`（facet 序列，h 与决策头都依赖它）和 `g`（累计代价）；
* `SearchResult` 少了 expanded/trace 之外的全部指标字段；
* 没有 `State.key()`，closed set 无法按"条款集合 + 边类型多重集"去重。

留档只用于对照 Day1 冻结版与最终版，两套混用会直接让搜索内核读不到状态。
"""

from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class State:
    clauses: frozenset[str]
    used_edges: tuple[tuple[str, str, str], ...]
    depth: int
    label_hypothesis: str

@dataclass(frozen=True)
class Action:
    src: str
    edge_type: str
    dst: str

@dataclass
class SearchResult:
    answer: str
    path: list[Action]
    expanded_nodes: int
    tree_root: Any
    trace: list[dict]

class Evaluator(Protocol):
    def action_valid(self, state: State, action: Action, question: str) -> tuple[bool, float]: ...
    def path_reward(self, path: list[Action], question: str) -> float: ...
    def success_prob(self, state: State, question: str) -> float: ...
