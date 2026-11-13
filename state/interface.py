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
