"""搜索树节点：供三套内核统一记录，最终由 eval/visualize_tree.py 消费。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    tid: int
    state_key: tuple
    depth: int
    node_id: str | None = None            # 到达该状态时扩展的 facet 节点
    clause: str | None = None
    role: str | None = None
    action_from_parent: dict | None = None
    parent: int | None = None
    children: list[int] = field(default_factory=list)

    g: float = 0.0
    h: float = 0.0
    f: float = 0.0
    visits: int = 0
    value: float = 0.0
    conf: float | None = None
    label_hypothesis: str | None = None

    status: str = "frontier"              # expanded | frontier | pruned | goal | deadend
    flags: list[str] = field(default_factory=list)   # success_path/redundant/llm_misjudge/rule_pruned/llm_rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "tid": self.tid, "parent": self.parent, "children": self.children,
            "depth": self.depth, "node_id": self.node_id, "clause": self.clause,
            "role": self.role, "action": self.action_from_parent,
            "g": round(self.g, 4), "h": round(self.h, 4), "f": round(self.f, 4),
            "visits": self.visits, "value": round(self.value, 4),
            "conf": None if self.conf is None else round(self.conf, 4),
            "label": self.label_hypothesis, "status": self.status, "flags": self.flags,
        }


class TreeRecorder:
    """极简记录器：三套内核共用，避免各自实现一份可视化数据结构。"""

    def __init__(self) -> None:
        self.nodes: list[TreeNode] = []
        self._index: dict[tuple, int] = {}

    def add(self, state_key: tuple, depth: int, *, parent: int | None = None,
            node_id: str | None = None, clause: str | None = None,
            role: str | None = None, action: dict | None = None,
            g: float = 0.0, h: float = 0.0, conf: float | None = None,
            label: str | None = None, status: str = "frontier",
            f: float | None = None, flags: list[str] | None = None) -> int:
        """f 缺省为 g+h；加权 A* 必须显式传入 g+w·h，否则可视化上的 f 与排序用的
        优先级对不上（同一个节点显示 f=3.2 却被排到 f=4.1 的前面）。"""
        tid = len(self.nodes)
        tn = TreeNode(tid=tid, state_key=state_key, depth=depth, node_id=node_id,
                      clause=clause, role=role, action_from_parent=action, parent=parent,
                      g=g, h=h, f=(g + h) if f is None else f, conf=conf,
                      label_hypothesis=label, status=status,
                      flags=list(flags or []))
        self.nodes.append(tn)
        if parent is not None:
            self.nodes[parent].children.append(tid)
        self._index.setdefault((parent, state_key), tid)
        return tid

    def get(self, tid: int) -> TreeNode:
        return self.nodes[tid]

    def mark_success_path(self, tids: list[int]) -> None:
        for t in tids:
            if 0 <= t < len(self.nodes):
                if "success_path" not in self.nodes[t].flags:
                    self.nodes[t].flags.append("success_path")
                self.nodes[t].status = "goal"

    def mark_redundant(self, success_tids: set[int]) -> None:
        """不在成功路径上的已扩展节点记为冗余分支。"""
        for tn in self.nodes:
            if tn.status == "expanded" and tn.tid not in success_tids:
                if "success_path" not in tn.flags and "redundant" not in tn.flags:
                    tn.flags.append("redundant")

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes]}
