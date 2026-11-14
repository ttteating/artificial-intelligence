"""条款图的数据结构（A、B 双方共用的只读契约）。

设计说明
--------
ACE 的每条 scenario 只附带 1~4 条 evidence 条款，且 ACE **不提供文档级结构**，
因此本项目的条款图是 **per-scenario 图**，而不是全局图：

    one scenario -> one ScenarioGraph

为了让"多跳"有实际意义，图的节点不是"条款"，而是条款内部的语义单元（facet）：
主定义/义务句、例外句（provisional clause）、条件句、定义句、交叉引用句、救济句。
边沿用 COMPACT 的六类关系：DEFINES / COMPLEMENTS / EXCEPTIONS / DEPENDS_ON /
CONFLICTS / SEQUENCE，另加 CITES（条款内交叉引用）以对齐文档 3.6 节的表述。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------- 节点角色
ROLE_OBL = "OBL"        # 主句：义务 / 授权 / 主谓陈述
ROLE_EXC = "EXC"        # 例外：except / unless / subject to / notwithstanding
ROLE_COND = "COND"      # 条件：if / provided that / in the event that
ROLE_DEF = "DEF"        # 定义：means / shall mean / as defined
ROLE_SCOPE = "SCOPE"    # 交叉引用：as set out in / pursuant to / in accordance with
ROLE_REMEDY = "REMEDY"  # 救济 / 责任：indemnify / liable / terminate / damages

ROLES = (ROLE_OBL, ROLE_EXC, ROLE_COND, ROLE_DEF, ROLE_SCOPE, ROLE_REMEDY)

# ---------------------------------------------------------------- 边类型
E_DEFINES = "DEFINES"
E_COMPLEMENTS = "COMPLEMENTS"
E_EXCEPTIONS = "EXCEPTIONS"
E_DEPENDS_ON = "DEPENDS_ON"
E_CONFLICTS = "CONFLICTS"
E_SEQUENCE = "SEQUENCE"
E_CITES = "CITES"

EDGE_TYPES = (
    E_DEFINES,
    E_COMPLEMENTS,
    E_EXCEPTIONS,
    E_DEPENDS_ON,
    E_CONFLICTS,
    E_SEQUENCE,
    E_CITES,
)


@dataclass(frozen=True)
class Node:
    nid: str          # "evidence_3#s0"
    clause: str       # "evidence_3"
    role: str         # one of ROLES
    text: str
    sent_idx: int = -1

    @property
    def key(self) -> tuple[str, int]:
        return (self.clause, self.sent_idx)


@dataclass(frozen=True)
class Edge:
    src: str
    etype: str
    dst: str
    w: float = 1.0


@dataclass
class ScenarioGraph:
    """单场景条款图。out / inc 为邻接表。"""

    sid: str
    nodes: dict[str, Node] = field(default_factory=dict)
    out: dict[str, list[Edge]] = field(default_factory=dict)
    inc: dict[str, list[Edge]] = field(default_factory=dict)
    entry: list[str] = field(default_factory=list)          # 每个条款的主句节点
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------- 查询
    def actions(self, nid: str) -> list[Edge]:
        return self.out.get(nid, [])

    def node(self, nid: str) -> Node:
        return self.nodes[nid]

    @property
    def clause_ids(self) -> list[str]:
        return sorted({n.clause for n in self.nodes.values()})

    @property
    def n_clauses(self) -> int:
        return len(self.clause_ids)

    def clause_nodes(self, clause: str) -> list[Node]:
        return sorted(
            (n for n in self.nodes.values() if n.clause == clause),
            key=lambda n: n.sent_idx,
        )

    def entry_of(self, clause: str) -> str | None:
        for n in self.clause_nodes(clause):
            if n.role == ROLE_OBL:
                return n.nid
        ns = self.clause_nodes(clause)
        return ns[0].nid if ns else None

    def stats(self) -> dict:
        ec: dict[str, int] = {}
        for es in self.out.values():
            for e in es:
                ec[e.etype] = ec.get(e.etype, 0) + 1
        rc: dict[str, int] = {}
        for n in self.nodes.values():
            rc[n.role] = rc.get(n.role, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": sum(len(v) for v in self.out.values()),
            "clauses": self.n_clauses,
            "roles": rc,
            "edge_types": ec,
        }

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "nodes": [
                {"nid": n.nid, "clause": n.clause, "role": n.role,
                 "sent_idx": n.sent_idx, "text": n.text}
                for n in self.nodes.values()
            ],
            "edges": [
                {"src": e.src, "etype": e.etype, "dst": e.dst, "w": e.w}
                for es in self.out.values() for e in es
            ],
            "entry": self.entry,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "ScenarioGraph":
        g = ScenarioGraph(sid=d["sid"], entry=list(d.get("entry", [])),
                          meta=dict(d.get("meta", {})))
        for nd in d["nodes"]:
            g.nodes[nd["nid"]] = Node(nd["nid"], nd["clause"], nd["role"],
                                      nd["text"], nd.get("sent_idx", -1))
        for ed in d["edges"]:
            e = Edge(ed["src"], ed["etype"], ed["dst"], ed.get("w", 1.0))
            g.out.setdefault(e.src, []).append(e)
            g.inc.setdefault(e.dst, []).append(e)
        return g
