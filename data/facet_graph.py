"""facet 级条款图构建器（A 侧补齐，搜索内核真正消费的图）。

分工说明
--------
`data/clause_graph_builder.py`（B 线）产出的是**条款级** `nx.DiGraph`，只用于
条款交叉引用统计；而 `state/search/heuristic` 全部跑在 `facet`（条款内语义单元）
上——节点带 role、边带 etype、状态在 facet 上累积，多跳才有意义。

`data/corpus.py` 需要的四个契约由本模块提供：

    GRAPH_VERSION
    build_graph(sid, clauses, clause_sim) -> ScenarioGraph
    precompute_clause_sim(scorer, scenario_text, clauses) -> dict[clause, sim]
    precompute_facet_sim(scorer, scenario_text, graph) -> dict[nid, sim]

建图规则（可复现，写进实验报告 §3.2）
------------------------------------
1. 每个条款按 `data.features.split_sentences` 切成 1~5 个 facet，`detect_role`
   判角色（OBL / EXC / COND / DEF / SCOPE / REMEDY）。
2. **条款内**：主句（entry）单向连向本条其余 facet，边类型由目标角色决定
   （EXC→EXCEPTIONS、COND→DEPENDS_ON、DEF→DEFINES、SCOPE→CITES、
   REMEDY/OBL→COMPLEMENTS）。
3. **条款间**：主句 → 主句，边类型按目标条款的首句线索决定
   （例外引导词→EXCEPTIONS、交叉引用→CITES、时间/顺序线索→SEQUENCE、
   条件引导词→DEPENDS_ON、其余→COMPLEMENTS）。
4. **冲突边**：只有在"两条款讲同一件事"的前提下才连——先剔除合同模板高频词，
   再要求共享 ≥2 个**实词**（或 1 个实词 + 显式 `notwithstanding / conflict with`
   标记），且两条款道义极性相反（一个许可、一个禁止）。
5. 边权 `w` 一律取目标条款与场景的相关度（`clause_sim`），**CONFLICTS 不再写死 1.0**
   （facet-v2 修正：旧版把冲突边固定成最低代价，等价于让搜索优先走冲突边）。
"""

from __future__ import annotations

import re

from data.clause_graph import (
    E_CITES, E_COMPLEMENTS, E_CONFLICTS, E_DEFINES, E_DEPENDS_ON, E_EXCEPTIONS,
    E_SEQUENCE, Edge, Node, ROLE_COND, ROLE_DEF, ROLE_EXC, ROLE_OBL, ROLE_REMEDY,
    ROLE_SCOPE, ScenarioGraph,
)
from data.features import (
    PERMISSIVE_CUES, PROHIBITIVE_CUES, TEMPORAL_CUES, XREF_CUES, detect_role,
    split_sentences,
)

GRAPH_VERSION = "facet-v2"

_EXC_LEAD = re.compile(
    r"^\s*[\(\[]?(except|unless|provided that|subject to|notwithstanding|other than)",
    re.IGNORECASE,
)
_COND_LEAD = re.compile(
    r"^\s*[\(\[]?(if|when|where|in the event|in case|so long as|as long as)",
    re.IGNORECASE,
)
_CONFLICT_HINT = re.compile(
    r"\b(notwithstanding|conflict(?:s|ing)? with|inconsistent with|supersede)", re.IGNORECASE
)
_SECTION_REF = re.compile(r"\b(?:section|clause|article)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
# 中文合同的条号：第十九条 / 第 9 条。只对含中文字符的片段生效，不影响英文语料。
_SECTION_REF_CJK = re.compile(r"第\s*(\d+(?:\.\d+)?)\s*条")
_CJK_XREF = re.compile(r"(依据|根据|按照|依照|本协议第|本合同第)")
_CJK_TEMPORAL = re.compile(r"(之后|以前|届满|期满|终止后|履行完毕)")
_EXC_LEAD_CJK = re.compile(r"^\s*(除非|除|尽管|但书)")
_COND_LEAD_CJK = re.compile(r"^\s*(如果|若|倘若|一旦|在.{0,20}情况下)")
_WORD = re.compile(r"[a-z]{5,}")

_ROLE_EDGE = {
    ROLE_EXC: E_EXCEPTIONS,
    ROLE_COND: E_DEPENDS_ON,
    ROLE_DEF: E_DEFINES,
    ROLE_SCOPE: E_CITES,
    ROLE_REMEDY: E_COMPLEMENTS,
    ROLE_OBL: E_COMPLEMENTS,
}


def _add(g: ScenarioGraph, src: str, etype: str, dst: str, w: float = 1.0) -> None:
    """加边并维护 out/inc 邻接表；同 (src,etype,dst) 不重复。"""
    if src == dst or src not in g.nodes or dst not in g.nodes:
        return
    for e in g.out.get(src, ()):
        if e.etype == etype and e.dst == dst:
            return
    e = Edge(src, etype, dst, round(float(w), 4))
    g.out.setdefault(src, []).append(e)
    g.inc.setdefault(dst, []).append(e)


def _content_words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


# 合同模板高频词：它们在任意两条条款里都会共现，用作"是否在讲同一件事"的证据
# 会把"保险单形式"与"分许可"这种无关条款连成冲突边（旧版抽查第 9、18 条的假阳性）。
_BOILERPLATE = frozenset({
    "agreement", "parties", "party", "shall", "hereunder", "hereof", "herein",
    "thereof", "therein", "thereunder", "section", "clause", "article", "provision",
    "provisions", "terms", "conditions", "termination", "terminate", "effective",
    "notice", "written", "applicable", "rights", "obligations", "liability",
    "damages", "breach", "claims", "period", "years", "provided", "hereafter",
    "notwithstanding", "conflict", "conflicts", "conflicting", "inconsistent",
})


def _distinctive_words(text: str) -> set[str]:
    """去掉模板高频词后的实词集合（用于判定两条款是否真在讲同一件事）。"""
    return _content_words(text) - _BOILERPLATE


def _clash(a: str, b: str) -> bool:
    """两条款是否存在道义冲突（用于 CONFLICTS 边）。

    facet-v2 收紧口径：旧版只要"极性相反 + 共享 ≥2 个长度≥5 的词"，而
    agreement / parties / shall 这类模板词在任何两份合同里都共现，于是无关条款
    也被连成 CONFLICTS（抽查第 9、18 条）。现在要求共享的是**剔除模板词后的
    实词**，显式冲突标记也至少要有 1 个实词支撑。
    """
    per_a, pro_a = bool(PERMISSIVE_CUES.search(a)), bool(PROHIBITIVE_CUES.search(a))
    per_b, pro_b = bool(PERMISSIVE_CUES.search(b)), bool(PROHIBITIVE_CUES.search(b))
    if not ((per_a and pro_b) or (pro_a and per_b)):
        return False                       # 极性不相反：不构成道义冲突

    shared = _distinctive_words(a) & _distinctive_words(b)
    if len(shared) >= 2:
        return True
    hint = _CONFLICT_HINT.search(a) or _CONFLICT_HINT.search(b)
    return bool(hint and shared)            # 标记词 + 至少 1 个共同实词



def _inter_type(text: str) -> str:
    """条款间主句连边的类型判定（看目标条款的线索词）。"""
    head = split_sentences(text, max_parts=1)[0] if text else ""
    if _EXC_LEAD.search(head) or _EXC_LEAD.search(text[:120]) or _EXC_LEAD_CJK.search(head):
        return E_EXCEPTIONS
    if XREF_CUES.search(head) or _CJK_XREF.search(head):
        return E_CITES
    if TEMPORAL_CUES.search(head) or _CJK_TEMPORAL.search(head):
        return E_SEQUENCE
    if _COND_LEAD.search(head) or _COND_LEAD_CJK.search(head):
        return E_DEPENDS_ON
    return E_COMPLEMENTS


def build_graph(sid: str, clauses: dict, clause_sim: dict | None = None) -> ScenarioGraph:
    """把一个场景的 clauses（{evidence_x: text}）建成 facet 级 ScenarioGraph。"""
    sim = {k: float(v) for k, v in (clause_sim or {}).items()}
    g = ScenarioGraph(sid=sid, meta={"version": GRAPH_VERSION, "clause_sim": dict(sim)})

    # ---------------- 1) facet 节点
    roles_by_clause: dict[str, list[str]] = {}
    for cid, raw in clauses.items():
        sents = [s for s in split_sentences(raw) if s.strip()] or [str(raw)]
        roles = [detect_role(s, i == 0) for i, s in enumerate(sents)]
        roles_by_clause[cid] = roles
        for i, (s, r) in enumerate(zip(sents, roles)):
            nid = f"{cid}#s{i}"
            g.nodes[nid] = Node(nid=nid, clause=cid, role=r, text=s, sent_idx=i)
        entry = next((f"{cid}#s{i}" for i, r in enumerate(roles) if r == ROLE_OBL),
                     f"{cid}#s0")
        g.entry.append(entry)

    # ---------------- 2) 条款内：主句 -> 其余 facet
    for cid in clauses:
        entry = g.entry_of(cid)
        w = sim.get(cid, 0.5)
        for n in g.clause_nodes(cid):
            if n.nid == entry:
                continue
            _add(g, entry, _ROLE_EDGE.get(n.role, E_COMPLEMENTS), n.nid, w)

    # ---------------- 3) 条款间：主句 -> 主句
    cids = list(clauses)
    for a in cids:
        for b in cids:
            if a == b:
                continue
            _add(g, g.entry_of(a), _inter_type(clauses[b]), g.entry_of(b), sim.get(b, 0.5))
            if _clash(clauses[a], clauses[b]):
                # facet-v2：冲突边权与其余条款间边同口径（取目标条款相关度），
                # 不再写死 1.0——旧写法等于给冲突边固定最低代价，搜索会优先走它。
                _add(g, g.entry_of(a), E_CONFLICTS, g.entry_of(b), sim.get(b, 0.5))

    # ---------------- 4) 显式交叉引用："as set out in Section 9" -> CITES
    for cid, raw in clauses.items():
        for n in g.clause_nodes(cid):
            if n.role not in (ROLE_SCOPE, ROLE_COND, ROLE_DEF):
                continue
            for num in list(_SECTION_REF.findall(n.text)) + list(_SECTION_REF_CJK.findall(n.text)):
                target = f"evidence_{num.split('.')[0]}"
                if target in clauses and target != cid:
                    _add(g, n.nid, E_CITES, g.entry_of(target), sim.get(target, 0.5))

    g.meta["roles"] = {cid: list(r) for cid, r in roles_by_clause.items()}
    g.meta["stats"] = g.stats()
    return g


def precompute_clause_sim(scorer, scenario_text: str, clauses: dict) -> dict[str, float]:
    """条款级相似度：场景文本 vs 每条条款全文（TF-IDF 余弦）。"""
    return {cid: round(float(scorer.cosine(scenario_text, text)), 4)
            for cid, text in clauses.items()}


def precompute_facet_sim(scorer, scenario_text: str, graph: ScenarioGraph) -> dict[str, float]:
    """facet 级相似度：场景文本 vs 每个 facet 文本。

    这一层是"多跳有意义"的关键：相似度拆到 facet 后，只有走到例外
    （EXC）facet，`sim_exc` 才会抬起来，判断才可能翻转。
    """
    return {nid: round(float(scorer.cosine(scenario_text, n.text)), 4)
            for nid, n in graph.nodes.items()}
