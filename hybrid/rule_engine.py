import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

import networkx as nx

from state.interface import Action, State

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_DEPTH = 4
TERMINATE_EDGE = "__terminate__"
HARD_RISK = 1.0

# Soft-rule risk deltas (per spec)
R5_EXCEPTION_FIRST_RISK = 0.8
R6_JOINT_REVIEW_RISK = 0.6
R7_DIRECTION_CONFLICT_RISK = 0.3

# Question keyword -> edge types that semantically match the question's focus.
# Used by R7 to detect direction conflicts.
QUESTION_EDGE_HINTS: List[tuple] = [
    (("exception", "例外", "notwithstanding"), {"exception_of"}),
    (("condition", "provided that", "条件", "contingent"), {"conditions_on"}),
    (("defin", "定义", "shall mean", "meaning", "referred to as"), {"defines"}),
    (("supersed", "取代", "prevail", "in lieu of"), {"supersedes"}),
]

# Keywords identifying an indemnification clause (for R6 joint review).
INDEMNIFICATION_KEYWORDS = ("indemnif", "赔偿", "补偿")


class RuleResult(TypedDict):
    allowed: bool
    risk_delta: float
    rule_name: str
    reason: str


RuleFn = Callable[[State, Action, Dict[str, Any]], RuleResult]


def _ok(rule_name: str) -> RuleResult:
    """Returns a default pass result for the given rule."""
    return {"allowed": True, "risk_delta": 0.0, "rule_name": rule_name, "reason": ""}


def r1_superseded_exclusion(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """
    R1 (hard prune): the destination clause has already been superseded
    (appears as the target of a supersedes edge on the used path).
    """
    res = _ok("R1_superseded_exclusion")
    for _src, edge_type, dst in state.used_edges:
        if edge_type == "supersedes" and dst == action.dst:
            res["allowed"] = False
            res["risk_delta"] = HARD_RISK
            res["reason"] = f"目标条款 {action.dst} 已被路径上的 supersedes 边失效，禁止扩展。"
            return res
    return res


def r2_cycle_detection(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """
    R2 (hard prune): the destination clause is already on the current path
    (revisiting a clause would form a cycle). same_section revisits are exempt.
    """
    res = _ok("R2_cycle_detection")
    if action.edge_type == "same_section":
        return res
    if action.dst in state.clauses:
        res["allowed"] = False
        res["risk_delta"] = HARD_RISK
        res["reason"] = f"目标条款 {action.dst} 已在当前路径中，非 same_section 边会构成环。"
    return res


def r3_duplicate_edge(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """R3 (hard prune): the exact edge has already been used."""
    res = _ok("R3_duplicate_edge")
    if (action.src, action.edge_type, action.dst) in state.used_edges:
        res["allowed"] = False
        res["risk_delta"] = HARD_RISK
        res["reason"] = f"边 ({action.src}, {action.edge_type}, {action.dst}) 已在 used_edges 中。"
    return res


def r4_depth_budget(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """
    R4 (hard prune): depth budget exhausted. Termination is not an expansion
    and stays allowed so the search can always close a path.
    """
    res = _ok("R4_depth_budget")
    if action.edge_type == TERMINATE_EDGE:
        return res
    max_depth = int(context.get("max_depth", DEFAULT_MAX_DEPTH))
    if state.depth >= max_depth:
        res["allowed"] = False
        res["risk_delta"] = HARD_RISK
        res["reason"] = f"当前深度 {state.depth} 已达上限 {max_depth}，禁止继续扩展。"
    return res


def r5_exception_first(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """
    R5 (soft): an exception has been confirmed but exception clauses are not
    fully reviewed yet; terminating now is strongly penalized (not pruned).
    """
    res = _ok("R5_exception_first")
    if action.edge_type != TERMINATE_EDGE:
        return res
    if "exception_confirmed" not in state.label_hypothesis:
        return res
    exception_targets = {dst for _s, et, dst in state.used_edges if et == "exception_of"}
    unreviewed = exception_targets - set(state.clauses)
    if unreviewed:
        res["risk_delta"] = R5_EXCEPTION_FIRST_RISK
        res["reason"] = (
            f"例外已确认，但例外条款 {sorted(unreviewed)} 尚未审查完毕，提前终止高风险。"
        )
    return res


def r6_joint_review(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """
    R6 (soft): liability/limitation scenarios require joint review of
    indemnification clauses reachable via cites/same_section; terminating
    while such a clause is unreviewed carries high risk (not pruned).
    """
    res = _ok("R6_joint_review")
    if action.edge_type != TERMINATE_EDGE:
        return res
    question = str(context.get("question", "")).lower()
    if "liability" not in question and "limitation" not in question:
        return res
    graph: Optional[nx.DiGraph] = context.get("graph")
    if graph is None:
        return res
    lookup: Dict[str, str] = context.get("clause_text_lookup") or {}
    indemnity = {
        n for n in graph.nodes
        if any(k in lookup.get(n, "").lower() for k in INDEMNIFICATION_KEYWORDS)
    }
    if not indemnity:
        return res
    connected = set()
    for cid in state.clauses:
        if not graph.has_node(cid):
            continue
        for nbr in graph.successors(cid):
            if graph.edges[cid, nbr].get("type") in ("cites", "same_section"):
                connected.add(nbr)
        for pred in graph.predecessors(cid):
            if graph.edges[pred, cid].get("type") in ("cites", "same_section"):
                connected.add(pred)
    unreviewed = (connected & indemnity) - set(state.clauses)
    if unreviewed:
        res["risk_delta"] = R6_JOINT_REVIEW_RISK
        res["reason"] = (
            f"场景涉及责任限制，赔偿条款 {sorted(unreviewed)} 经 cites/same_section "
            f"可达且尚未审查，终止高风险。"
        )
    return res


def r7_direction_consistency(state: State, action: Action, context: Dict[str, Any]) -> RuleResult:
    """
    R7 (soft): the edge type clearly conflicts with the question's semantic
    focus (e.g. the question asks about an exception but the action is a
    defines edge).
    """
    res = _ok("R7_direction_consistency")
    if action.edge_type in ("same_section", TERMINATE_EDGE):
        return res
    question = str(context.get("question", "")).lower()
    for keywords, expected_edges in QUESTION_EDGE_HINTS:
        if any(k in question for k in keywords):
            if action.edge_type not in expected_edges:
                res["risk_delta"] = R7_DIRECTION_CONFLICT_RISK
                res["reason"] = (
                    f"问题关注 '{keywords[0]}' 类关系，但动作使用 {action.edge_type} 边，"
                    f"语义方向不一致。"
                )
            return res
    return res


# Hard-pruning rules in priority order (R1-R4). allowed=False is absolute.
HARD_RULES: List[RuleFn] = [r1_superseded_exclusion, r2_cycle_detection, r3_duplicate_edge, r4_depth_budget]
# Soft constraints (R5-R7): they only add risk, never set allowed=False.
SOFT_RULES: List[RuleFn] = [r5_exception_first, r6_joint_review, r7_direction_consistency]
DEFAULT_RULES: List[RuleFn] = HARD_RULES + SOFT_RULES


class RuleEngine:
    """
    Pure-Python rule engine over State/Action. No LLM involved.

    - mask() applies hard-pruning rules (R1-R4) and returns the filtered actions.
    - risk() returns the weighted sum of soft-rule risk deltas in [0, 1];
      if any hard rule fires, risk is 1.0 (absolute priority, not weighted).
    - explain() returns all triggered rules for failure analysis.
    """

    def __init__(
        self,
        rules: Optional[List[RuleFn]] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        context: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.rules: List[RuleFn] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self.hard_rules: List[RuleFn] = [r for r in self.rules if r in HARD_RULES]
        self.soft_rules: List[RuleFn] = [r for r in self.rules if r in SOFT_RULES]
        self.max_depth = max_depth
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.weights: Dict[str, float] = dict(weights) if weights else {}

    def set_context(self, context: Dict[str, Any]) -> None:
        """Replaces the engine context (question, graph, clause_text_lookup, ...)."""
        self.context = dict(context)

    def update_context(self, **kwargs: Any) -> None:
        """Merges key/value pairs into the engine context."""
        self.context.update(kwargs)

    def _context(self) -> Dict[str, Any]:
        """Builds the per-call context: engine defaults overridden by user context."""
        ctx: Dict[str, Any] = {"max_depth": self.max_depth}
        ctx.update(self.context)
        return ctx

    def _apply_rule(self, rule: RuleFn, state: State, action: Action) -> RuleResult:
        try:
            return rule(state, action, self._context())
        except Exception as e:  # a broken rule must never crash the search
            logger.exception(f"Rule {getattr(rule, '__name__', rule)} raised: {e}")
            return _ok(getattr(rule, "__name__", "unknown_rule"))

    def mask(self, state: State, actions: List[Action]) -> List[Action]:
        """Filters out any action forbidden by a hard-pruning rule (R1-R4)."""
        kept: List[Action] = []
        for action in actions:
            allowed = True
            for rule in self.hard_rules:
                res = self._apply_rule(rule, state, action)
                if not res["allowed"]:
                    logger.debug(f"mask pruned {action} via {res['rule_name']}: {res['reason']}")
                    allowed = False
                    break
            if allowed:
                kept.append(action)
        return kept

    def risk(self, state: State, action: Action) -> float:
        """
        Returns the normalized risk of taking `action` in [0, 1].
        Hard rules dominate: if one fires, risk is 1.0. Otherwise the
        weighted sum of soft-rule risk deltas, clipped to [0, 1].
        """
        for rule in self.hard_rules:
            res = self._apply_rule(rule, state, action)
            if not res["allowed"]:
                return 1.0

        total = 0.0
        for rule in self.soft_rules:
            res = self._apply_rule(rule, state, action)
            total += res["risk_delta"] * self.weights.get(res["rule_name"], 1.0)
        return min(1.0, max(0.0, total))

    def explain(self, state: State, action: Action) -> List[Dict[str, Any]]:
        """Returns all triggered rules (hard or soft) for failure analysis."""
        triggered: List[Dict[str, Any]] = []
        for rule in self.rules:
            res = self._apply_rule(rule, state, action)
            if (not res["allowed"]) or res["risk_delta"] > 0:
                triggered.append(dict(res))
        return triggered


if __name__ == "__main__":
    # --- Shared fixtures -----------------------------------------------------
    clauses = {
        "evidence_1": "The Recipient shall keep the Information confidential.",
        "evidence_2": "Information means any data disclosed under this Agreement.",
        "evidence_3": "Notwithstanding the foregoing, disclosure to employees with need to know is permitted.",
        "evidence_4": "This clause supersedes Section 2.",
        "evidence_6": "The Recipient shall indemnify the Discloser against all losses.",  # 赔偿条款
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(clauses.keys())
    graph.add_edge("evidence_1", "evidence_6", type="cites")
    graph.add_edge("evidence_1", "evidence_2", type="same_section")

    engine = RuleEngine(max_depth=4, context={
        "question": "Does the limitation of liability affect the confidentiality exception?",
        "graph": graph,
        "clause_text_lookup": clauses,
    })
    # Neutral-question engine for cases that must not pick up R6/R7 signals.
    engine_n = RuleEngine(max_depth=4, context={
        "question": "Is the Recipient allowed to share the data with a junior assistant?",
    })

    def make_state(clauses_set, used_edges, depth, hypothesis=""):
        return State(
            clauses=frozenset(clauses_set),
            used_edges=tuple(used_edges),
            depth=depth,
            label_hypothesis=hypothesis,
        )

    def check(name, condition):
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
        return condition

    all_pass = True
    print("\n=== RuleEngine Smoke Test ===")

    # --- Case 1: R1 superseded exclusion ------------------------------------
    print("\nCase 1: R1 失效条款排除")
    st_r1 = make_state(
        {"evidence_4"}, [("evidence_4", "supersedes", "evidence_2")], 1,
    )
    act_r1_hit = Action(src="evidence_4", edge_type="cites", dst="evidence_2")
    act_r1_ok = Action(src="evidence_4", edge_type="cites", dst="evidence_3")
    masked = engine_n.mask(st_r1, [act_r1_hit, act_r1_ok])
    all_pass &= check("R1触发: 失效目标被剪枝", act_r1_hit not in masked)
    all_pass &= check("R1未触发: 其他目标保留", act_r1_ok in masked)
    all_pass &= check("R1触发: risk == 1.0", engine_n.risk(st_r1, act_r1_hit) == 1.0)
    all_pass &= check("R1未触发: risk == 0.0", engine_n.risk(st_r1, act_r1_ok) == 0.0)

    # --- Case 2: R2 cycle detection ------------------------------------------
    print("\nCase 2: R2 环检测")
    st_r2 = make_state({"evidence_1", "evidence_2"}, [], 1)
    act_r2_hit = Action(src="evidence_1", edge_type="cites", dst="evidence_2")
    act_r2_ok = Action(src="evidence_1", edge_type="same_section", dst="evidence_2")
    masked = engine.mask(st_r2, [act_r2_hit, act_r2_ok])
    all_pass &= check("R2触发: 非same_section回边被剪枝", act_r2_hit not in masked)
    all_pass &= check("R2豁免: same_section保留", act_r2_ok in masked)

    # --- Case 3: R3 duplicate edge -------------------------------------------
    print("\nCase 3: R3 重复边")
    st_r3 = make_state({"evidence_1"}, [("evidence_1", "defines", "evidence_2")], 1)
    act_r3_hit = Action(src="evidence_1", edge_type="defines", dst="evidence_2")
    act_r3_ok = Action(src="evidence_1", edge_type="cites", dst="evidence_2")
    masked = engine.mask(st_r3, [act_r3_hit, act_r3_ok])
    all_pass &= check("R3触发: 重复边被剪枝", act_r3_hit not in masked)
    all_pass &= check("R3未触发: 不同edge_type保留", act_r3_ok in masked)

    # --- Case 4: R4 depth budget ----------------------------------------------
    print("\nCase 4: R4 跳数预算")
    st_r4 = make_state({"evidence_1"}, [], 4)  # depth == max_depth
    act_r4_expand = Action(src="evidence_1", edge_type="defines", dst="evidence_2")
    act_r4_term = Action(src="evidence_1", edge_type=TERMINATE_EDGE, dst="")
    masked = engine.mask(st_r4, [act_r4_expand, act_r4_term])
    all_pass &= check("R4触发: 超限扩展被剪枝", act_r4_expand not in masked)
    all_pass &= check("R4豁免: 终止动作保留", act_r4_term in masked)

    # --- Case 5: R5 exception first -------------------------------------------
    print("\nCase 5: R5 例外优先（软约束）")
    st_r5 = make_state(
        {"evidence_1"}, [("evidence_1", "exception_of", "evidence_3")], 2,
        hypothesis="exception_confirmed",
    )
    act_term = Action(src="evidence_1", edge_type=TERMINATE_EDGE, dst="")
    masked = engine_n.mask(st_r5, [act_term])
    r5_risk = engine_n.risk(st_r5, act_term)
    all_pass &= check("R5触发: 终止动作仍在mask中", act_term in masked)
    all_pass &= check(f"R5触发: risk == {R5_EXCEPTION_FIRST_RISK}", r5_risk == R5_EXCEPTION_FIRST_RISK)
    st_r5_neg = make_state({"evidence_1", "evidence_3"}, [], 2, hypothesis="")
    all_pass &= check("R5未触发: risk == 0.0", engine_n.risk(st_r5_neg, act_term) == 0.0)

    # --- Case 6: R6 joint review ----------------------------------------------
    print("\nCase 6: R6 联合审查（软约束）")
    st_r6 = make_state({"evidence_1"}, [], 2)  # evidence_6 (赔偿) 可达但未审查
    r6_risk = engine.risk(st_r6, act_term)
    all_pass &= check("R6触发: 终止动作仍在mask中", act_term in engine.mask(st_r6, [act_term]))
    all_pass &= check(f"R6触发: risk == {R6_JOINT_REVIEW_RISK}", r6_risk == R6_JOINT_REVIEW_RISK)
    st_r6_neg = make_state({"evidence_1", "evidence_6"}, [], 2)  # 赔偿条款已审查
    all_pass &= check("R6未触发: risk == 0.0", engine.risk(st_r6_neg, act_term) == 0.0)

    # --- Case 7: R7 direction consistency -------------------------------------
    print("\nCase 7: R7 关系方向一致性（软约束）")
    st_r7 = make_state({"evidence_1"}, [], 1)
    act_r7_hit = Action(src="evidence_1", edge_type="defines", dst="evidence_2")  # 问题问"例外"
    r7_risk = engine.risk(st_r7, act_r7_hit)
    all_pass &= check("R7触发: 动作仍在mask中", act_r7_hit in engine.mask(st_r7, [act_r7_hit]))
    all_pass &= check(f"R7触发: risk == {R7_DIRECTION_CONFLICT_RISK}", r7_risk == R7_DIRECTION_CONFLICT_RISK)
    act_r7_ok = Action(src="evidence_1", edge_type="exception_of", dst="evidence_3")  # 方向一致
    all_pass &= check("R7未触发: 方向一致risk == 0.0", engine.risk(st_r7, act_r7_ok) == 0.0)

    # --- Case 8: explain + combined risk --------------------------------------
    print("\nCase 8: explain 与风险叠加")
    st_combo = make_state(
        {"evidence_1"}, [("evidence_1", "exception_of", "evidence_3")], 2,
        hypothesis="exception_confirmed",
    )
    combo_risk = engine.risk(st_combo, act_term)
    expected_combo = min(1.0, R5_EXCEPTION_FIRST_RISK + R6_JOINT_REVIEW_RISK)
    all_pass &= check(
        f"R5+R6叠加归一化: risk == {expected_combo}",
        combo_risk == expected_combo,
    )
    explanations = engine.explain(st_combo, act_term)
    for e in explanations:
        print(f"    - {e['rule_name']}: risk_delta={e['risk_delta']}, {e['reason']}")
    all_pass &= check("explain返回触发的规则", len(explanations) == 2)

    print(f"\n{'All checks passed.' if all_pass else 'SOME CHECKS FAILED.'}")
