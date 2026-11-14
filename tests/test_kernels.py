"""三套内核的单元测试（文档 Day1 要求：3~5 节点手工图 + A*(h=0) 与 BFS 一致）。

不依赖 pytest，直接 `python -m tests.test_kernels` 也能跑。
"""

from __future__ import annotations

from data.clause_graph import (
    E_COMPLEMENTS, E_DEPENDS_ON, E_EXCEPTIONS, ROLE_EXC, ROLE_OBL,
    Edge, Node, ScenarioGraph,
)
from search.astar import astar
from search.bfs import bfs
from search.mcts import mcts
from state.core import Action, ProblemConfig, SearchProblem, State


def toy_graph() -> ScenarioGraph:
    """4 节点手工图：

        ROOT -> A#s0(义务) --EXCEPTIONS--> A#s1(例外) --DEPENDS_ON--> B#s0
                       \\--COMPLEMENTS--> B#s0
    """
    g = ScenarioGraph(sid="toy")
    g.nodes["A#s0"] = Node("A#s0", "A", ROLE_OBL, "Licensee shall pay the fee.", 0)
    g.nodes["A#s1"] = Node("A#s1", "A", ROLE_EXC,
                           "except where the fee is waived under Section 5.", 1)
    g.nodes["B#s0"] = Node("B#s0", "B", ROLE_OBL, "Section 5 waives the fee for trials.", 0)
    g.entry = ["A#s0", "B#s0"]
    for e in [
        Edge("A#s0", E_EXCEPTIONS, "A#s1"),
        Edge("A#s1", E_DEPENDS_ON, "B#s0"),
        Edge("A#s0", E_COMPLEMENTS, "B#s0"),
        Edge("B#s0", E_COMPLEMENTS, "A#s0"),
    ]:
        g.out.setdefault(e.src, []).append(e)
        g.inc.setdefault(e.dst, []).append(e)
    return g


def make_problem(tau=0.9, rules=None, h_fn=None):
    """确定性的假终局判据：走到例外节点才判 Compliant，否则 Non-Compliant。"""
    g = toy_graph()

    def decision(st: State):
        roles = {g.node(n).role for n in st.nodes}
        if ROLE_EXC in roles:
            return "Compliant", 1.0
        if st.depth == 0:
            return "undetermined", 0.0
        return "Non-Compliant", 0.55

    return SearchProblem(g, "the fee was waived under the trial exception",
                         {"A": 0.5, "B": 0.05}, decision=decision,
                         h_fn=h_fn or (lambda s, p: 0.0), rules=rules,
                         cfg=ProblemConfig(max_depth=3, max_expansions=20, tau_conf=tau))


def test_bfs_finds_exception() -> None:
    p = make_problem()
    r = bfs(p)
    assert r.answer == "Compliant", r.answer
    assert r.reached_goal and r.steps >= 2, (r.reached_goal, r.steps)


def test_astar_zero_h_matches_bfs() -> None:
    p = make_problem()
    rb, ra = bfs(p), astar(make_problem(), w_h=1.5)
    assert rb.answer == ra.answer
    assert rb.steps == ra.steps
    assert rb.expanded_nodes == ra.expanded_nodes, (rb.expanded_nodes, ra.expanded_nodes)


def test_closed_set_uses_edge_types() -> None:
    """同一条款集合、不同边类型多重集，必须是两个不同状态。"""
    a = State(frozenset({"A"}), (("A#s0", E_EXCEPTIONS, "A#s1"),), 1)
    b = State(frozenset({"A"}), (("A#s0", E_COMPLEMENTS, "A#s1"),), 1)
    assert a.key() != b.key()


def test_mcts_runs_and_returns_path() -> None:
    r = mcts(make_problem(), n_simulations=40, seed=1)
    assert r.answer == "Compliant"
    assert r.expanded_nodes >= 2
    assert len(r.tree_root.nodes) >= 2


def test_budget_respected() -> None:
    p = make_problem(tau=0.99 + 1e9)   # 永远达不到目标 -> 走满预算
    r = bfs(p, budget=3)
    assert r.expanded_nodes <= 3


def _replay(problem: SearchProblem, result) -> State:
    """把返回的 path 从初始状态重放一遍，得到"路径末端状态"。"""
    st = problem.initial_state()
    for act in result.path:
        st = problem.apply(st, act)
    return st


def test_answer_matches_returned_path() -> None:
    """回归测试：三内核返回的 answer 必须等于"返回路径末端状态"的判定。

    这条不是形式主义——MCTS 早期实现把 rollout 末端状态当最佳候选，却配上了扩展节点的
    tree id，导致 answer 来自 in-rollout 的状态、path 却停在树上，二者对不上
    （ACE 上 tr00402 就是这样把 Compliant 判成 Non-Compliant 的）。
    """
    for name, runner in (("bfs", lambda p: bfs(p)),
                         ("astar", lambda p: astar(p, w_h=1.5)),
                         ("mcts", lambda p: mcts(p, n_simulations=40, seed=3))):
        p = make_problem(tau=0.99 + 1e9)         # 目标不可达 -> 必走 anytime 回退分支
        r = runner(p)
        end = _replay(p, r)
        label, _conf = p.evaluate(end)
        assert label == r.answer, (name, label, r.answer, r.steps)


def test_trace_has_contract_fields() -> None:
    """文档 §2.4：trace 每行必须有 g/h/f/llm_reward/clf_score/rule_mask。"""
    need = {"g", "h", "f", "llm_reward", "clf_score", "rule_mask"}
    for name, runner in (("bfs", lambda p: bfs(p)),
                         ("astar", lambda p: astar(p, w_h=1.5))):
        r = runner(make_problem())
        assert r.trace, name
        for row in r.trace:
            missing = need - set(row)
            assert not missing, (name, missing, sorted(row))
            assert isinstance(row["rule_mask"], list), (name, row["rule_mask"])
            assert 0.0 <= row["llm_reward"] <= 1.0, (name, row["llm_reward"])


def test_mcts_trace_has_contract_fields() -> None:
    """文档 §2.4 要求三套内核的 trace 行都带 g/h/f/llm_reward/clf_score/rule_mask。

    MCTS 早期只记 sim/depth/reward，三内核的 trace 无法横向对比，可视化里的
    f 列与 llm_reward 列对 MCTS 也全是空的。
    """
    need = {"g", "h", "f", "llm_reward", "clf_score", "rule_mask"}
    r = mcts(make_problem(), n_simulations=40, seed=5)
    assert r.trace, "MCTS trace 为空"
    for row in r.trace:
        missing = need - set(row)
        assert not missing, (missing, sorted(row))
        assert isinstance(row["rule_mask"], list), row["rule_mask"]
        assert 0.0 <= row["llm_reward"] <= 1.0, row["llm_reward"]


def test_pruned_branches_are_recorded() -> None:
    """被规则剪掉的分支必须留在树里（status=pruned + rule_pruned 标记）。

    否则"节点颜色区分规则剪枝 / LLM 误杀"这条交付物做不出来：剪枝分支原本
    根本不进树，可视化只能画出成功与冗余两色。同时确认它不会虚增扩展节点数。
    """
    from hybrid.rule_engine import RuleEngine

    p = make_problem(tau=0.99 + 1e9,                     # 目标不可达 -> 会一直展开到出现 DEPENDS_ON
                     rules=RuleEngine(max_ref_chain=0))   # 任何 DEPENDS_ON 都被 R2 截断
    r = bfs(p)
    flags = [f for n in r.tree_root.nodes for f in n.flags]
    assert "rule_pruned" in flags, flags
    assert any(n.status == "pruned" for n in r.tree_root.nodes)
    assert r.expanded_nodes <= p.cfg.max_expansions
    # 剪枝节点不计入扩展节点：树里节点数必然多于扩展数
    assert len(r.tree_root.nodes) > r.expanded_nodes


def test_astar_tree_f_is_weighted() -> None:
    """树节点的 f 必须是排序真正用的 g + w·h，不能是 g + h。

    早期 record_state 里写死 f=g+h，w=1.5 时可视化上会出现"f 小的节点排在后"。
    """
    p = make_problem(h_fn=lambda s, _p: 1.0)
    r = astar(p, w_h=1.5)
    root = r.tree_root.nodes[0]
    assert abs(root.f - 1.5 * root.h) < 1e-9, (root.f, root.h)
    for n in r.tree_root.nodes:
        assert abs(n.f - (n.g + 1.5 * n.h)) < 1e-9, (n.tid, n.f, n.g, n.h)


def test_rule_reasons_are_read_only() -> None:
    """rule_mask 取 reasons() 而不是 mask()：不能把剪枝量数第二遍。

    早期 `pruned` 一列恒为 0（计数没回填），修好之后如果 trace 再调一次 `mask()`，
    剪枝量就会翻倍——这条断言把"只读"钉死。
    """
    from hybrid.rule_engine import RuleEngine

    p = make_problem()
    eng = RuleEngine(max_ref_chain=0)             # 任何 DEPENDS_ON 都算超链
    st = State(frozenset({"A"}), (("A#s0", E_EXCEPTIONS, "A#s1"),), 1, nodes=("A#s1",))
    act = Action("A#s1", E_DEPENDS_ON, "B#s0")

    assert "R2" in eng.reasons(st, act, p)
    eng.reasons(st, act, p)
    assert eng.pruned == 0 and eng.hits == {}, (eng.pruned, eng.hits)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[tests] {len(fns)} passed")
