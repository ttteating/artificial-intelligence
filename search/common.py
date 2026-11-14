"""三套内核共用的收尾逻辑：路径回填、成功路径标注、SearchResult 组装。"""

from __future__ import annotations

from state.core import Action, SearchResult, SearchProblem, State, trace_row
from search.tree import TreeRecorder


def path_tids(recorder: TreeRecorder, tid: int) -> list[int]:
    out = []
    cur = tid
    while cur is not None and cur >= 0:
        out.append(cur)
        cur = recorder.get(cur).parent
    return list(reversed(out))


def finalize(problem: SearchProblem, state: State, tid: int, recorder: TreeRecorder,
             expanded: int, trace: list[dict], goal: bool,
             best_fallback: tuple[float, State, int] | None = None,
             llm_calls: int = 0, pruned: int = 0, llm_rejected: int = 0) -> SearchResult:
    if not goal and best_fallback is not None:
        score, state, tid = best_fallback
    label, conf = problem.evaluate(state)
    tids = path_tids(recorder, tid) if tid >= 0 else []
    recorder.mark_success_path(tids)
    if tids:
        recorder.get(tid).status = "goal"
    recorder.mark_redundant(set(tids))

    return SearchResult(
        answer=label,
        path=state.path(),
        expanded_nodes=expanded,
        tree_root=recorder,
        trace=trace,
        sid=problem.sid,
        score=round(problem.terminal_score(state), 4),
        confidence=round(conf, 4),
        reached_goal=goal,
        n_clauses_visited=len(state.clauses),
        llm_calls=llm_calls,
        pruned_by_rule=pruned,
        llm_rejected=llm_rejected,
        margins={"n_relevant": len(problem.relevant), "n_clauses": problem.n_clauses,
                 "needs_exc": problem.needs_exc},
    )


def better(a: tuple[float, State, int] | None, b: tuple[float, State, int]):
    if a is None:
        return b
    return b if b[0] > a[0] else a


def aux_fields(problem: SearchProblem, state: State, priority_fn=None) -> dict:
    """补齐文档 §2.4 要求的两个 trace 字段：`llm_reward` 与 `rule_mask`。

    两个字段都必须**只读**，否则会污染主表：

    * `llm_reward` 取自融合器的 `components()`，内部有 `_r` 缓存；astar 的
      `adjust` 项先算过，这里只是命中缓存，不会新增评估。
    * `rule_mask` 只问规则引擎 `reasons()`，**故意不走 `mask()`**——后者会累加
      `pruned` / `hits`，在 trace 里再算一遍会让剪枝量翻倍（这个坑踩过：
      早期 `pruned` 列一直是 0，修好后又差点被 trace 二次计数）。
      同时复刻 `mask()` 的退化规则：某状态下动作全被剪光时实际不剪，掩码记为 `[]`。
    """
    llm_reward = 0.0
    comp = getattr(priority_fn, "components", None)
    if callable(comp):
        try:
            llm_reward = float(comp(state).get("llm_reward", 0.0) or 0.0)
        except Exception:          # 融合器异常不应打断搜索
            llm_reward = 0.0

    rule_mask: list[str] = []
    rules = getattr(problem, "rules", None)
    reasons = getattr(rules, "reasons", None)
    if callable(reasons):
        acts = problem.succ_actions(state)
        hits = [tuple(reasons(state, a, problem)) for a in acts]
        if acts and not all(hits):      # 全命中 -> 规则退化，实际不剪
            rule_mask = sorted({r for hs in hits for r in hs})
    return {"llm_reward": round(llm_reward, 4), "rule_mask": rule_mask}


def record_state(recorder: TreeRecorder, problem: SearchProblem, state: State,
                 parent: int | None, act: Action | None, h: float,
                 conf: float, label: str, status: str = "frontier",
                 f: float | None = None) -> int:
    node = state.nodes[-1] if state.nodes else None
    role = clause = None
    if node is not None:
        n = problem.graph.node(node)
        role, clause = n.role, n.clause
    return recorder.add(
        state.key(), state.depth, parent=parent, node_id=node,
        clause=clause, role=role, action=None if act is None else act.to_dict(),
        g=state.g, h=h, f=f, conf=conf, label=label, status=status,
    )


def record_pruned(problem: SearchProblem, recorder: TreeRecorder, state: State,
                  parent_tid: int | None, actions: list[Action]) -> int:
    """把"被剪掉、因此不会展开"的动作记成树上的 pruned 节点。

    交付物要求搜索树能区分「规则剪枝」与「LLM 误杀」，但被剪分支原本根本不进树，
    可视化只能画出成功/冗余两色。这里补上：

    * 规则剪掉的动作用 `reasons()` 能说得出理由 -> `rule_pruned`
    * 说不出理由的（HybridRules 里被 LLM 驳回的）-> `llm_rejected`

    `reasons()` 是只读的，不会累加 `pruned` 计数，因此主表的剪枝量不受影响；
    这些节点 status="pruned"，也不会被 `mark_redundant` 误算成冗余。
    """
    if not actions:
        return 0
    reasons = getattr(getattr(problem, "rules", None), "reasons", None)
    for a in actions:
        hit = tuple(reasons(state, a, problem)) if callable(reasons) else ()
        flag = "rule_pruned" if hit else "llm_rejected"
        node = problem.graph.node(a.dst) if a.dst != "ROOT" else None
        recorder.add(
            state_key=("pruned", state.key(), a.src, a.edge_type, a.dst),
            depth=state.depth + 1, parent=parent_tid,
            node_id=None if node is None else node.nid,
            clause=None if node is None else node.clause,
            role=None if node is None else node.role,
            action=a.to_dict(), g=state.g + 1.0, h=problem.heuristic(state),
            conf=None, label=None, status="pruned", flags=[flag],
        )
    return len(actions)
