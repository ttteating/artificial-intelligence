"""BFS：盲目搜索基准（文档条件 B0 / 作为 A* 的一致性对照）。

特点
----
* 按深度逐层扩展，**不使用任何启发信息**（h 恒为 0）
* closed set 与 A* 完全一致（条款集合 + 边类型多重集），保证两者可比
* 找到第一个满足"证据充分性"的目标即返回；预算耗尽则返回已访问终局中的最优者
"""

from __future__ import annotations

from collections import deque

from state.core import SearchProblem, SearchResult, trace_row
from search.common import aux_fields, better, finalize, record_pruned, record_state
from search.tree import TreeRecorder


def bfs(problem: SearchProblem, recorder: TreeRecorder | None = None,
        budget: int | None = None) -> SearchResult:
    recorder = recorder or TreeRecorder()
    budget = budget or problem.cfg.max_expansions

    root = problem.initial_state()
    label0, conf0 = problem.evaluate(root)
    root_tid = record_state(recorder, problem, root, None, None, 0.0, conf0, label0)

    frontier = deque([(root, root_tid)])
    closed: set[tuple] = {root.key()}
    expanded = 0
    trace: list[dict] = []
    best = None

    while frontier:
        state, tid = frontier.popleft()
        if expanded >= budget:
            break
        expanded += 1
        recorder.get(tid).status = "expanded"
        label, conf = problem.evaluate(state)
        trace.append(trace_row(state, None, state.g, 0.0, label, conf,
                               {"expanded": True, "clf_score": conf,
                                **aux_fields(problem, state)}))

        if problem.is_goal(state):
            return finalize(problem, state, tid, recorder, expanded, trace, goal=True)

        # anytime 语义：预算耗尽时返回"已见到的最好证据状态"，而不是退回根节点
        if state.depth > 0:
            best = better(best, (problem.terminal_score(state), state, tid))

        succ, pruned = problem.expand(state)
        record_pruned(problem, recorder, state, tid, pruned)
        if not succ:
            continue

        for act, nxt in succ:
            k = nxt.key()
            if k in closed:
                continue
            closed.add(k)
            l2, c2 = problem.evaluate(nxt)
            ctid = record_state(recorder, problem, nxt, tid, act, 0.0, c2, l2)
            frontier.append((nxt, ctid))

    return finalize(problem, root if best is None else best[1],
                    -1 if best is None else best[2], recorder, expanded, trace,
                    goal=False, best_fallback=best)
