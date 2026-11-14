"""A*（及加权 A*、混合优先级变体）。

    f(n) = g(n) + w_h · h_θ(n)          ── A0 / A1
    f(n) = g + w_h·h − λ1·r_llm − λ2·p_clf + λ4·risk   ── A2 / H（由 priority_fn 注入）

* h ≡ 0 且 w_h 任意时，按 g 的层序扩展，与 BFS 在同一目标测试下返回同一答案
  （tests/test_kernels.py 里有断言）。
* closed set 用 State.key()（条款集合 + 边类型多重集），与 BFS 共用同一抽象，
  避免"A* 因为状态压缩更狠而显得更省"这种伪优势。
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable

from state.core import SearchProblem, SearchResult, State, trace_row
from search.common import aux_fields, better, finalize, record_pruned, record_state
from search.tree import TreeRecorder

PriorityFn = Callable[[State, float, float], float]   # (state, g, h) -> 调整量（越大越差）


def astar(problem: SearchProblem, w_h: float = 1.5, recorder: TreeRecorder | None = None,
          budget: int | None = None, priority_fn: PriorityFn | None = None) -> SearchResult:
    recorder = recorder or TreeRecorder()
    budget = budget or problem.cfg.max_expansions

    root = problem.initial_state()
    label0, conf0 = problem.evaluate(root)
    h0 = problem.heuristic(root)
    root_tid = record_state(recorder, problem, root, None, None, h0, conf0, label0,
                            f=h0 * w_h)

    tie = itertools.count()
    open_heap: list[tuple[float, int, State, int]] = [(h0 * w_h, next(tie), root, root_tid)]
    best_g: dict[tuple, float] = {root.key(): 0.0}
    closed: set[tuple] = set()
    expanded = 0
    trace: list[dict] = []
    best = None

    while open_heap:
        f, _, state, tid = heapq.heappop(open_heap)
        k = state.key()
        if k in closed:
            continue
        if state.g > best_g.get(k, 1e9) + 1e-9:
            continue
        closed.add(k)
        if expanded >= budget:
            break
        expanded += 1
        recorder.get(tid).status = "expanded"

        h = problem.heuristic(state)
        label, conf = problem.evaluate(state)
        trace.append(trace_row(state, None, state.g, h, label, conf,
                               {"f": round(state.g + w_h * h, 4), "expanded": True,
                                "clf_score": conf,
                                "adjust": 0.0 if priority_fn is None else round(priority_fn(state, state.g, h), 4),
                                **aux_fields(problem, state, priority_fn)}))

        if problem.is_goal(state):
            return finalize(problem, state, tid, recorder, expanded, trace, goal=True)

        if state.depth > 0:
            best = better(best, (problem.terminal_score(state), state, tid))

        succ, pruned = problem.expand(state)
        record_pruned(problem, recorder, state, tid, pruned)
        for act, nxt in succ:
            nk = nxt.key()
            if nk in closed:
                continue
            if nxt.g >= best_g.get(nk, 1e9) - 1e-9:
                continue
            best_g[nk] = nxt.g
            hn = problem.heuristic(nxt)
            adj = 0.0 if priority_fn is None else priority_fn(nxt, nxt.g, hn)
            fn = nxt.g + w_h * hn + adj
            l2, c2 = problem.evaluate(nxt)
            ctid = record_state(recorder, problem, nxt, tid, act, hn, c2, l2, f=fn)
            heapq.heappush(open_heap, (fn, next(tie), nxt, ctid))

    return finalize(problem, root if best is None else best[1],
                    -1 if best is None else best[2], recorder, expanded, trace,
                    goal=False, best_fallback=best)
