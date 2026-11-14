"""MCTS：四阶段（Selection / Expansion / Rollout / Backpropagation）。

对齐文档 Day2 的要求：
* Selection：UCB1（c 可配）
* Expansion：**按边类型分层采样** —— 先挑"该类型下已试次数最少"的边类型，
  再在类型内随机取一条未试动作。这样 EXCEPTIONS / CONFLICTS 这类稀有边不会被
  COMPLEMENTS 的海量边淹没（纯均匀采样下，例外边被抽中的概率 < 10%）。
* Rollout：限深随机策略（默认只再走 rollout_depth 跳），reward 用终局的
  `terminal_score`（置信度 + 覆盖奖励）
* Backprop：reward 截断到 [0, reward_cap]，节点价值支持 mean / median 两种聚合
  （median 对 LLM 打分的长尾噪声更稳健，见报告 §LLM 分析）
* trace：与 BFS/A* 共用 `trace_row`，每行都带 g/h/f/llm_reward/clf_score/rule_mask，
  只记**树上的扩展**（rollout 不落 trace，也不调 LLM —— 与文档 §3.6 的成本策略一致）
* priority：可选的融合器（FusedPriority）。传进来只是为了把 llm_reward 写进 trace；
  MCTS 的搜索偏置仍由 `bias` 负责。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from state.core import Action, SearchProblem, SearchResult, State, trace_row
from search.common import aux_fields, better, finalize, record_pruned, record_state
from search.tree import TreeRecorder

PolicyBias = Callable[[State, Action], float]


@dataclass
class MNode:
    state: State
    parent: "MNode | None" = None
    action: Action | None = None
    children: list["MNode"] = field(default_factory=list)
    untried: list[Action] = field(default_factory=list)
    visits: int = 0
    rewards: list[float] = field(default_factory=list)
    tid: int = -1
    depth: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)
    pruned: list[Action] = field(default_factory=list)   # 该节点上被掩码剪掉的动作
    pruned_done: bool = False                            # 只在节点被扩展时入树一次

    def value(self, agg: str = "mean") -> float:
        if not self.rewards:
            return 0.0
        if agg == "median":
            r = sorted(self.rewards)
            return r[len(r) // 2]
        return sum(self.rewards) / len(self.rewards)


def _ucb(parent: MNode, child: MNode, c: float, agg: str) -> float:
    if child.visits == 0:
        return float("inf")
    return child.value(agg) + c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)


def _stratified_pick(node: MNode, rng: random.Random, bias: PolicyBias | None) -> Action:
    """按边类型分层：先取已试次数最少的类型，再在类型内按权重采样。"""
    buckets: dict[str, list[Action]] = {}
    for a in node.untried:
        buckets.setdefault(a.edge_type, []).append(a)
    if not buckets:
        raise IndexError("untried 为空")
    # 稀有权重：类型内未试动作越多的类型，被优先尝试的概率越高（保证覆盖）
    def score(t: str) -> tuple[int, int]:
        tried = node.type_counts.get(t, 0)
        return (tried, -len(buckets[t]))
    etype = min(buckets, key=score)
    cands = buckets[etype]
    if bias is None:
        return rng.choice(cands)
    ws = [max(0.05, 1.0 + bias(node.state, a)) for a in cands]
    tot = sum(ws)
    r = rng.random() * tot
    acc = 0.0
    for a, w in zip(cands, ws):
        acc += w
        if r <= acc:
            return a
    return cands[-1]


def mcts(problem: SearchProblem, n_simulations: int = 90, c_ucb: float = 1.3,
         rollout_depth: int = 2, reward_cap: float = 1.0, agg: str = "mean",
         seed: int = 0, recorder: TreeRecorder | None = None,
         budget: int | None = None, bias: PolicyBias | None = None,
         hybrid_weight: float = 0.0, priority=None) -> SearchResult:
    rng = random.Random(seed)
    recorder = recorder or TreeRecorder()
    budget = budget or problem.cfg.max_expansions

    root_state = problem.initial_state()
    label0, conf0 = problem.evaluate(root_state)
    h0 = problem.heuristic(root_state)
    root_tid = record_state(recorder, problem, root_state, None, None, h0, conf0, label0)
    root_untried, root_pruned = problem.allowed_actions(root_state)
    root = MNode(state=root_state, tid=root_tid, depth=0,
                 untried=root_untried, pruned=root_pruned)

    expanded = 1
    trace: list[dict] = []
    best = None                      # (score, state, tid)
    n_rollouts = 0

    def clip(r: float) -> float:
        return max(0.0, min(reward_cap, r))

    for sim in range(n_simulations):
        node = root

        # ---------------- 1. Selection
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: _ucb(node, ch, c_ucb, agg))

        # ---------------- 2. Expansion
        if node.untried and expanded < budget:
            # 被剪分支在"节点真正被扩展"时入树一次，与 BFS/A* 的口径一致；
            # 只创建未展开的节点时也记会画出没有父节点的孤儿剪枝。
            if not node.pruned_done:
                record_pruned(problem, recorder, node.state, node.tid, node.pruned)
                node.pruned_done = True
            act = _stratified_pick(node, rng, bias)
            node.untried.remove(act)
            node.type_counts[act.edge_type] = node.type_counts.get(act.edge_type, 0) + 1
            nxt = problem.apply(node.state, act)
            h = problem.heuristic(nxt)
            l2, c2 = problem.evaluate(nxt)
            ctid = record_state(recorder, problem, nxt, node.tid, act, h, c2, l2,
                                status="expanded")
            child_untried, child_pruned = problem.allowed_actions(nxt)
            child = MNode(state=nxt, parent=node, action=act, tid=ctid,
                          depth=node.depth + 1, untried=child_untried,
                          pruned=child_pruned)
            # trace 按文档 §2.4 的统一步长记录（此前只记 sim/reward，缺 g/h/f/
            # llm_reward/clf_score/rule_mask，三内核的 trace 因此不可比）
            trace.append(trace_row(nxt, act, nxt.g, h, l2, c2,
                                   {"expanded": True, "clf_score": c2, "sim": sim,
                                    **aux_fields(problem, nxt, priority)}))
            node.children.append(child)
            expanded += 1
            node = child
            # anytime 候选：所有**树上**的扩展节点（有 tid、有路径），不只是命中目标的那些
            best = better(best, (problem.terminal_score(nxt), nxt, ctid))

        # ---------------- 3. Rollout（限深）
        st = node.state
        depth_cap = min(problem.cfg.max_depth, st.depth + rollout_depth)
        steps = 0
        while st.depth < depth_cap:
            acts, _ = problem.allowed_actions(st)
            if not acts:
                break
            if bias is None or rng.random() < 0.5:
                a = rng.choice(acts)
            else:
                ws = [max(0.05, 1.0 + bias(st, x)) for x in acts]
                a = rng.choices(acts, weights=ws, k=1)[0] if rng.random() < hybrid_weight + 0.2 else rng.choice(acts)
            st = problem.apply(st, a)
            steps += 1
        n_rollouts += 1
        reward = clip(problem.terminal_score(st))
        # 注意：这里**不能**把 st 当作 best 候选。
        # st 是 rollout 末端状态（比 node 深 1~2 跳），它不在树上、没有对应的 tree id；
        # 早期实现写成 better(best, (score(st), st, node.tid))，于是"答案取自 st、
        # 路径取自 node.tid"，返回的答案可能不落在返回的路径上（tr00402 就是这样错的）。
        # 修正后 best 只从树节点取，与 BFS/A* 的 anytime 语义一致。

        # ---------------- 4. Backprop
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.rewards.append(reward)
            if len(cur.rewards) > 64:
                cur.rewards = cur.rewards[-64:]
            cur = cur.parent

        if expanded >= budget and sim > 8:
            # 预算耗尽即停：trace 只保留**扩展开销**的行（每行都是完整契约字段）。
            # rollout 末端状态不在树上、也没有 tree id，若把它写进 trace，就会出现
            # 「行里有状态但树上找不到节点」的错行；rollout 次数已记在 margins 里。
            break

    # 树统计回写：visits + value 便于可视化
    def push(n: MNode):
        tn = recorder.get(n.tid)
        tn.visits = n.visits
        tn.value = n.value(agg)
        for ch in n.children:
            push(ch)

    push(root)

    final_state = root.state if best is None else best[1]
    final_tid = root_tid if best is None else best[2]
    # 用 MCTS 访问量最高的子节点路径作为"最终路径"（若它比 best 更受支持）
    res = finalize(problem, final_state, final_tid, recorder, expanded, trace,
                   goal=problem.is_goal(final_state), best_fallback=best)
    res.margins["rollouts"] = n_simulations
    res.margins["rollout_steps"] = n_rollouts
    return res
