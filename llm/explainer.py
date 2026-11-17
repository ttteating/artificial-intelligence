"""推理自然语言解释器（补课程第 3 项「LLM 推理自然语言解释」）。

口径（与 `adapter.py` 的三条硬约束一致）
------------------------------------
课程要求 LLM 承担三件事：动作合法性校验、路径 reward、**推理自然语言解释**，
且「模型不能直接输出答案，只能给搜索树节点评估值」。因此这里的解释是
**事后解释（post-hoc explanation）**：输入是已经跑完的搜索轨迹、规则命中与
LLM/clf 信号，输出的是「这条路径为什么这样走、证据是否充分」的说明文字。

两条硬约束（比搜索侧更严，因为解释直接给人看）：
  1. 解释文本不得出现合规标签词（Compliant / Non-Compliant / Not-Applicable）。
     一旦出现，说明模型越权给了结论 —— 整段丢弃、回退模板，并记 `violation=True`，
     原文留在 `raw` 字段供审计（报告里要能数出"越权率"）。
  2. 解释**不参与搜索**：它只写进 `SearchResult.explanation`，不影响答案、扩展数、
     主表任何一列。默认关闭，只在 `tools/explain_case.py` 或 `/api/search?explain=1`
     时启用。

两种模式
--------
* `template`：完全离线、确定性。按路径分步生成中文说明，数字全部取自
  `result` / `problem`，不依赖任何外部服务 —— 答辩机房断网也能演示。
* `real`：调用 OpenAI 兼容接口，只让模型描述证据链，禁止下结论；无 key 或调用
  失败时自动回退模板，并在 `mode` 里标 `real->template`。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from data.clause_graph import (
    E_CITES, E_COMPLEMENTS, E_CONFLICTS, E_DEFINES, E_DEPENDS_ON, E_EXCEPTIONS,
    E_SEQUENCE, ROLE_COND, ROLE_DEF, ROLE_EXC, ROLE_OBL, ROLE_REMEDY, ROLE_SCOPE,
)
from llm.adapter import LABEL_WORDS, LLMClient

# ---------------------------------------------------------------- 语义词表
ROLE_CN = {
    ROLE_OBL: "义务/授权主句", ROLE_EXC: "例外句", ROLE_COND: "条件句",
    ROLE_DEF: "定义句", ROLE_SCOPE: "交叉引用句", ROLE_REMEDY: "救济/责任句",
}

# 这一步"在法律推理上是在做什么"——解释的主体就是它，而不是分数
EDGE_CN = {
    "ENTRY": "从场景切入条款图中的一个语义单元，作为推理起点（通常是该条款里与场景最相关的一句）",
    E_EXCEPTIONS: "走向例外句，检查刚刚读到的义务是否被豁免",
    E_CONFLICTS: "走向与当前条款存在冲突关系的条款，检查两条规则是否互相排斥",
    E_DEPENDS_ON: "走向条件句，确认义务的触发前提是否成立",
    E_DEFINES: "走向定义句，确认术语的确切含义",
    E_CITES: "走向被引用条款，确认本条规则的适用范围",
    E_COMPLEMENTS: "走向补充句（救济/责任），补齐违反后的后果",
    E_SEQUENCE: "走向时间/顺序条款，确认义务的先后次序",
}

EXPLAIN_SYS = (
    "You are a legal-reasoning explainer for a search system. You receive the SCENARIO, "
    "the STEPS the search actually took on a clause graph, the EVIDENCE coverage and the "
    "node evaluation signals. Explain, in Chinese, WHY the search took these steps and "
    "whether the gathered evidence is sufficient. "
    "Hard rules: (1) NEVER state or imply a compliance conclusion or label — do not use "
    "the words Compliant, Non-Compliant or Not-Applicable in any form; (2) do not invent "
    "clauses or facts not present in the input; (3) refer to clauses by their ids "
    "(evidence_N) and quote at most a short fragment; (4) 120-180 Chinese characters. "
    'Reply with JSON only: {"explanation": "...", "step_notes": ["..."], '
    '"sufficiency_note": "<=40 chars"}'
)


# ---------------------------------------------------------------- 数据结构
@dataclass
class Explanation:
    """解释的结构化结果，可直接 JSON 化给前端。"""

    text: str
    steps: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    sufficiency: dict = field(default_factory=dict)
    mode: str = "template"          # template | real | real->template
    violation: bool = False
    llm_calls: int = 0
    raw: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- 工具函数
def _fmt(x, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _excerpt(text: str, n: int = 90) -> str:
    t = " ".join((text or "").split())
    return t[:n] + ("…" if len(t) > n else "")


def _replay_final_state(problem, result):
    """按 result.path 重放一遍，拿到终局 State（用于覆盖/例外判定，只读）。

    不用 trace 的最后一行代替：trace 记录的是"扩展过的状态"，而最终答案是
    在 `finalize` 里由 path 末状态给出的，两者在"取最优终局"的场景下可能不同。
    """
    st = problem.initial_state()
    for a in result.path:
        st = problem.apply(st, a)
    return st


def _trace_by_action(trace: list[dict]) -> dict[tuple, dict]:
    idx: dict[tuple, dict] = {}
    for row in trace:
        a = row.get("action") or {}
        if not a:
            continue
        k = (a.get("src"), a.get("edge_type"), a.get("dst"))
        idx[k] = row          # 后出现的覆盖先出现的：取最后一次评估
    return idx


# ---------------------------------------------------------------- 模板解释器
class TemplateExplainer:
    """离线确定性解释：把路径、规则命中、信号拼成中文说明。

    刻意不写"所以该合同合规"这类句子 —— 解释的职责是说明**证据链**，
    结论由 `result.answer` 单独给出，两者在页面上分开展示。
    """

    mode = "template"

    def explain(self, result, problem, limit_steps: int = 8) -> Explanation:
        g = problem.graph
        trace_idx = _trace_by_action(result.trace or [])
        st = _replay_final_state(problem, result)

        # ---- 分步说明
        steps: list[dict] = []
        for i, a in enumerate(result.path[:limit_steps], start=1):
            dst = g.node(a.dst) if a.dst != "ROOT" else None
            row = trace_idx.get((a.src, a.edge_type, a.dst), {})
            note = EDGE_CN.get(a.edge_type, "沿条款图扩展一步")
            if dst is not None:
                note += f"；目标句被判为「{ROLE_CN.get(dst.role, dst.role)}」"
                fs = (getattr(problem, "facet_sim", {}) or {}).get(dst.nid, 0.0)
                if fs:
                    note += f"，与场景的 facet 相似度 {_fmt(fs, 2)}"
            steps.append({
                "i": i,
                "from": a.src, "to": a.dst, "edge": a.edge_type,
                "clause": None if dst is None else dst.clause,
                "role": None if dst is None else dst.role,
                "excerpt": "" if dst is None else _excerpt(dst.text),
                "g": row.get("g"), "h": row.get("h"), "f": row.get("f"),
                "conf": row.get("conf"),
                "llm_reward": row.get("llm_reward"),
                "clf_score": row.get("clf_score"),
                "rule_mask": row.get("rule_mask") or [],
                "note": note,
            })

        # ---- 证据充分性（三条目标测试）
        covered = sorted(st.clauses)
        relevant = sorted(getattr(problem, "relevant", []) or [])
        cov_ok = problem.coverage_ok(st)
        exc_ok = problem.exception_ok(st)
        conf = float(result.confidence or 0.0)
        sufficiency = {
            "covered_clauses": covered,
            "relevant_clauses": relevant,
            "coverage_ok": bool(cov_ok),
            "exception_required": bool(getattr(problem, "needs_exc", False)),
            "exception_ok": bool(exc_ok),
            "confidence": round(conf, 4),
            "tau_conf": float(problem.cfg.tau_conf),
            "confidence_ok": conf >= problem.cfg.tau_conf,
            "reached_goal": bool(result.reached_goal),
        }

        # ---- 信号（终局三项 + 规则命中）
        last = result.trace[-1] if result.trace else {}
        risk = 0.0
        if result.path:
            try:
                risk = float(problem.rules.risk(st, result.path[-1], problem))
            except Exception:      # 规则引擎缺失或接口不同都不该让解释失败
                risk = 0.0
        hits = getattr(getattr(problem, "rules", None), "hits", {}) or {}
        signals = {
            "llm_reward": last.get("llm_reward", 0.0),
            "clf_score": last.get("clf_score"),
            "risk": round(risk, 4),
            "rule_hits": hits,
            "expanded_nodes": result.expanded_nodes,
            "pruned_by_rule": result.pruned_by_rule,
            "llm_rejected": result.llm_rejected,
            "llm_suspicious": result.llm_suspicious,
            "llm_calls": result.llm_calls,
        }

        # ---- 文本
        n_nodes, n_edges = len(g.nodes), sum(len(v) for v in g.out.values())
        head = (f"该场景含 {problem.n_clauses} 条条款，条款图切分为 {n_nodes} 个 facet、"
                f"{n_edges} 条带类型边。搜索共扩展 {result.expanded_nodes} 个节点，"
                f"最终证据链为 {result.steps} 步。")
        if steps:
            # 连续走同类边时区分"第几个分支"，否则文本会重复同一句话
            same_type_count: dict[str, int] = {}
            prev_edge = None
            segs = []
            for s in steps:
                if s["edge"] == prev_edge:
                    same_type_count[s["edge"]] = same_type_count.get(s["edge"], 1) + 1
                    tag = f"（同为 {s['edge']} 边的第 {same_type_count[s['edge']]} 个分支）"
                else:
                    same_type_count[s["edge"]] = 1
                    prev_edge = s["edge"]
                    tag = ""
                segs.append(
                    f"第 {s['i']} 步沿 {s['edge']} 边从 {s['from']} 走到 {s['to']}"
                    f"（{ROLE_CN.get(s['role'], s['role'] or '未知')}）{tag}，"
                    f"{EDGE_CN.get(s['edge'], '扩展一步')}")
            body = "推理链条：" + "；".join(segs) + "。"
        else:
            body = "本次搜索未产生有效路径（在预算内没有找到满足目标测试的状态），因此没有证据链可解释。"

        missing = [c for c in relevant if c not in covered]
        tail = (f"证据充分性：强相关条款 {len(relevant)} 条，已审查 {len(covered)} 条"
                f"（覆盖{'达标' if cov_ok else '未达标'}"
                + (f"，缺 {'/'.join(missing)}" if missing else "") + "）；"
                f"{'场景出现违约线索，例外须解开且已解开' if (sufficiency['exception_required'] and exc_ok) else ('场景出现违约线索，例外须解开但未解开' if sufficiency['exception_required'] else '场景未见违约线索，不强制解开例外')}；"
                f"终局置信度 {_fmt(conf)}，门槛 {_fmt(problem.cfg.tau_conf, 2)}。"
                f"三项目标测试{'同时满足，搜索在此终止' if result.reached_goal else '未同时满足，结果取自预算内最优终局'}。")
        if signals["pruned_by_rule"] or signals["llm_rejected"]:
            tail += (f" 期间规则引擎剪掉 {signals['pruned_by_rule']} 个高风险动作，"
                     f"LLM 驳回 {signals['llm_rejected']} 个语义不合法动作"
                     f"（其中低置信但合法 {signals['llm_suspicious']} 个进入可疑缓冲区）。")

        # ---- 三项融合信号的归因（只在有分类器/LLM 的条件下出现）
        if signals.get("clf_score") is not None:
            try:
                from experiments.factory import LAMBDA

                lam = LAMBDA
            except Exception:      # noqa: BLE001 - 单独使用本模块时不强依赖实验包
                lam = {"llm": 0.15, "clf": 0.30, "risk": 0.40}
            tail += (f" 终局状态的三项融合信号：llm_reward {_fmt(signals['llm_reward'], 2)}"
                     f"（路径语义相关度）、clf_score {_fmt(signals['clf_score'])}"
                     f"（分类器判为正确路径的概率）、risk {_fmt(signals['risk'], 2)}（规则风险）；"
                     f"A* 的优先级调整项按 −{lam['llm']}·reward −{lam['clf']}·clf "
                     f"+{lam['risk']}·risk 生效，数值越小越先展开。")
        tail += " 本条解释只描述证据链与充分性，最终三分类结果由终局评估器单独给出。"


        citations = []
        for c in covered:
            # 该条款第一条 facet 的原文片段（图里就是条款文本本身，不另做摘要）
            txt = next((n.text for n in g.nodes.values() if n.clause == c), "")
            citations.append({"clause": c, "excerpt": _excerpt(txt, 110)})

        return Explanation(
            text="\n\n".join([head, body, tail]).strip(),
            steps=steps, citations=citations, signals=signals,
            sufficiency=sufficiency, mode="template",
        )


# ---------------------------------------------------------------- 真实 LLM 解释器
class LLMExplainer:
    """真实模型解释：只让它描述证据链，越权给结论就整段作废。"""

    mode = "real"

    def __init__(self, client: LLMClient, base: TemplateExplainer | None = None):
        self.client = client
        self.base = base or TemplateExplainer()

    def _payload(self, result, problem, tpl: Explanation) -> str:
        return json.dumps({
            "SCENARIO": _excerpt(problem.text, 900),
            "CONDITION": getattr(problem, "llm_mode", "off"),
            "STEPS": [{k: s[k] for k in ("i", "from", "to", "edge", "role", "clause", "excerpt")}
                      for s in tpl.steps],
            "EVIDENCE": {
                "relevant_clauses": tpl.sufficiency["relevant_clauses"],
                "covered_clauses": tpl.sufficiency["covered_clauses"],
                "exception_required": tpl.sufficiency["exception_required"],
                "exception_resolved": tpl.sufficiency["exception_ok"],
            },
            "SIGNALS": {k: v for k, v in tpl.signals.items() if k != "rule_hits"},
            "TEMPLATE_DRAFT": tpl.text,      # 允许模型改写，但不许改事实
        }, ensure_ascii=False)

    def explain(self, result, problem, limit_steps: int = 8) -> Explanation:
        tpl = self.base.explain(result, problem, limit_steps=limit_steps)
        # ---- 真实调用
        try:
            obj = self.client.json_call(EXPLAIN_SYS, self._payload(result, problem, tpl),
                                        tag="explain")
        except Exception:                    # noqa: BLE001 - 网络/配额失败一律回退
            tpl.mode = "real->template"
            tpl.raw = {"error": "llm_call_failed"}
            return tpl

        text = str(obj.get("explanation", "")).strip()
        notes = obj.get("step_notes") or []
        suf_note = str(obj.get("sufficiency_note", "")).strip()

        # ---- 越权检查：解释里出现标签词即作废
        low = " ".join([text, suf_note, " ".join(map(str, notes))]).lower()
        if not text or any(w in low for w in LABEL_WORDS):
            tpl.mode = "real->template"
            tpl.violation = True
            tpl.llm_calls = max(1, getattr(self.client, "calls", 0))
            tpl.raw = {"violation_obj": obj}
            return tpl

        # 模型改写仍然保留事实字段（steps/citations/signals）来自模板，不采信模型自述
        if isinstance(notes, list):
            for i, s in enumerate(tpl.steps):
                if i < len(notes) and notes[i]:
                    s["llm_note"] = str(notes[i])[:160]
        out = Explanation(
            text=text if not suf_note else f"{text}\n\n{suf_note}",
            steps=tpl.steps, citations=tpl.citations, signals=tpl.signals,
            sufficiency=tpl.sufficiency, mode="real",
            llm_calls=1, raw=obj,
        )
        return out


# ---------------------------------------------------------------- 入口
def build_explainer(mode: str = "auto", client: LLMClient | None = None):
    """mode: template | real | auto。auto 有 key 用 real，否则 template。"""
    if mode == "template":
        return TemplateExplainer()
    cli = client or LLMClient()
    if mode == "real" and cli.available:
        return LLMExplainer(cli)
    if mode == "real":
        print("[explain] 未检测到 LLM_API_KEY，回退模板解释")
    return TemplateExplainer()


def explain_result(result, problem, mode: str = "template", client: LLMClient | None = None
                   ) -> dict:
    """对一次已完成的搜索结果做解释，返回可 JSON 化的 dict。"""
    ex = build_explainer(mode, client=client)
    exp = ex.explain(result, problem)
    result.explanation = exp.to_dict()
    return result.explanation


def attach_and_dump(result, problem, out_path, mode: str = "template") -> dict:
    """解释 + 落盘（供 tools/explain_case.py 与可视化导出复用）。"""
    from pathlib import Path

    exp = explain_result(result, problem, mode=mode)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
    return exp
