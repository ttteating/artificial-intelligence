"""推理自然语言解释器的单元测试（不依赖 pytest / 网络）。

`python -m tests.test_explainer`

三条断言对应课程第 3 项的三条口径：
  1. 解释必须覆盖路径每一步，且**不出现合规标签词**（模板本身就是合规的）
  2. 真实模型若越权给出标签词，必须整段作废、回退模板、记 violation
  3. 真实模型合规输出时，事实字段（steps/citations/signals）仍来自模板，
     不采信模型自述 —— 避免"解释编造证据"
"""

from __future__ import annotations

import json

from data.clause_graph import (
    E_EXCEPTIONS, ROLE_EXC, ROLE_OBL, Edge, Node, ScenarioGraph,
)
from llm.adapter import LABEL_WORDS
from llm.explainer import LLMExplainer, TemplateExplainer, explain_result
from search.bfs import bfs
from state.core import ProblemConfig, SearchProblem, State


def toy_problem() -> SearchProblem:
    g = ScenarioGraph(sid="toy")
    g.nodes["A#s0"] = Node("A#s0", "A", ROLE_OBL, "Licensee shall pay the fee within 30 days.", 0)
    g.nodes["A#s1"] = Node("A#s1", "A", ROLE_EXC,
                           "except where the fee is waived under the trial programme.", 1)
    g.entry = ["A#s0"]
    e = Edge("A#s0", E_EXCEPTIONS, "A#s1")
    g.out.setdefault("A#s0", []).append(e)
    g.inc.setdefault("A#s1", []).append(e)

    def decision(st: State):
        return ("Compliant", 0.95) if "A#s1" in st.nodes else ("Non-Compliant", 0.55)

    return SearchProblem(g, "the fee was waived under the trial exception",
                         {"A": 0.6}, decision=decision,
                         h_fn=lambda s, p: 0.0,
                         cfg=ProblemConfig(max_depth=2, max_expansions=8, tau_conf=0.9))


def _has_label_word(txt: str) -> bool:
    return any(w in (txt or "").lower() for w in LABEL_WORDS)


def test_template_explains_every_step() -> None:
    p = toy_problem()
    r = bfs(p)
    exp = TemplateExplainer().explain(r, p)

    assert exp.text and not _has_label_word(exp.text), exp.text
    assert len(exp.steps) == len(r.path) >= 2, (len(exp.steps), len(r.path))
    assert exp.steps[1]["edge"] == E_EXCEPTIONS
    assert exp.steps[1]["role"] == ROLE_EXC
    assert exp.citations and exp.citations[0]["clause"] == "A"
    assert exp.sufficiency["coverage_ok"] is True
    assert exp.mode == "template" and exp.violation is False


class _FakeClient:
    """最小 LLMClient 替身：只实现 json_call 与 available。"""

    def __init__(self, obj: dict):
        self.obj = obj
        self.calls = 0
        self.available = True

    def json_call(self, system: str, user: str, tag: str = "") -> dict:
        self.calls += 1
        return dict(self.obj)


def test_real_explainer_rejects_label_leak() -> None:
    """模型越权给结论 -> 整段作废，回退模板，violation=True。"""
    p = toy_problem()
    r = bfs(p)
    fake = _FakeClient({"explanation": "The clause is Non-Compliant, so the search stops.",
                        "step_notes": [], "sufficiency_note": ""})
    exp = LLMExplainer(fake).explain(r, p)

    assert exp.violation is True
    assert exp.mode == "real->template"
    assert not _has_label_word(exp.text)
    assert exp.raw and "violation_obj" in exp.raw


def test_real_explainer_keeps_fact_fields() -> None:
    """合规输出被采用，但 steps/citations/signals 仍来自模板。"""
    p = toy_problem()
    r = bfs(p)
    fake = _FakeClient({
        "explanation": "搜索先读主句确认付款义务，再沿例外边检查豁免是否成立。",
        "step_notes": ["读主句", "查例外"], "sufficiency_note": "证据覆盖完整",
    })
    exp = LLMExplainer(fake).explain(r, p)

    assert exp.mode == "real" and exp.violation is False
    assert exp.text.startswith("搜索先读主句")
    assert len(exp.steps) == len(r.path)
    assert exp.citations[0]["clause"] == "A"
    assert exp.steps[0].get("llm_note") == "读主句"


def test_explain_result_writes_back() -> None:
    p = toy_problem()
    r = bfs(p)
    d = explain_result(r, p, mode="template")
    assert isinstance(d, dict) and d["text"]
    assert r.explanation == d
    # 解释不改变搜索结果本身
    assert r.answer == "Compliant" and r.reached_goal is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[tests] {len(fns)} passed")
