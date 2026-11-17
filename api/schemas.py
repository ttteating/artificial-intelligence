"""FastAPI 请求/响应模型（pydantic v2）。

只做"接口形状"的校验与文档化，业务逻辑全在 `api.service`。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Split = Literal["dev", "test"]
Condition = Literal["B0", "B1", "A0", "A1", "A2", "M0", "M1", "H"]


class Health(BaseModel):
    status: str = Field(description="ok / loading / error")
    ready: bool = Field(description="语料与模型是否已加载完成")
    uptime_sec: float
    n_scenarios: int = 0
    llm_mode: str = "mock"
    graph_version: str = ""
    error: str | None = None


class ConditionInfo(BaseModel):
    id: str
    kernel: str
    rules: bool
    h: str
    llm: str
    clf: bool
    description: str


class SearchRequest(BaseModel):
    sid: str = Field(description="ACE 场景 id，例如 tr00632 / te00262")
    condition: Condition = "H"
    explain: bool = Field(default=False, description="是否生成推理自然语言解释")
    explain_mode: Literal["template", "real", "auto"] = "template"
    max_expansions: int | None = Field(default=None, ge=1, le=48,
                                       description="覆盖扩展节点预算（默认取 config.yaml）")
    include_trace: bool = True
    include_tree: bool = True


class StepOut(BaseModel):
    i: int
    src: str
    edge_type: str
    dst: str


class SearchResponse(BaseModel):
    sid: str
    condition: str
    answer: str
    confidence: float
    steps: int
    expanded_nodes: int
    reached_goal: bool
    llm_calls: int
    pruned_by_rule: int
    llm_rejected: int
    llm_mode: str
    tree: list[dict] | None = None
    n_nodes_raw: int = 0
    trace: list[dict] | None = None
    path: list[dict] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] | None = None
    elapsed_ms: float = 0.0


class ScenarioRun(BaseModel):
    condition: str
    answer: str
    gold: str
    correct: bool
    expanded: int
    steps: int
    redundant: int
    llm_calls: int
    llm_rejected: int
    conf: float


class ScenarioOut(BaseModel):
    sid: str
    split: str
    gold: str
    n_clauses: int
    oracle_hops: int | None = None
    text: str
    clauses: dict[str, str]
    runs: list[ScenarioRun] = Field(default_factory=list)


class ScenarioListItem(BaseModel):
    sid: str
    split: str
    gold: str
    n_clauses: int
    oracle_hops: int | None = None
    runs: dict[str, dict] = Field(default_factory=dict)


class ScenarioList(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ScenarioListItem]


class GraphOut(BaseModel):
    sid: str
    graph_version: str
    n_nodes: int
    n_edges: int
    entry: list[str]
    nodes: list[dict]
    edges: list[dict]


class ExplainRequest(BaseModel):
    sid: str
    condition: Condition = "H"
    mode: Literal["template", "real", "auto"] = "template"


class TableOut(BaseModel):
    name: str
    kind: Literal["json", "csv", "markdown"]
    n_rows: int = 0
    columns: list[str] = Field(default_factory=list)
    data: Any = None
    text: str | None = None


# ---------------------------------------------------------------- 自定义合同
class CustomScenarioIn(BaseModel):
    """客户自己的合同：一段场景描述 + 1~10 条条款。

    与 ACE 样本同构（`scenario_text + clauses`），因此可以复用同一条
    `build_graph -> build_problem -> run_kernel` 链路，不需要另写推理代码。
    """

    scenario_text: str = Field(min_length=1, max_length=20000,
                               description="自然语言场景，例如违约事实或待审交易描述")
    clauses: dict[str, str] = Field(min_length=1, max_length=10,
                                    description="{evidence_1: 条款原文, ...}，键可自定义，服务端会归一")
    title: str | None = Field(default=None, max_length=120)
    label: str | None = Field(default=None, description="已知标注（可空，仅用于对照）")
    persist: bool = Field(default=False, description="是否落盘到 data/custom/（默认只在内存）")


class CustomScenarioOut(BaseModel):
    sid: str
    title: str
    n_clauses: int
    n_nodes: int
    n_edges: int
    entry: list[str]
    clause_sim: dict[str, float]
    relevant: list[str]
    graph_version: str
    cached: bool = Field(default=False, description="该合同此前已登记过，直接复用")
