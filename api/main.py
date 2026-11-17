"""FastAPI 应用入口。

启动：
    python -m api                # 等价于 uvicorn api.main:app --host 127.0.0.1 --port 8000
    python -m api --port 8010 --reload
    uvicorn api.main:app --reload

文档：
    http://127.0.0.1:8000/docs    （自动 OpenAPI）

约定：
* 所有业务端点都在 `/api/...` 下，前端可直连；
* 若存在 `web/` 目录（前端静态页），自动挂到 `/`，实现"一个服务同时给页面与接口"；
* 服务启动即返回，语料在后台加载，`/api/health.ready` 为 True 后再发重请求。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (ConditionInfo, CustomScenarioIn, CustomScenarioOut,
                         ExplainRequest, GraphOut, Health, ScenarioList,
                         ScenarioOut, SearchRequest, SearchResponse, TableOut)
from api.service import NotReady, TABLES, service

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 后台加载语料与模型：不阻塞启动，避免前端一开就等 40 秒
    task = asyncio.create_task(asyncio.to_thread(service.warmup))
    yield
    task.cancel()


app = FastAPI(
    title="合同合规条款图多跳推理 · 搜索服务",
    version="1.0.0",
    description="ACE 数据集上的 BFS / A* / MCTS 搜索、GA 启发优化、LLM 评估与混合架构的统一 HTTP 接口。"
                "所有指标数据来自 results/ 既有产物，现场搜索走与主表实验完全相同的装配路径。",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 课程项目：本地静态页(http://localhost / file://)都要能连
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(NotReady)
async def not_ready_handler(request: Request, exc: NotReady):
    """语料还在加载：快速失败 + Retry-After，前端据此显示"暖机中"而不是转圈。"""
    return JSONResponse(status_code=503, content={"detail": str(exc)},
                        headers={"Retry-After": "5"})


# ---------------------------------------------------------------- 基础
@app.get("/api/health", response_model=Health, tags=["基础"])
def health():
    return service.health()


@app.get("/api/conditions", response_model=list[ConditionInfo], tags=["基础"])
def conditions():
    return service.conditions()


@app.get("/api/overview", tags=["基础"])
def overview():
    return service.overview()


@app.get("/api/summary/{split}", tags=["指标"])
def summary(split: str):
    if split not in ("dev", "test"):
        raise HTTPException(400, "split 只能是 dev 或 test")
    return service.table(f"summary_{split}")["data"]


@app.get("/api/table/{name}", response_model=TableOut, tags=["指标"])
def table(name: str):
    try:
        return service.table(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"产物不存在：{exc}") from exc


# ---------------------------------------------------------------- 场景
@app.get("/api/scenarios", response_model=ScenarioList, tags=["场景"])
def scenarios(split: str = "dev",
              condition: str | None = None,
              correct: bool | None = None,
              min_hops: int | None = Query(default=None, ge=0, le=8),
              max_hops: int | None = Query(default=None, ge=0, le=8),
              n_clauses: int | None = Query(default=None, ge=1, le=10),
              q: str | None = None,
              limit: int = Query(default=50, ge=1, le=500),
              offset: int = Query(default=0, ge=0)):
    if split not in ("dev", "test"):
        raise HTTPException(400, "split 只能是 dev 或 test")
    return service.list_scenarios(split=split, condition=condition, correct=correct,
                                  min_hops=min_hops, max_hops=max_hops,
                                  n_clauses=n_clauses, q=q, limit=limit, offset=offset)


@app.get("/api/scenario/{sid}", response_model=ScenarioOut, tags=["场景"])
def scenario(sid: str):
    try:
        return service.scenario(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/scenario/{sid}/runs", tags=["场景"])
def scenario_runs(sid: str):
    return scenario(sid)["runs"]


@app.post("/api/scenarios", response_model=CustomScenarioOut, status_code=201,
          tags=["场景"])
async def create_scenario(req: CustomScenarioIn):
    """登记客户自己的合同（等价于一条 ACE 样本：场景描述 + 1~10 条条款）。

    只建图、不跑搜索；前端拿到 sid 后再调 `/api/search`，与数据集场景共用同一
    条装配路径。默认只在内存，`persist=true` 才落 `data/custom/`。
    """
    try:
        return await asyncio.to_thread(
            service.create_scenario, req.scenario_text, req.clauses,
            req.title, req.label, req.persist)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/graph/{sid}", response_model=GraphOut, tags=["场景"])
def graph(sid: str):
    try:
        return service.graph(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- 搜索
@app.post("/api/search", response_model=SearchResponse, tags=["搜索"])
async def search(req: SearchRequest):
    try:
        # 搜索是 CPU 密集，放线程池避免阻塞事件循环
        return await asyncio.to_thread(
            service.search, req.sid, req.condition, req.explain, req.explain_mode,
            req.max_expansions, req.include_trace, req.include_tree)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/explain", tags=["搜索"])
async def explain(req: ExplainRequest):
    try:
        return await asyncio.to_thread(service.explain, req.sid, req.condition, req.mode)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- 树 / LLM
@app.get("/api/trees", tags=["案例"])
def trees():
    return service.trees()


@app.get("/api/tree/{name}", tags=["案例"])
def tree(name: str):
    try:
        return service.tree(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/llm/report", tags=["LLM"])
def llm_report():
    return service.llm_report()


@app.get("/api/tables", tags=["指标"])
def tables():
    return {k: v[0] for k, v in TABLES.items()}


@app.get("/", include_in_schema=False)
def index():
    if (DIST / "index.html").exists():
        return RedirectResponse("/app")
    web = ROOT / "web" / "index.html"
    if web.exists():
        return RedirectResponse("/web/")
    return RedirectResponse("/docs")


# ---------------------------------------------------------------- 前端（Vue 3）
# history 模式：/app 下的任意深链都要回 index.html，静态文件按真实路径返回。
@app.get("/app", include_in_schema=False)
@app.get("/app/{full_path:path}", include_in_schema=False)
def spa(full_path: str = ""):
    idx = DIST / "index.html"
    if not idx.exists():
        raise HTTPException(404, "前端尚未构建：cd frontend && npm install && npm run build")
    if full_path:
        target = (DIST / full_path).resolve()
        try:
            target.relative_to(DIST.resolve())
        except ValueError:                 # 防目录穿越
            raise HTTPException(404) from None
        if target.is_file():
            return FileResponse(target)
    return FileResponse(idx)


# 旧版静态证据册（存在才挂载；挂载点必须放在所有 API 路由之后）
if (ROOT / "web").exists():
    app.mount("/web", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
