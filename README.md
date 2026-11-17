# 面向企业合同合规审查的条款图多跳推理

在 **ACE**（Assessing Compliance in Enterprise）数据集上，把"跨条款合规判定"建模成**状态空间搜索**问题：
从场景文本 + 一组合同条款出发，在自建的 **facet 级条款图**上做多跳扩展（义务 → 例外 → 冲突 → 判定），
交付 **BFS / 加权 A\* / MCTS** 三套搜索内核、**GA 优化的启发函数**、**LLM 节点评估**与**规则 + 分类器混合架构**的最小可运行链路。

主表覆盖八个条件（B0 / B1 / A0 / A1 / A2 / M0 / M1 / H），在 dev 600 与 test 1,100 上报告 macro-F1、平均扩展节点、推理步数、冗余路径占比、LLM 调用次数。

- 搜索建模与实验结论：`docs/报告_成员A_搜索算法线.md`
- 主表：`results/main_table.md`（dev）、`results/main_table_test.md`（test）
- 搜索树可视化：`results/trees/*.html`（成功 / A\* 误导 / LLM 误判）
- 失败案例：`results/failure_cases.md`
- 按跳数分层：`results/stratified_dev.md`、`results/stratified_test.md`
- **推理自然语言解释**：`python -m tools.explain_case --sid tr00632 --condition H`（`llm/explainer.py`）
- **HTTP 服务（前端接口）**：`python -m api --port 8000`，文档 `http://127.0.0.1:8000/docs`（`api/`）

---

## 1. 环境

要求 Python 3.10+（本项目在 3.13.2 上验证）。

```bat
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖只有 CPU 版：`numpy scipy scikit-learn pandas matplotlib networkx pyyaml tqdm`，加上 H/M1 用的 `lightgbm`、真实 LLM 通路用的 `openai`、前端接口用的 `fastapi / uvicorn`。
条款图、启发式语义相似度、路径价值分类器、决策头全部在本仓库内实现；分类器优先加载 `models/path_classifier.lgb`（LightGBM，真实轨迹训练，dev AUC 0.6878），缺失时回退到逻辑回归 `results/models/path_classifier.pkl`。**不需要 GPU，也不需要下载任何模型权重。**

> **Windows + 中文注释的坑**：`requirements.txt` 带中文注释，若文件是 UTF-8 **无 BOM**，pip 23.x 在 Windows 会用 GBK 解码并抛
> `UnicodeDecodeError: 'gbk' codec can't decode byte ...`。本仓库的 `requirements.txt` 已存为 **UTF-8 with BOM**；
> 若你重写该文件，请保留 BOM，或先升级 pip（`python -m pip install -U pip`，26.x 起容错更好）。

## 2. 数据

ACE 数据不随仓库分发，需自己放到：

```
data/raw/train.json     # 3,600 条，含 clauses / scenario_text / gd_tr / DeepSeek_reasoning_trace
data/raw/test.json      # 1,100 条
```

```bat
:: 生成分层切分 train_rest 2900 / dev 600 / heldout 100（固定种子）
python -m data.ace_loader
```

`data/split.json` 已随仓库提供；只要 `data/raw/*` 存在，切分步骤可跳过。

## 3. 一键复现

```bat
run_all.bat
```

`run_all.bat` 依次执行（任一步失败立即停止）：

| 步骤 | 命令 | 产物 |
|---|---|---|
| 0 | 依赖自检 | — |
| 1 | `python -m data.ace_loader` | `data/split.json` |
| 2 | `python -m experiments.run_ga` | `results/models/ga_params.json`、`results/ga_history.json`、`results/figs/ga_curve.png` |
| 3 | `python -m experiments.run_main_table --allow-test` | `results/main_table.md`、`main_table_test.md`、`rows_*.csv`、`summary_*.json`、`figs/main_*.png` |
| 4 | `python -m experiments.run_visuals` | `results/trees/*.html`、`*.json`、`manifest.json` |
| 5 | `python -m experiments.run_failure_analysis` | `results/failure_cases.md`、`failure_cases.json` |
| 6 | `python -m experiments.run_hparam_sweep` | `results/hparam_sweep.csv`、`figs/hparam_landscape.png` |
| 7 | `python -m experiments.run_decision_eval` | `results/decision_path_eval.json` |
| 8 | `python -m experiments.run_stratified` | `results/stratified_dev.md`、`stratified_test.md`、`stratified_all.csv` |
| 9 | `python -m experiments.run_hallucination_sweep` | `results/hallucination_sweep.csv/.md`、`figs/hallucination_curve.png` |
| 10 | `python -m tests.test_kernels` | 11 passed |
| 11 | `python -m tools.check_report_numbers` | 报告数字与产物一致性 |

预计耗时（8 核 CPU）：GA 约 2 分钟（106 次适应度评估），主表约 1 分钟（dev 600 + test 1,100 × 8 条件），其余步骤各 10～30 秒。

## 4. 单步用法

```bat
:: 冒烟（10 条，30 秒内）：先确认链路通
python -m experiments.run_baselines --condition B0 --split dev --limit 10

:: 单条件 / 单 split
python -m experiments.run_baselines --condition A1 --split dev

:: 重新统计条款图规模与边类型分布
python -m tools.graph_stats --split dev
python -m tools.graph_stats --split test --limit 200
```

`--limit N` 一律是**固定种子随机抽样**，不是取前 N 条。
原因：ACE 的 `train.json` 是按标签排好序的，`sids[:N]` 会取到"清一色 Compliant"的前缀（dev 前 60 条全是 Compliant、前 300 条是 200 Compliant + 100 Non-Compliant），
F1 会系统性偏低到 0.34 左右。`experiments/runner.py::run_condition` 因此改成随机抽样（种子 = `ctx.seed*7919 + limit`）。

### 4.1 推理自然语言解释（课程第 3 项的第三件事）

`llm/explainer.py` 实现**事后解释**：输入是已经跑完的搜索轨迹（路径每一步的边类型、
目标 facet、role、g/h/f/conf、LLM reward、规则命中），输出是该证据链的中文说明，
**不参与搜索、不改答案**，只写进 `SearchResult.explanation`。

```bat
:: 离线模板解释（无 key、断网可用，答辩现场稳妥）
python -m tools.explain_case --sid tr00632 --condition H

:: 真实 LLM 解释（需 LLM_API_KEY；越权输出标签词会自动作废并回退模板）
python -m tools.explain_case --sid tr00632 --condition H --mode real

:: 三棵样例树批量生成 + 导出带解释卡片的搜索树 HTML
python -m tools.explain_case --preset samples --export-tree

:: 单测（4 条：模板完整性 / 标签越权拦截 / 事实字段不被模型篡改 / 写回不改结果）
python -m tests.test_explainer
```

输出落 `results/explanations/{sid}_{condition}.json`，字段：`text` / `steps` / `citations` /
`signals` / `sufficiency` / `mode` / `violation` / `llm_calls`。

**两条硬约束**（与 `llm/adapter.py` 的搜索侧约束一致，解释侧更严）：

1. 解释文本出现 `Compliant / Non-Compliant / Not-Applicable` → 整段丢弃、回退模板、
   `violation=true`，原文留在 `raw` 供审计（报告里据此统计"越权率"）。
2. 模型只许改写措辞：`steps / citations / signals / sufficiency` 一律取自本地轨迹，
   不采信模型自述 —— 防止"解释编造证据"。

### 4.2 HTTP 服务（供前端调用）

```bat
:: 启动（默认 127.0.0.1:8000；--reload 为开发模式）
python -m api --port 8000

:: 或直接 uvicorn
uvicorn api.main:app --port 8000
```

服务启动**秒回**，语料在后台加载（约 10 秒，取决于图缓存），`GET /api/health` 的
`ready=true` 后再发重请求。所有端点都在 `/api/` 下，自动生成 OpenAPI 文档在 `/docs`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 就绪状态、场景数 4,700、图版本 `facet-v2` |
| GET | `/api/conditions` | 8 个条件的装配说明（内核/规则/h/LLM/分类器） |
| GET | `/api/overview` | 总览页一次性取齐：dev/test 主表、图统计、三棵树、GA 摘要、λ 最优 |
| GET | `/api/summary/{split}` | `summary_dev.json` / `summary_test.json` |
| GET | `/api/table/{name}` | 20 张既有产物表（csv / json / markdown），名单见 `GET /api/tables` |
| GET | `/api/scenarios` | 场景浏览器：`split/condition/correct/min_hops/max_hops/n_clauses/q/limit/offset` |
| GET | `/api/scenario/{sid}` | 场景原文 + 全部条款 + 8 条件判定 |
| GET | `/api/scenario/{sid}/runs` | 只取 8 条件判定 |
| GET | `/api/graph/{sid}` | facet 条款图（节点带 role/text，边带类型/权重） |
| GET | `/api/trees`、`/api/tree/{name}` | 三棵样例搜索树（含 `explanation` 字段） |
| POST | `/api/search` | **现场跑一次搜索**：`{sid, condition, explain, explain_mode, max_expansions}` |
| POST | `/api/explain` | 只生成解释（`{sid, condition, mode}`） |
| GET | `/api/llm/report` | 误杀率 JSON + 成本表 / 真实 vs mock 对照 markdown |

`POST /api/search` 返回的 `tree` 已经过 `prune_frontier`（与 SVG 导出一致），
`meta` 里带场景原文与全部条款，`explanation` 为 `null` 或 §4.1 的结构。
搜索是 CPU 密集，服务内部用 `Semaphore(1)` 串行执行，并发请求会排队而不是互相拖慢。

### 4.3 前端（Vue 3 · ClauseScope）

正式前端在 `frontend/`，构建产物由本服务托管在 **`/app`**（旧版静态证据册仍在 `/web/`）：

```bat
:: 一次性安装（Node 18+）
cd frontend && npm install

:: 开发模式（/api 自动代理到 127.0.0.1:8000）
npm run dev

:: 构建产物到 frontend/dist（python -m api 会自动挂到 /app）
npm run build
```

页面结构：工作台 `/app/workspace`、审查台 `/app/review/new`（粘贴合同 / 数据集场景 → 配置条件与预算 → 现场推理）、
结果页 `/app/review/:sid?condition=H&budget=16&explain=1`（结论条 + 证据链 + 条款图 + 搜索树 + 轨迹表 + 解释，三视图联动）、
场景库、评测、LLM 治理、系统。

**客户自己的合同**：`POST /api/scenarios` 接收 `{scenario_text, clauses{evidence_1..10}, title, persist}`，
登记后返回 `sid`（`cus_xxxxxxxx`），随后照常走 `/api/search`。默认只在内存，`persist=true` 才落 `data/custom/`。
中文合同支持：TF-IDF 词表来自英文训练集，`data/features.py` 对含中文字符的文本启用**字级 bigram 余弦回退**，
`detect_role` 用中文句首线索判定例外/条件/引用角色（只对中文生效，不影响英文实验口径，`tests.test_kernels` 11 项全过）。

**就绪约定**：语料加载期间 `/api/search` 等重接口返回 `503 + Retry-After`，前端轮询 `/api/health` 并显示"加载语料中"，
就绪后自动发起推理。

## 5. 目录结构

```
artificial-intelligence/
├── data/                 # ACE 加载、切分、facet 级条款图
│   ├── ace_loader.py     #   train/test 读取、分层切分
│   ├── facet_graph.py    #   facet 节点 + 7 类边（DEFINES/COMPLEMENTS/EXCEPTIONS/DEPENDS_ON/CONFLICTS/SEQUENCE/CITES）
│   ├── clause_graph.py   #   图数据结构 ScenarioGraph / Edge
│   └── corpus.py         #   语料 + 语义相似度（TF-IDF）+ 图缓存
├── state/                # State / Action / 转移 / 终止
├── search/               # bfs.py / astar.py / mcts.py / decision.py（终局决策头）
├── heuristic/            # handcrafted.py（h_θ）、ga_optimizer.py（GA）
├── hybrid/               # rule_engine.py、path_value.py（路径价值分类器）、fusion.py
├── llm/                  # adapter.py（校验/reward）、explainer.py（自然语言解释）、mock_evaluator.py
├── api/                  # FastAPI 服务层：schemas.py / service.py / main.py（只读调用实验层）
├── eval/                 # metrics.py、visualize_tree.py
├── experiments/          # run_ga / run_main_table / run_visuals / run_failure_analysis /
│                         # run_hparam_sweep / run_decision_eval / run_stratified / runner
├── tools/                # check_report_numbers.py、graph_stats.py、explain_case.py
├── tests/                # test_kernels.py（11 条）、test_explainer.py（4 条）
├── docs/                 # 报告
├── results/              # 所有实验产物（见 §3）
└── run_all.bat
```

## 6. 实验纪律

- **test 只跑一次**：`run_main_table` 不加 `--allow-test` 时只跑 dev；最终配置确定后才加该开关。
- **GA 不看 test**：适应度只用 dev，过拟合监控用 held-out 100 条（GA 不可见）。
- **数字可追溯**：报告里每个数字都能在 `results/` 找到对应产物，`python -m tools.check_report_numbers` 会逐项比对。

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 图表中文变成方框、日志里一堆 `Glyph ... missing from font(s) DejaVu Sans` | matplotlib 默认字体不含中文；图像里的标题改用英文（代码已如此），日志中的 warning 不影响结果 |
| `python -m pip install -r requirements.txt` 报 `UnicodeDecodeError: 'gbk'` | `requirements.txt` 丢了 BOM，见 §1 |
| 日志出现 `[LightGBM] [Fatal] Could not open D:\...\models\path_classifier.lgb`，随后又 `Model loaded ...` | 中文绝对路径的已知噪声：LightGBM booster 先按原路径打开失败，`hybrid/path_value_lgb.py` 用 `os.path.relpath` 重试后成功加载。不影响结果 |
| `python -m api` 起来后 `/api/health` 长时间 `ready=false` | 语料在后台加载（首次无图缓存时约 30～60 秒）；看日志出现 `TF-IDF fitted` 即完成 |
| `run_main_table` 报 `'NoneType' object has no attribute 'items'` | `eval/metrics.py::bucket_by` 被改坏（返回 None）；应返回 `dict[str, list[Row]]` |
| `ImportError: cannot import name 'PathValueClassifier' from 'hybrid.path_classifier'` | 分类器在 `hybrid/path_value.py`（`experiments/setup.py` 已指向该模块） |
| `--limit` 下 F1 只有 0.34 | 采样偏差，见 §4；全量 dev 应为 0.63～0.65 |

## 8. 分工

| | A（搜索算法线） | B（知识与评估线） |
|---|---|---|
| 职责 | 状态空间、BFS/A\*/MCTS、手工 h + GA、终局决策头、搜索侧实验与报告 | ACE 加载切分、条款图构建、LLM 封装、路径价值分类器、规则引擎、指标与可视化 |

接口按项目文档 §2.4 冻结；A 侧以 `SearchResult(answer/path/expanded_nodes/tree_root/trace)` 为唯一出口，B 侧以 `Evaluator(action_valid/path_reward/success_prob)` 与 `rule_engine.mask/risk` 为唯一入口。
