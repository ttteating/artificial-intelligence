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

---

## 1. 环境

要求 Python 3.10+（本项目在 3.13.2 上验证）。

```bat
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖只有 CPU 版：`numpy scipy scikit-learn pandas matplotlib networkx pyyaml tqdm`。
条款图、启发式语义相似度、路径价值分类器、决策头全部在本仓库内用 TF-IDF/词袋 + 逻辑回归实现，**不依赖 sentence-transformers / torch / LightGBM**，因此不需要下载模型权重，也不需要 GPU。

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
├── llm/                  # 校验 / reward / 解释（磁盘缓存，未配置 key 时走确定性 mock）
├── eval/                 # metrics.py、visualize_tree.py
├── experiments/          # run_ga / run_main_table / run_visuals / run_failure_analysis /
│                         # run_hparam_sweep / run_decision_eval / run_stratified / runner
├── tools/                # check_report_numbers.py、graph_stats.py
├── tests/                # test_kernels.py（11 条）
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
| `run_main_table` 报 `'NoneType' object has no attribute 'items'` | `eval/metrics.py::bucket_by` 被改坏（返回 None）；应返回 `dict[str, list[Row]]` |
| `ImportError: cannot import name 'PathValueClassifier' from 'hybrid.path_classifier'` | 分类器在 `hybrid/path_value.py`（`experiments/setup.py` 已指向该模块） |
| `--limit` 下 F1 只有 0.34 | 采样偏差，见 §4；全量 dev 应为 0.63～0.65 |

## 8. 分工

| | A（搜索算法线） | B（知识与评估线） |
|---|---|---|
| 职责 | 状态空间、BFS/A\*/MCTS、手工 h + GA、终局决策头、搜索侧实验与报告 | ACE 加载切分、条款图构建、LLM 封装、路径价值分类器、规则引擎、指标与可视化 |

接口按项目文档 §2.4 冻结；A 侧以 `SearchResult(answer/path/expanded_nodes/tree_root/trace)` 为唯一出口，B 侧以 `Evaluator(action_valid/path_reward/success_prob)` 与 `rule_engine.mask/risk` 为唯一入口。
