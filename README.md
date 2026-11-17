# 面向企业合同合规审查的条款图多跳推理

在 **ACE**（Assessing Compliance in Enterprise）数据集上，把"跨条款合规判定"建模成**状态空间搜索**问题：
从场景文本 + 一组合同条款出发，在自建的 **facet 级条款图**上做多跳扩展（义务 → 例外 → 冲突 → 判定），
交付 **BFS / 加权 A\* / MCTS** 三套搜索内核、**GA 优化的参数化启发函数 h_θ**、**LLM 节点评估**与
**规则 + LightGBM + 启发搜索的混合架构**。

本项目同时提供可交互的 Web 审查台（Vue 3 + FastAPI），用于现场演示"给出结论，更给出可复核证据链"的完整链路。

> 本文档面向两类读者：
> 1. **报告撰写人员**：§1–§6 给出问题定义、方法、实验设计与结果口径，§8 给出可直接引用的产物索引；
> 2. **复现与工程评审人员**：§7 给出环境、数据、LLM 配置、一键复现与 HTTP/前端入口。

---

## 1. 项目概述

### 1.1 研究问题

给定一条企业合同场景描述（谁做了什么、缺少什么同意/通知、造成什么后果）与 1~4 条候选条款，
判定该场景属于以下三类之一，并给出支撑判断的条款证据链：

- `Compliant`：合规；
- `Non-Compliant`：不合规；
- `Not-Applicable`：不适用。

难点在于：判断往往不在单条条款内完成，而需要跨条款读取
**义务（OBL）→ 例外（EXC）/ 条件（COND）→ 定义（DEF）/ 引用（SCOPE）→ 冲突（CONFLICTS）→ 救济（REMEDY）**。
如果只做单条款分类，模型会在"读了主句、没看例外"时过早下结论；本项目因此把该任务显式建模为搜索。

### 1.2 核心贡献

| 编号 | 贡献 | 对应实现 |
|---|---|---|
| C1 | facet 级状态空间：节点不是条款，而是条款内语义单元，使 2~4 跳推理成为可能 | `state/core.py`、`data/facet_graph.py` |
| C2 | 三套搜索内核：BFS、加权 A\*、MCTS，共用同一状态抽象与目标测试 | `search/bfs.py`、`search/astar.py`、`search/mcts.py` |
| C3 | 参数化启发函数 h_θ 与遗传算法（GA）自动调参 | `heuristic/handcrafted.py`、`heuristic/ga_optimizer.py` |
| C4 | 规则专家系统：硬掩码 + 风险分（R1–R6） | `hybrid/rule_engine.py` |
| C5 | LightGBM 路径价值分类器（真实轨迹训练） | `hybrid/path_value_lgb.py`、`models/path_classifier.lgb` |
| C6 | LLM 只做三件事：动作合法性校验、路径 reward、事后解释；**不输出合规标签** | `llm/adapter.py`、`llm/explainer.py` |
| C7 | 八条件消融（B0/B1/A0/A1/A2/M0/M1/H）与失败案例分析 | `experiments/factory.py`、`results/` |

### 1.3 项目边界

- 本系统是**辅助审查工具**，输出用于复核与教学演示，不构成法律意见。
- 主表数字默认在 **mock LLM** 口径下产生，用于保证可复现；真实 LLM 通路需自行配置 key（见 §7.3）。
- ACE 原始数据不随代码仓库分发（`data/raw/` 被 `.gitignore` 忽略），需按 §7.2 放置。

---

## 2. 数据与任务定义

### 2.1 ACE 数据集

| 项目 | 规模 / 说明 |
|---|---|
| `data/raw/train.json` | 3,600 条带标签场景 |
| `data/raw/test.json` | 1,100 条测试场景 |
| `data/split.json` | 固定种子 42 的分层切分 |
| 训练/验证/留出 | `train_rest = 2,900`、`dev = 600`、`heldout = 100` |
| 每条样本 | `scenario_text` + `clauses`（`evidence_*`）+ `gd_tr` 标签 |
| 标签 | `Compliant` / `Non-Compliant` / `Not-Applicable` |

数据纪律：

- TF-IDF 词表与 IDF **只在 `train_rest` 上拟合**，dev/heldout/test 不参与，避免词汇泄漏；
- GA 适应度只看 dev，`heldout` 只用于过拟合监控，GA 完全不可见；
- **test 只在最终配置确定后跑一次**（`run_main_table --allow-test`）。

> 当前工作区已将 `train.json` / `test.json` 放入 `data/raw/`，因此可直接启动服务；
> 新克隆仓库时该目录不存在，需要自行放置（来源见 `Fujitsu-Assessing-Compliance-in-Enterprise-Dataset-main/`）。

### 2.2 评价指标

| 指标 | 定义 |
|---|---|
| `macro-F1` | 三分类宏平均 F1（主指标） |
| `accuracy` | 三分类准确率 |
| `avg_expanded` | 平均扩展节点数（预算效率） |
| `steps` | 平均推理步数（最终路径长度） |
| `redundant_ratio` | 扩展了但不在最终路径上的节点占比：`Σ(扩展数 − 路径节点数) / Σ扩展数` |
| `llm_calls` | 平均 LLM 调用次数 |
| `rule_prune` | 规则硬掩码平均剪枝数 |

---

## 3. 方法

### 3.1 facet 级状态空间

**节点（facet）**：每条条款按句切分为 1~5 个语义单元，角色包括
`OBL`（义务）、`EXC`（例外）、`COND`（条件）、`DEF`（定义）、`SCOPE`（引用）、`REMEDY`（救济）。

**状态**：

```
State = (已审查条款集合, 已使用边序列(带类型), 深度, 标签假设, 已访问 facet 序列, g)
```

**动作**：沿着一条**带类型的边**走到下一个 facet。
边类型包括 `DEFINES / COMPLEMENTS / EXCEPTIONS / DEPENDS_ON / CONFLICTS / SEQUENCE / CITES`；
根节点通过 `ENTRY` 动作进入图。

**转移**：目标 facet 所属条款并入状态，边记入 `used_edges`，深度 +1，每跳代价 `g += 1`。

**状态去重**：`State.key() = (条款集合, 边类型多重集)`。
不按"节点不可重访"剪枝——同一组条款经由"例外边"与"补充边"到达，法律含义不同。

**目标测试（证据充分性）**：

1. 覆盖所有强相关条款（与场景相似度 ≥ `0.6 × max_sim`）；
2. 若场景出现违约线索且图中存在例外/冲突节点，则例外必须已解（`needs_exc`）；
3. 终局决策头置信度 ≥ `tau_conf = 0.60`。

深度上限 `max_depth = 4`，扩展预算 `max_expansions = 16`（现场可通过 API 覆盖 1~48）。

### 3.2 条款图构建

建图规则（`data/facet_graph.py`，图版本 `facet-v2`）：

1. 条款内：主句单向连向本条其余 facet，边类型由目标角色决定；
2. 条款间：主句 → 主句，按目标条款首句线索判定例外/引用/顺序/条件/补充；
3. 冲突边：仅当两条款讲同一件事、共享 ≥2 个实词（或 1 个实词 + 显式冲突标记）且道义极性相反时连接；
4. 边权统一取目标条款与场景的相关度，`CONFLICTS` 不再写死为 1.0（避免搜索优先走冲突边）。

dev 600 条场景的图统计（`results/graph_stats_dev.json`）：

| 指标 | 数值 |
|---|---:|
| 平均 facet 节点数 | 3.61 |
| 平均边数 | 4.10 |
| 边总数 | 2,460 |

边类型占比：

| 边类型 | 占比 |
|---|---:|
| `EXCEPTIONS` | 36.5% |
| `COMPLEMENTS` | 17.9% |
| `CITES` | 17.7% |
| `CONFLICTS` | 12.9% |
| `DEPENDS_ON` | 9.3% |
| `SEQUENCE` | 5.0% |
| `DEFINES` | 0.65% |

### 3.3 三套搜索内核

#### BFS（B0 / B1）

按层扩展，目标是验证"盲目搜索在受限预算下能走多远"。
BFS 与 A\* 共用同一 `State.key()` 与目标测试，避免"A\* 因状态压缩更狠而显得更省"的伪优势。

#### 加权 A\*（A0 / A1 / A2 / H）

```
f(n) = g(n) + w_h · h_θ(n)
```

- `w_h = 1.5`；`w_h = 1` 时退化为标准 A\*；
- A2 / H 在优先级中加入融合修正：

```
f(n) = g(n) + w_h · h_θ(n)
       − λ1 · r_llm(n)
       − λ2 · p_clf(n)
       + λ4 · risk(n)
```

其中 `r_llm` 为 LLM 路径 reward，`p_clf` 为路径价值分类器给出的 P(正确路径)，`risk` 为规则风险分。
当前服务默认 `λ1 = 0.15`、`λ2 = 0.30`、`λ4 = 0.40`（`config.yaml`）。

#### MCTS（M0 / M1）

四阶段：Selection / Expansion / Rollout / Backpropagation。

- Selection：UCB1，`c = 1.3`；
- Expansion：**按边类型分层采样**，先挑已试次数最少的边类型，避免稀有 `EXCEPTIONS/CONFLICTS` 被 `COMPLEMENTS` 淹没；
- Rollout：限深随机策略，默认再走 2 跳；
- 终局 reward：`terminal_score = min(1, 0.75·conf + 0.15·coverage + 0.10·exception_bonus)`；
- Backprop：reward 截断到 `[0, 1]`，默认 mean 聚合（可切 median）；
- 模拟次数 `n_simulations = 90`。

### 3.4 参数化启发函数 h_θ 与 GA

`heuristic/handcrafted.py` 定义 4 维启发：

```
h_θ(n) = D · [ w1·(1 − ŝim) + w2·(1 − prior) + w3·(1 − exc) + w4·depth/D ]
         ────────────────────────────────────────────────────────────────
                          w1 + w2 + w3 + w4
```

- `ŝim`：当前状态已审查条款与场景的语义相似度（按场景可达最大值归一）；
- `prior`：前沿 facet 的角色先验，`1 − H(label | role)/log 3`，只在 `train_rest` 上统计；
- `exc`：例外信号（是否走过 EXCEPTIONS/CONFLICTS 边或到达 EXC/COND 节点）；
- `depth`：深度惩罚；
- `D = max_depth`，h 缩放到 `[0, D]`，与每跳成本 1 同量纲。

GA 配置与结果（`results/ga_history.json`）：

| 项目 | 值 |
|---|---|
| 染色体 | 4 维实数，范围 `[0, 2]` |
| 种群 / 代数 | 12 / 9 |
| 选择 / 交叉 / 变异 | 锦标赛 k=3；SBX `eta=15, pc=0.9`；高斯 `σ=0.12, pm=0.35` |
| 精英 | 1 |
| 适应度 | `dev macro-F1 − 0.5 · max(0, avg_expanded/B0_expanded − 1)` |
| 最优参数 | `w_sim=0.7366, w_prior=0.0, w_exc=0.9143, w_depth=0.0002` |
| dev 结果 | macro-F1 0.6444，平均扩展 5.103 |
| held-out 监控 | macro-F1 0.6972（100 条，GA 不可见） |
| 评估次数 | 106 |

### 3.5 规则专家系统（B1 / H / M1）

`hybrid/rule_engine.py` 提供 `mask(state, actions)` 与 `risk(state, action)`：

| 规则 | 作用 |
|---|---|
| R1 无关扩展阻断 | 目标条款相似度低于 floor 且已覆盖强相关条款时剪枝 |
| R2 深链截断 | `DEPENDS_ON` 连续引用链超过 2 跳时剪枝 |
| R3 冲突边双证据 | 冲突边的两端都必须已出现在状态中，否则延后 |
| R4 例外未解不下结论 | `needs_exc` 且例外未解时，禁止继续把预算花在义务条款 |
| R5 环检测 | 同一 `(src, edge_type, dst)` 已走过则剪枝 |
| R6 失效条款排除 | 命中"已被取代/不再有效"线索的条款不再作为推理依据（例外/冲突边放行） |

掩码全剪光时退化为不剪，保证搜索不会因规则空转。

### 3.6 路径价值分类器

- 模型：LightGBM（`models/path_classifier.lgb`），真实搜索轨迹训练；
- 训练规模：600 个场景、4,993 条路径；dev AUC **0.6878**，`best_iteration=13`；
- 18 维特征：深度、条款数、边类型计数、例外/条件占比、语义相似度、出入度、LLM reward、规则 risk 等；
- 缺失或不可用时回退逻辑回归 `results/models/path_classifier.pkl`。

### 3.7 终局决策头

`search/decision.py` 把"一条已审查路径"映射为三分类概率：

- 15 维 facet 级特征：`sim_obl`、`sim_exc`、`exc_rel`、覆盖度、违约线索、同意缺口、例外/定义/冲突标记等；
- 只用 `train_rest` 上随机游走采样的路径训练，dev/heldout/test 不参与；
- `argmax` 作为答案，最大概率作为置信度。

路径级诊断（`results/decision_path_eval.json`）：

| split | 路径数 | 场景数 | accuracy | macro-F1 |
|---|---:|---:|---:|---:|
| dev | 14,372 | 600 | 0.5373 | 0.5345 |
| heldout | 2,375 | 100 | 0.5941 | 0.5924 |

> 该表是**路径级决策头**的诊断，不是主表场景级指标，报告中不要与 §5 主表混用。

### 3.8 LLM 的三项职责（不输出标签）

| 职责 | 实现 | 约束 |
|---|---|---|
| 动作合法性校验 | `llm/adapter.py::LLMEvaluator.action_valid` | 只对语义复杂边（`EXCEPTIONS / CONFLICTS / DEPENDS_ON` 或例外节点）调用；输出 `{valid, confidence}` |
| 路径 reward | `LLMEvaluator.path_reward` | 输出 0~1 分，不输出标签；融合进 A\*/MCTS 优先级 |
| 事后解释 | `llm/explainer.py` | 输入已完成的搜索轨迹，只改写措辞；事实字段一律取自本地轨迹 |

硬约束：

1. 模型输出出现 `Compliant / Non-Compliant / Not-Applicable` 等标签词时，分数压到 0 并标记 `violation`；
2. 解释文本出现标签词时整段作废，回退模板解释，原文保留在 `raw` 供审计；
3. LLM 调用结果按 prompt 哈希缓存到 `results/cache/llm/`，重复状态不重复付费；
4. 未配置 key 时自动回退 `MockLLMEvaluator`，结果中的 `llm_mode` 会明确标记 `mock`。

LLM 误杀统计（`results/llm_reject_rate.json`，dev 600）：

| 条件 | LLM 驳回数 | 误杀数 | 误杀率 |
|---|---:|---:|---:|
| A2 | 1,385 | 275 | 19.9% |
| H | 1,177 | 253 | 21.5% |
| M1 | 5,635 | 718 | 12.7% |

### 3.9 混合架构 H

H = **A\* 搜索 + 规则硬掩码 + LightGBM 路径分类器 + GA 启发 h_θ + LLM 校验/reward**。

一次请求的装配顺序（`experiments/factory.py::build_problem`）：

```
场景 + 条款
  → facet 图构建
  → 规则掩码（R1–R6）
  → 启发 h_θ（GA 参数）
  → LLM 动作校验（复杂边）+ 路径 reward
  → 路径价值分类器 P(正确)
  → 融合优先级 f
  → A* / BFS / MCTS 搜索
  → 终局决策头三分类
  → 证据链解释（可选）
```

### 3.10 八条件装配

| 条件 | 内核 | 规则 | 启发 h | LLM | 分类器 | 说明 |
|---|---|---|---|---|---|---|
| B0 | BFS | 否 | 0 | 否 | 否 | 盲目基准 |
| B1 | BFS | 是 | 0 | 否 | 否 | 规则 + 盲目 |
| A0 | A\* | 否 | 手工 h | 否 | 否 | 固定手工启发 |
| A1 | A\* | 否 | GA h | 否 | 否 | GA 优化启发 |
| A2 | A\* | 否 | GA h | 是 | 否 | LLM 参与校验 + reward |
| M0 | MCTS | 否 | 0 | 否 | 否 | 蒙特卡洛基准 |
| M1 | MCTS | 是 | 0 | 是 | 是 | MCTS + 规则 + LLM + 分类器 |
| H | A\* | 是 | GA h | 是 | 是 | 规则 + 学习 + 启发混合 |

---

## 4. 实验设计

| 项目 | 设置 |
|---|---|
| dev 规模 | 600 |
| test 规模 | 1,100 |
| held-out | 100（GA 不可见） |
| 搜索预算 | `max_depth=4`，`max_expansions=16` |
| 目标置信度 | `tau_conf=0.60` |
| A\* 启发权重 | `w_h=1.5` |
| MCTS | 90 次模拟，`c=1.3`，rollout 深度 2 |
| 随机性 | 场景级稳定种子（crc32），保证进程无关复现 |
| 显著性口径 | 主表报告 macro-F1 / accuracy / 扩展节点 / 步数 / 冗余占比 / LLM 调用 / 规则剪枝 |

补充实验：

| 实验 | 命令 | 产物 |
|---|---|---|
| GA 曲线与参数 | `python -m experiments.run_ga` | `results/ga_history.json`、`results/figs/ga_curve.png` |
| λ 融合权重网格 | `python -m experiments.run_lambda_grid --limit 300` | `logs/lambda_grid.json` |
| 预算扫描 | `python -m experiments.run_hparam_sweep` | `results/hparam_sweep.csv`、`results/figs/hparam_landscape.png` |
| 按跳数/条款数分层 | `python -m experiments.run_stratified` | `results/stratified_*.md/csv` |
| LLM 幻觉率扫描 | `python -m experiments.run_hallucination_sweep` | `results/hallucination_sweep.*` |
| 失败案例挖掘 | `python -m experiments.run_failure_analysis` | `results/failure_cases.md/json` |
| 判据路径评估 | `python -m experiments.run_decision_eval` | `results/decision_path_eval.json` |
| LLM 误杀与成本 | `python -m tools.llm_reject_report`、`python -m tools.llm_cost_table` | `results/llm_reject_rate.*`、`results/llm_cost_table.md` |

> λ 网格在 300 条 dev 子集上多组并列 `macro-F1=0.6437`、扩展节点在 4.51~4.52 间抖动，
> 说明融合权重不是当前瓶颈；`config.yaml` 取 `λ1=0.15, λ2=0.30, λ4=0.40`，报告中应如实说明"平坦最优"。

---

## 5. 结果与消融

### 5.1 dev 主表（600 条）

| 条件 | macro-F1 | accuracy | 平均扩展 | 推理步数 | 冗余占比 | LLM 调用 | 规则剪枝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 BFS（盲目基准） | 0.6409 | 0.6483 | 5.55 | 1.77 | 0.501 | 0.0 | 0.0 |
| B1 BFS + 规则 | 0.6439 | 0.6500 | 5.09 | 1.81 | 0.448 | 0.0 | 2.3 |
| A0 A\* + 手工 h | 0.6471 | 0.6533 | 5.23 | 1.84 | 0.457 | 0.0 | 0.0 |
| A1 A\* + GA h | 0.6444 | 0.6500 | 5.10 | 1.88 | 0.437 | 0.0 | 0.0 |
| A2 A\* + GA + LLM | **0.6478** | 0.6517 | 4.74 | 1.86 | 0.396 | 20.2 | 0.0 |
| M0 MCTS（基准） | 0.6339 | 0.6433 | 9.34 | 1.86 | 0.694 | 0.0 | 0.0 |
| M1 MCTS + LLM + 分类器 | 0.6473 | 0.6517 | 8.00 | 1.95 | 0.631 | 52.3 | 10.4 |
| H 混合（规则+分类器+GA h+LLM） | 0.6461 | 0.6500 | **4.53** | 1.88 | **0.365** | 17.5 | 2.0 |

### 5.2 test 主表（1,100 条，一次性）

| 条件 | macro-F1 | accuracy | 平均扩展 | 推理步数 | 冗余占比 | LLM 调用 | 规则剪枝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6110 | 0.6073 | 6.21 | 1.86 | 0.540 | 0.0 | 0.0 |
| B1 | 0.6214 | 0.6145 | 5.77 | 1.93 | 0.493 | 0.0 | 2.7 |
| A0 | 0.6263 | 0.6182 | 5.96 | 1.94 | 0.506 | 0.0 | 0.0 |
| A1 | 0.6245 | 0.6164 | 5.85 | 1.99 | 0.489 | 0.0 | 0.0 |
| A2 | 0.6267 | 0.6164 | 5.35 | 1.99 | 0.441 | 22.2 | 0.0 |
| M0 | 0.6098 | 0.6073 | 9.55 | 1.95 | 0.692 | 0.0 | 0.0 |
| M1 | 0.6264 | 0.6164 | 8.19 | 2.09 | 0.623 | 51.7 | 10.1 |
| H | **0.6270** | 0.6164 | **5.06** | 2.01 | **0.404** | 18.2 | 2.2 |

### 5.3 如何正确解读结果

1. **H 的主要收益是搜索效率，不是 F1 绝对领先**：
   - dev 上 H 的 F1 为 0.6461，低于 A2（0.6478）、M1（0.6473）、A0（0.6471）；
   - 但 H 的平均扩展节点最少（dev 4.53 / test 5.06），冗余路径占比最低（0.365 / 0.404）。
2. **test 上 H 的 F1 最高（0.6270）**，但相对 A0 的增益仅 0.0007，应表述为"持平或略优"，不能声称显著提升。
3. **MCTS 的扩展代价约为 A\* 的两倍**（M0 9.34 vs H 4.53），在受限预算下不占优。
4. **规则与 LLM 都有代价**：M1 的 LLM 调用约 52 次/场景，H 约 17.5 次；LLM 误杀率 H 为 21.5%。
5. **4+ 跳仍是明显短板**：

| 分层 | dev H F1 / 扩展 | test H F1 / 扩展 |
|---|---:|---:|
| 1–2 跳 | 0.636 / 3.2 | 0.611 / 3.6 |
| 3 跳 | 0.620 / 8.3 | 0.602 / 8.4 |
| 4+ 跳 | 0.301 / 9.7 | 0.498 / 9.7 |

报告撰写建议：把 H 定位为"在相同 F1 水平下显著降低扩展预算与冗余路径"的混合架构，而不是"全面超越所有基线"。

### 5.4 图与表

| 产物 | 用途 |
|---|---|
| `results/figs/main_f1.png` | 八条件 macro-F1 对比 |
| `results/figs/main_expanded.png` | 八条件平均扩展节点对比 |
| `results/figs/ga_curve.png` | GA dev / held-out 适应度曲线 |
| `results/figs/hparam_landscape.png` | 预算与参数地形 |
| `results/figs/hallucination_curve.png` | LLM 幻觉率对指标的影响 |
| `results/figs/budget_sweep.png` | 扩展预算扫描 |

---

## 6. 失败分析与限制

`results/failure_cases.md` 汇总了 dev + test 的偏离案例：

| 现象 | 条数 |
|---|---:|
| A1 ≠ B0 且 B0 正确（启发误导） | 11 |
| H ≠ A1 且 A1 正确（LLM/融合引入错误） | 12 |
| 规则过约束（B1 ≠ B0 且 H 剪枝主导） | 8 |
| M0 ≠ A0 且 A0 正确（MCTS 采样误差） | 28 |

主要限制：

1. **4+ 跳场景 F1 仅约 0.30（dev）/ 0.50（test）**，多跳推理仍是瓶颈；
2. **规则可能过约束**：R4 在例外未解时禁止回到义务条款，部分场景会剪掉必要动作；
3. **LLM 会误杀**：H 的误杀率约 21.5%，主要发生在语义复杂边；
4. **单条款场景占比高**：dev 图中平均仅 3.61 个 facet，很多场景没有多跳空间，BFS 与 A\* 差距自然很小；
5. **中文合同**：TF-IDF 词表来自英文训练集，`data/features.py` 对含中文文本启用字级 bigram 余弦回退，角色判定使用中文句首线索，效果弱于英文数据；
6. **LLM 口径**：主表默认 mock；真实 LLM 结果需另跑 `tools/llm_smoke.py` 与 `run_* --llm real`，不能与 mock 主表混报；
7. **数据不随仓库分发**：`data/raw/` 被忽略，复现前必须自行放置。

---

## 7. 工程实现与复现

### 7.1 环境

要求 Python 3.10+（当前工作区实测 **3.12.5**），CPU 即可，无需 GPU。

```powershell
cd "d:\program\artificial intelligence"
# 当前工作区已有 venv；新环境可用：
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖：`numpy scipy scikit-learn pandas matplotlib networkx pyyaml tqdm lightgbm openai python-dotenv fastapi uvicorn`。

> Windows 中文注释坑：`requirements.txt` 必须保存为 **UTF-8 with BOM**，否则 pip 23.x 可能用 GBK 解码并报
> `UnicodeDecodeError: 'gbk' codec can't decode byte ...`。

### 7.2 数据准备

把 ACE 原始数据放到：

```
data/raw/train.json     # 3,600 条
data/raw/test.json      # 1,100 条
```

当前工作区已就位；如需重新切分（固定种子 42）：

```powershell
.\venv\Scripts\python.exe -m data.ace_loader
```

`data/split.json` 已随仓库提供；只要 `data/raw/*` 存在，切分步骤可跳过。

### 7.3 LLM 凭据配置（`.env`）

项目支持项目根目录的 `.env`（模板见 `.env.example`）：

```env
LLM_API_KEY=sk-你的key
# 可选：OpenAI 兼容网关
# LLM_BASE_URL=https://api.deepseek.com/v1
# LLM_MODEL=gpt-4o-mini
```

读取优先级：**进程环境变量 > `.env` > 代码默认值**。修改 `.env` 后需重启后端。

两条链路分开控制：

| 用途 | 控制方式 | 说明 |
|---|---|---|
| 搜索侧 A2 / M1 / H 的动作校验与 reward | `config.yaml` 的 `llm.mode: real` + key | 只设 key 不改 mode，搜索仍是 mock |
| 事后解释（审查台"推理说明"） | 前端选 `real` 或 `auto` | `auto` 有 key 用真实，否则模板 |

注意：`config.yaml` 的 `llm.model` 只用于成本统计；实际调用模型由 `LLM_MODEL` 环境变量或代码默认 `gpt-4o-mini` 决定。

真实通路冒烟（3 次调用，结果写入 `results/llm_smoke.md`）：

```powershell
.\venv\Scripts\python.exe -m tools.llm_smoke
```

### 7.4 一键复现（`run_all.bat`）

| 步骤 | 命令 | 主要产物 |
|---|---|---|
| 0 | 依赖自检 | — |
| 1 | `python -m data.ace_loader` | `data/split.json` |
| 2 | `python -m data.clause_graph_builder` + `python -m tools.audit_clause_graph` | 条款图审计 |
| 3 | `python -m experiments.train_lgb_classifier --out models\pc_new.lgb` | `models/path_classifier.lgb` |
| 4 | `python -m experiments.run_ga` | `results/ga_history.json`、`ga_curve.png` |
| 5 | `python -m experiments.run_lambda_grid --limit 300` | `logs/lambda_grid.json` |
| 6 | `python -m experiments.run_main_table --allow-test` | `results/main_table*.md`、`rows_*.csv`、`summary_*.json` |
| 7 | `python -m experiments.run_visuals` | `results/trees/*.html/json` |
| 8 | `run_failure_analysis` + `run_hparam_sweep` + `run_decision_eval` + `run_stratified` + `run_hallucination_sweep` | 失败案例、参数地形、分层、幻觉曲线 |
| 9 | `python -m tools.llm_reject_report` + `python -m tools.llm_cost_table` | LLM 误杀率与成本表 |
| 10 | `python -m tests.test_kernels` + `python -m tests.test_explainer` | 11 + 4 项单测 |
| 10b | `python -m tools.explain_case --preset samples` | `results/explanations/`、解释版搜索树 |
| 11 | `python -m tools.check_report_numbers` | 报告数字一致性校验 |

> `run_all.bat` 内部使用全局 `python`；若依赖装在 venv，请先激活 venv 或把命令替换为
> `.\venv\Scripts\python.exe`。

### 7.5 单步命令

```powershell
# 冒烟：10 条 dev，30 秒内
.\venv\Scripts\python.exe -m experiments.run_baselines --condition B0 --split dev --limit 10

# 单条件 / 单 split
.\venv\Scripts\python.exe -m experiments.run_baselines --condition A1 --split dev

# 条款图规模与边类型分布
.\venv\Scripts\python.exe -m tools.graph_stats --split dev

# 离线模板解释 / 真实 LLM 解释
.\venv\Scripts\python.exe -m tools.explain_case --sid tr00632 --condition H
.\venv\Scripts\python.exe -m tools.explain_case --sid tr00632 --condition H --mode real
```

`--limit N` 是固定种子随机抽样，不是取前 N 条；ACE 的 `train.json` 按标签排序，直接取前缀会造成严重采样偏差。

### 7.6 HTTP 服务

```powershell
.\venv\Scripts\python.exe -m api --port 8000
# 或：uvicorn api.main:app --port 8000
```

- 地址：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/docs`
- 前端（构建产物）：`http://127.0.0.1:8000/app`
- 服务启动秒回，语料后台加载（10~60 秒）；`GET /api/health` 的 `ready=true` 后再发重请求。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 就绪状态、场景数 4,700、图版本 `facet-v2`、LLM 模式 |
| GET | `/api/conditions` | 八个条件的装配说明 |
| GET | `/api/overview` | 总览页聚合数据 |
| GET | `/api/summary/{split}` | `summary_dev.json` / `summary_test.json` |
| GET | `/api/table/{name}` | 既有产物表（csv/json/markdown） |
| GET | `/api/scenarios` | 场景浏览器：split/condition/correct/min_hops/max_hops/n_clauses/q/limit/offset |
| GET | `/api/scenario/{sid}` | 场景原文 + 条款 + 八条件判定 |
| GET | `/api/scenario/{sid}/runs` | 只取八条件判定 |
| GET | `/api/graph/{sid}` | facet 条款图 |
| POST | `/api/scenarios` | 登记客户合同并建图，返回 `cus_*` sid |
| POST | `/api/search` | 现场跑一次搜索 |
| POST | `/api/explain` | 只生成解释 |
| GET | `/api/trees`、`/api/tree/{name}` | 三棵样例搜索树 |
| GET | `/api/llm/report` | 误杀率、成本表、真实 vs mock |

`POST /api/search` 的 `tree` 已做 `prune_frontier`，`meta` 带场景原文与全部条款，`explanation` 为 `null` 或解释结构。
搜索 CPU 密集，服务内部用 `Semaphore(1)` 串行执行。

### 7.7 前端

```powershell
cd frontend
npm install
npm run dev      # 开发模式，/api 代理到 127.0.0.1:8000
npm run build    # 构建到 frontend/dist，由 python -m api 挂到 /app
```

页面：

| 路径 | 页面 |
|---|---|
| `/app/workspace` | 工作台 |
| `/app/review/new` | 审查台：粘贴合同或选数据集场景 → 配置条件与预算 → 推理 |
| `/app/review/:sid` | 结果页：结论条 + 证据链 + 条款图 + 搜索树 + 轨迹表 + 解释 |
| `/app/scenarios` | 场景库 |
| `/app/benchmark` | 评测与主表 |
| `/app/llm` | LLM 治理 |
| `/app/system` | 系统与接口说明 |

客户合同：`POST /api/scenarios` 接收 `{scenario_text, clauses{evidence_1..10}, title, persist}`，
默认只在内存，`persist=true` 才落 `data/custom/`。中文合同支持字级 bigram 相似度回退与中文句首角色线索。

### 7.8 当前工作区状态（交付前自检）

- `data/raw/train.json`、`data/raw/test.json` 已就位；
- `models/path_classifier.lgb` 可加载，dev AUC 0.6878；
- `.env` / `.env.example` 已加入，未填 key 时自动 mock；
- 服务实测：`ready=true`、`n_scenarios=4700`、`graph_version=facet-v2`；
- 单测实测：`tests.test_kernels` 11 项通过，`tests.test_explainer` 4 项通过；
- 新建审查实测：示例合同建图 3 节点 / 6 边，H 条件搜索返回 `Non-Compliant`，约 0.2 秒。

---

## 8. 产物索引（报告引用对照）

| 报告章节 | 建议引用 |
|---|---|
| 数据与切分 | `data/split.json`、§2.1 |
| 条款图构建 | `data/facet_graph.py`、`results/graph_stats_dev.json`、§3.2 |
| 搜索建模 | `state/core.py`、§3.1、§3.3 |
| 启发函数与 GA | `results/ga_history.json`、`results/figs/ga_curve.png`、§3.4 |
| 规则专家系统 | `hybrid/rule_engine.py`、§3.5 |
| 路径价值分类器 | `models/path_classifier.lgb.meta.json`、§3.6 |
| LLM 治理 | `results/llm_reject_rate.json`、`results/llm_cost_table.md`、`results/llm_smoke.md`、§3.8 |
| 主表（dev/test） | `results/main_table.md`、`results/main_table_test.md`、§5.1–§5.2 |
| 消融图 | `results/figs/main_f1.png`、`results/figs/main_expanded.png` |
| 分层分析 | `results/stratified_dev.md`、`results/stratified_test.md`、§5.3 |
| 失败分析 | `results/failure_cases.md`、`results/failure_cases.json`、§6 |
| 搜索树可视化 | `results/trees/success.html`、`astar_misled.html`、`llm_misjudge.html` |
| 解释样例 | `results/explanations/*.json`、`results/trees/*_explained.html` |
| 参数敏感性 | `results/hparam_sweep.csv`、`results/figs/hparam_landscape.png` |
| 预算扫描 | `results/budget_sweep.csv`、`results/figs/budget_sweep.png` |
| 幻觉率扫描 | `results/hallucination_sweep.csv/.md`、`results/figs/hallucination_curve.png` |

> `results/trees/manifest.json` 中的 `json/html` 字段可能是历史机器的绝对路径；
> 报告与前端引用时请使用当前仓库 `results/trees/` 下的同名文件，不要直接依赖 manifest 中的绝对路径。

---

## 9. 目录结构

```
artificial-intelligence/
├── api/                  # FastAPI 服务层：main.py / service.py / schemas.py
├── data/                 # ACE 加载、切分、facet 条款图
│   ├── ace_loader.py
│   ├── facet_graph.py    # facet 节点 + 7 类边
│   ├── clause_graph.py   # 图数据结构
│   ├── clause_graph_builder.py
│   └── features.py       # 语义相似度、角色判定、中文回退
├── state/                # State / Action / 转移 / 目标测试
├── search/               # bfs.py / astar.py / mcts.py / decision.py / tree.py
├── heuristic/            # handcrafted.py（h_θ）/ ga_optimizer.py（GA）
├── hybrid/               # rule_engine.py / path_value_lgb.py / path_value.py / fusion.py
├── llm/                  # env.py / adapter.py / explainer.py / mock_evaluator.py / 旧版客户端
├── eval/                 # metrics.py / visualize_tree.py
├── experiments/          # factory.py / runner.py / run_*.py / setup.py
├── tools/                # explain_case.py / graph_stats.py / llm_smoke.py / check_report_numbers.py
├── tests/                # test_kernels.py（11）/ test_explainer.py（4）
├── models/               # path_classifier.lgb 及其 meta
├── results/              # 主表、图、树、解释、失败案例、模型缓存
├── frontend/             # Vue 3 + Vite 前端；dist 由 API 挂到 /app
├── config.yaml           # 搜索、MCTS、GA、融合权重、LLM 模式
├── .env.example          # LLM 凭据模板（.env 不入库）
├── requirements.txt
└── run_all.bat
```

---

## 10. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 页面上出现 `无法连接推理服务：Failed to fetch` | 后端未启动，或前端没有 `/api` 代理。用 `python -m api --port 8000` 启动，访问 `http://127.0.0.1:8000/app`；开发模式用 5173 的 Vite 代理 |
| `/api/health` 长期 `ready=false` | 语料加载中（首次 10~60 秒）；若 `error` 提示 `data/raw/train.json` 不存在，按 §7.2 放置数据 |
| 填了 `.env` 但仍是 mock | 服务未重启；或只设置了 key 但 `config.yaml` 里 `llm.mode` 仍为 `mock` |
| `pip install -r requirements.txt` 报 `UnicodeDecodeError: 'gbk'` | `requirements.txt` 丢了 BOM；保留 UTF-8 with BOM，或升级 pip |
| LightGBM 报 `Could not open D:\...\path_classifier.lgb` 后又加载成功 | 中文绝对路径的已知噪声，`path_value_lgb.py` 会用相对路径重试，不影响结果 |
| 图表中文变方框 | matplotlib 默认字体不含中文；图像标题已使用英文，日志 warning 不影响结果 |
| `--limit` 下 F1 只有 0.34 | 采样偏差；`--limit` 是固定种子随机抽样，不是取前 N 条 |
| `run_main_table` 报 `'NoneType' object has no attribute 'items'` | `eval/metrics.py::bucket_by` 必须返回 `dict[str, list[Row]]` |

---

## 11. 分工与接口契约

| | A（搜索算法线） | B（知识与评估线） |
|---|---|---|
| 职责 | 状态空间、BFS/A\*/MCTS、手工 h + GA、终局决策头、搜索侧实验与报告 | ACE 加载切分、条款图构建、LLM 封装、路径价值分类器、规则引擎、指标与可视化 |

接口按项目文档 §2.4 冻结：

- A 侧唯一出口：`SearchResult(answer/path/expanded_nodes/tree_root/trace)`；
- B 侧唯一入口：`Evaluator(action_valid/path_reward/success_prob)` 与 `rule_engine.mask/risk`。

---

## 12. 数据来源与声明

- 数据集：ACE（Assessing Compliance in Enterprise），原始文件见 `Fujitsu-Assessing-Compliance-in-Enterprise-Dataset-main/`；
- 本项目为课程/研究用途的辅助审查系统，输出不构成法律意见；
- 报告中的每个数字都应能在 `results/` 找到对应产物，`python -m tools.check_report_numbers` 会逐项比对。
